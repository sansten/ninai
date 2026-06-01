"""
Memory Service
==============

Service for memory operations with integrated permission checking,
vector storage, and audit logging.

HYBRID MEMORY ARCHITECTURE:
- By default, memories start as short-term in Redis
- Frequently accessed or important memories are auto-promoted to long-term
- Use create_memory() for direct long-term storage (explicit)
- Use create_memory_smart() for hybrid auto-classification (recommended)
"""

import hashlib
import logging
import math
import re
import uuid
from typing import Optional, List, Union
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import QdrantService
from app.models.graph_relationship import GraphRelationship
from app.models.memory import MemoryMetadata, MemorySharing
from app.models.memory_feedback import MemoryFeedback
from app.services.embedding_service import EmbeddingService
from app.services.permission_checker import PermissionChecker, AccessDecision
from app.services.audit_service import AuditService
from app.services.short_term_memory import ShortTermMemory, ShortTermMemoryService
from app.services.memory_promoter import MemoryPromoter
from app.core.actor_context import normalize_actor_context
from app.services.identity_policy_service import ResolvedActorContext
from app.schemas.memory import (
    MemoryCreate,
    MemoryUpdate,
    MemorySearchRequest,
    MemoryShareRequest,
)


logger = logging.getLogger(__name__)


class MemoryService:
    """
    Memory operations service.
    
    Handles all memory CRUD operations with:
    - Permission checking before every operation
    - Dual write to Postgres (metadata) and Qdrant (vectors)
    - Audit logging for all operations
    - RLS-verified search results
    
    SECURITY: All vector search results are re-verified against
    Postgres RLS before being returned to the user.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        clearance_level: int = 0,
    ):
        """
        Initialize memory service.
        
        Args:
            session: Database session with tenant context set
            user_id: Current user's UUID
            org_id: Current organization's UUID
            clearance_level: User's security clearance level
        """
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.clearance_level = clearance_level
        
        self.permission_checker = PermissionChecker(session)
        self.audit_service = AuditService(session)
        self._last_search_diagnostics: dict[str, object] = {}

    @staticmethod
    def _normalize_utc_timestamp(value: Optional[datetime]) -> Optional[datetime]:
        """Normalize datetime values to timezone-aware UTC."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _tokenize_query_terms(text: str) -> list[str]:
        """Extract stable query terms from snippet text for enrichment variants."""
        if not text:
            return []
        parts = re.split(r"[^a-zA-Z0-9_]+", text.lower())
        stop = {
            "the", "and", "for", "with", "this", "that", "from", "into", "about", "have", "has",
            "was", "were", "will", "would", "should", "could", "not", "you", "your", "their", "they",
            "our", "are", "but", "can", "all", "any", "use", "using", "new", "old", "what", "when",
            "where", "why", "how", "who", "which", "than", "then", "also", "more", "less", "into",
        }
        out: list[str] = []
        seen: set[str] = set()
        for p in parts:
            if len(p) < 4 or p.isdigit() or p in stop:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    @staticmethod
    def _retrieval_learning_prior(memory: "MemoryMetadata") -> tuple[float, int]:
        """Extract org-level retrieval prior learned from applied feedback."""
        meta = getattr(memory, "extra_metadata", {}) or {}
        learning = meta.get("retrieval_learning") if isinstance(meta, dict) else {}
        if not isinstance(learning, dict):
            return 0.0, 0

        try:
            score = float(learning.get("relevance_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        try:
            count = int(learning.get("relevance_feedback_count") or 0)
        except (TypeError, ValueError):
            count = 0

        return max(-1.0, min(1.0, score)), max(0, count)

    @staticmethod
    def _is_complex_query(query: str) -> bool:
        """Heuristic complexity detector for adaptive retrieval overfetch."""
        q = str(query or "").strip().lower()
        if not q:
            return False
        tokens = re.findall(r"[a-z0-9_]+", q)
        if len(tokens) >= 10:
            return True
        markers = {
            "before",
            "after",
            "between",
            "during",
            "while",
            "compared",
            "difference",
            "versus",
            "vs",
            "because",
            "reason",
            "caused",
            "then",
            "first",
            "second",
        }
        return any(m in tokens for m in markers)

    @staticmethod
    def _query_rewrite_aliases() -> dict[str, str]:
        """Small alias map used for retrieval-time query expansion."""
        return {
            "usa": "united states",
            "us": "united states",
            "u.s.": "united states",
            "uk": "united kingdom",
            "u.k.": "united kingdom",
            "ai": "artificial intelligence",
        }

    def _build_query_variants(self, base_query: str, max_variants: int) -> list[str]:
        """Build lightweight lexical variants to improve semantic recall."""
        q = str(base_query or "").strip()
        if not q or max_variants <= 0:
            return []

        variants: list[str] = []
        q_norm = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9_\-\s]", " ", q)).strip()
        if q_norm and q_norm.lower() != q.lower():
            variants.append(q_norm)

        aliases = self._query_rewrite_aliases()
        tokens = re.findall(r"[a-zA-Z0-9_.-]+", q.lower())
        alias_terms = [aliases[t] for t in tokens if t in aliases]
        if alias_terms:
            variants.append((q + " " + " ".join(alias_terms)).strip())

        key_terms = self._tokenize_query_terms(q)
        if key_terms:
            variants.append((q + " " + " ".join(key_terms[:4])).strip())

        deduped: list[str] = []
        seen: set[str] = {q.lower()}
        for v in variants:
            lv = v.lower().strip()
            if not lv or lv in seen:
                continue
            seen.add(lv)
            deduped.append(v)
            if len(deduped) >= max_variants:
                break
        return deduped

    @staticmethod
    def _build_multihop_subqueries(base_query: str, max_subqueries: int = 3) -> list[str]:
        """Split complex questions into focused sub-queries for multi-hop retrieval."""
        q = str(base_query or "").strip()
        if not q or max_subqueries <= 0:
            return []

        parts = re.split(r"\b(?:and then|then|before|after|because|while|whereas|and)\b", q, flags=re.IGNORECASE)
        out: list[str] = []
        seen: set[str] = {q.lower()}

        for raw in parts:
            p = re.sub(r"\s+", " ", str(raw or "")).strip(" ,.;:-")
            if len(p) < 8:
                continue
            lp = p.lower()
            if lp in seen:
                continue
            seen.add(lp)
            out.append(p)
            if len(out) >= max_subqueries:
                break
        return out

    @staticmethod
    def _query_persona_hints(query: str) -> set[str]:
        """Extract likely person/entity hints from query text for perspective reranking."""
        q = str(query or "")
        hints: set[str] = set()
        for m in re.findall(r"\b[A-Z][a-z]{2,}\b", q):
            hints.add(m.lower())
        return hints

    @staticmethod
    def _memory_session_key(memory: "MemoryMetadata") -> str:
        """Best-effort session key extraction from memory metadata."""
        meta = getattr(memory, "extra_metadata", {}) or {}
        if isinstance(meta, dict):
            for key in ("session_id", "session", "thread_id", "conversation_id", "conv_id", "run_tag"):
                val = meta.get(key)
                if val:
                    return str(val)
        return ""

    @staticmethod
    def _memory_speaker_key(memory: "MemoryMetadata") -> str:
        """Best-effort speaker/actor extraction for perspective-aware reranking."""
        meta = getattr(memory, "extra_metadata", {}) or {}
        if isinstance(meta, dict):
            for key in ("speaker", "actor", "author", "participant"):
                val = meta.get(key)
                if val:
                    return str(val).lower()
        for attr in ("write_actor_id", "write_role"):
            val = getattr(memory, attr, None)
            if val:
                return str(val).lower()
        # Fallback for conversation-style memories where speaker is prefixed in content.
        preview = str(getattr(memory, "content_preview", "") or "")
        m = re.match(r"^\s*\[(?P<speaker>[^\]]+)\]", preview)
        if m:
            return str(m.group("speaker") or "").strip().lower()
        return ""

    @staticmethod
    def _query_overlap_score(query: str, memory: "MemoryMetadata") -> float:
        """Compute lexical overlap between query terms and memory text/tags.

        Returns a value in [0, 1] representing weighted query-term coverage.
        """
        stop = {
            "the", "a", "an", "is", "was", "did", "do", "what", "when", "where", "who", "how",
            "and", "or", "of", "in", "on", "to", "for", "at", "i", "my", "me", "we", "our",
            "you", "your", "he", "she", "it", "they", "their", "that", "this", "these", "those",
            "be", "been", "have", "has", "had", "will", "would", "could", "should", "may", "might",
            "about", "with", "from", "are", "were", "any", "some", "which", "also",
        }

        q_terms = {
            t for t in re.findall(r"[a-z0-9_]+", str(query or "").lower())
            if t and t not in stop and len(t) > 2
        }
        if not q_terms:
            return 0.0

        title = str(getattr(memory, "title", "") or "")
        content = str(getattr(memory, "content_preview", "") or "")
        tags = getattr(memory, "tags", None) or []
        haystack = " ".join([title, content, " ".join(str(t) for t in tags if t)])
        h_terms = set(re.findall(r"[a-z0-9_]+", haystack.lower()))
        if not h_terms:
            return 0.0

        overlap = q_terms & h_terms
        if not overlap:
            return 0.0

        # Weight rarer/longer terms slightly higher to prefer specific factual matches.
        numer = sum(1.0 + (0.15 if len(t) >= 8 else 0.0) for t in overlap)
        denom = sum(1.0 + (0.15 if len(t) >= 8 else 0.0) for t in q_terms)
        return max(0.0, min(1.0, numer / (denom or 1.0)))

    @staticmethod
    def _infer_dominant_session(memories: list["MemoryMetadata"], vector_scores: dict[str, float]) -> str:
        """Infer dominant session among candidates weighted by vector score."""
        score_by_session: dict[str, float] = {}
        for memory in memories:
            sid = MemoryService._memory_session_key(memory)
            if not sid:
                continue
            sid = sid.lower()
            score_by_session[sid] = score_by_session.get(sid, 0.0) + float(vector_scores.get(str(memory.id), 0.0))
        if not score_by_session:
            return ""
        return max(score_by_session.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _should_auto_hybrid(vector_scores: dict[str, float], request: MemorySearchRequest) -> bool:
        """Decide whether lexical retrieval should auto-activate due to sparse semantic recall."""
        if getattr(request, "hybrid", False):
            return True
        if not bool(getattr(settings, "SEARCH_AUTO_HYBRID_ENABLED", True)):
            return False

        min_hits = int(getattr(settings, "SEARCH_AUTO_HYBRID_MIN_VECTOR_HITS", 4) or 4)
        max_peak = float(getattr(settings, "SEARCH_AUTO_HYBRID_MAX_TOP_SCORE", 0.55) or 0.55)
        if len(vector_scores) < min_hits:
            return True
        top_score = max(vector_scores.values(), default=0.0)
        if top_score <= max_peak:
            return True
        return False

    def get_last_search_diagnostics(self) -> dict[str, object]:
        """Return diagnostics produced by the most recent search call."""
        return dict(self._last_search_diagnostics or {})

    async def _load_graph_neighbor_scores(self, seed_memory_ids: list[str], limit: int) -> dict[str, float]:
        """Collect neighboring memory ids from graph edges with similarity-based scores."""
        if not seed_memory_ids or limit <= 0:
            return {}

        try:
            org_uuid = uuid.UUID(str(self.org_id))
        except Exception:
            return {}

        stmt = (
            select(
                GraphRelationship.from_memory_id,
                GraphRelationship.to_memory_id,
                GraphRelationship.similarity_score,
                GraphRelationship.relationship_type,
            )
            .where(
                GraphRelationship.organization_id == org_uuid,
                or_(
                    GraphRelationship.from_memory_id.in_(seed_memory_ids),
                    GraphRelationship.to_memory_id.in_(seed_memory_ids),
                ),
            )
            .order_by(GraphRelationship.similarity_score.desc().nullslast())
            .limit(max(limit * 4, 32))
        )

        rows = (await self.session.execute(stmt)).all()
        if not rows:
            return {}

        seed_set = set(seed_memory_ids)
        neighbors: dict[str, float] = {}
        type_weight = {
            "DEPENDS_ON": 1.0,
            "REFINES": 0.95,
            "REFERENCES": 0.9,
            "RELATES_TO": 0.85,
            "CONTRADICTS": 0.7,
        }

        for from_id, to_id, similarity, relationship_type in rows:
            from_s = str(from_id)
            to_s = str(to_id)
            if from_s in seed_set and to_s not in seed_set:
                neighbor_id = to_s
            elif to_s in seed_set and from_s not in seed_set:
                neighbor_id = from_s
            else:
                continue

            base = float(similarity or 0.5)
            rel = str(relationship_type or "RELATES_TO").upper()
            weighted = max(0.0, min(1.0, base * type_weight.get(rel, 0.85)))
            prev = neighbors.get(neighbor_id, 0.0)
            if weighted > prev:
                neighbors[neighbor_id] = weighted

        if len(neighbors) <= limit:
            return neighbors
        return dict(sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)[:limit])

    async def _build_graph_query_variants(
        self,
        base_query: str,
        graph_neighbor_scores: dict[str, float],
        max_variants: int,
    ) -> list[str]:
        """Build enriched query variants from graph-linked memory snippets."""
        if not base_query or max_variants <= 0 or not graph_neighbor_scores:
            return []

        neighbor_ids = [mid for mid, _ in sorted(graph_neighbor_scores.items(), key=lambda kv: kv[1], reverse=True)[:12]]
        stmt = select(MemoryMetadata.title, MemoryMetadata.content_preview, MemoryMetadata.tags).where(
            MemoryMetadata.organization_id == self.org_id,
            MemoryMetadata.is_active.is_(True),
            MemoryMetadata.id.in_(neighbor_ids),
        )
        rows = (await self.session.execute(stmt)).all()

        terms: list[str] = []
        seen: set[str] = set()
        for title, content_preview, tags in rows:
            snippets = [str(title or ""), str(content_preview or "")[:240]]
            if isinstance(tags, list):
                snippets.extend([str(t) for t in tags[:6] if t])
            for token in self._tokenize_query_terms(" ".join(snippets)):
                if token in seen:
                    continue
                seen.add(token)
                terms.append(token)
                if len(terms) >= max(6, max_variants * 4):
                    break
            if len(terms) >= max(6, max_variants * 4):
                break

        if not terms:
            return []

        variants: list[str] = []
        chunk_size = 4
        for i in range(0, len(terms), chunk_size):
            chunk = terms[i : i + chunk_size]
            if not chunk:
                continue
            variants.append((base_query + " " + " ".join(chunk)).strip())
            if len(variants) >= max_variants:
                break
        return variants
    
    # =========================================================================
    # Create
    # =========================================================================
    
    async def create_memory(
        self,
        data: MemoryCreate,
        embedding: List[float],
        request_id: Optional[str] = None,
        actor_ctx: Optional[ResolvedActorContext] = None,
    ) -> MemoryMetadata:
        """
            # Determine TTL: query param > body.ttl > default
            effective_ttl = ttl if ttl is not None else getattr(data, 'ttl', None)
        
        Writes metadata to Postgres and embedding to Qdrant.
        
        Args:
            stm = await stm_service.store(
                content=data.content,
                title=data.title,
                scope=data.scope,
                tags=data.tags,
                entities=data.entities,
                metadata=data.extra_metadata,
                ttl=effective_ttl,
            )
            PermissionError: If user lacks permission to create
        """
        # Check permission
        permission_check = await self.permission_checker.check_permission(
            self.user_id, self.org_id, f"memory:create:{data.scope}"
        )
        
        if not permission_check.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id="",
                operation="create",
                success=False,
                error_message=permission_check.reason,
            )
            raise PermissionError(permission_check.reason)
        
        # Generate IDs
        memory_id = str(uuid4())
        vector_id = str(uuid4())
        
        # Compute content hash for deduplication
        content_hash = hashlib.sha256(
            data.content.encode("utf-8")
        ).hexdigest()
        
        # Normalize temporal metadata for downstream time-series analysis.
        write_ts = datetime.now(timezone.utc)
        occurred_at = self._normalize_utc_timestamp(data.occurred_at)
        if actor_ctx is not None:
            writer_ctx = {
                "actor_id": actor_ctx.actor_id or "anonymous",
                "actor_type": actor_ctx.actor_type or "anonymous",
                "role": actor_ctx.role or "anonymous",
                "responsibility": "",
            }
        else:
            writer_ctx = normalize_actor_context(
                actor_id=None,
                actor_type=None,
                role=None,
                responsibility=None,
            )
        extra_metadata = dict(data.extra_metadata or {})
        extra_metadata.setdefault("written_at", write_ts.isoformat())
        extra_metadata.setdefault("write_actor_id", writer_ctx["actor_id"])
        extra_metadata.setdefault("write_actor_type", writer_ctx["actor_type"])
        extra_metadata.setdefault("write_role", writer_ctx["role"])
        extra_metadata.setdefault("write_responsibility", writer_ctx["responsibility"])
        if actor_ctx is not None:
            if actor_ctx.department:
                extra_metadata["write_department"] = actor_ctx.department
            if actor_ctx.mode_applied:
                extra_metadata["write_identity_mode"] = actor_ctx.mode_applied
        if occurred_at is not None:
            extra_metadata.setdefault("event_time", occurred_at.isoformat())
            extra_metadata.setdefault("event_date", occurred_at.date().isoformat())

        # Create metadata record
        memory = MemoryMetadata(
            id=memory_id,
            organization_id=self.org_id,
            owner_id=self.user_id,
            scope=data.scope,
            scope_id=data.scope_id,
            memory_type=data.memory_type,
            classification=data.classification,
            required_clearance=data.required_clearance or 0,
            title=data.title,
            content_preview=data.content[:2000],
            content_hash=content_hash,
            tags=data.tags or [],
            entities=data.entities or {},
            extra_metadata=extra_metadata,
            source_type=data.source_type,
            source_id=data.source_id,
            vector_id=vector_id,
            embedding_model=settings.EMBEDDING_MODEL or "text-embedding-3-small",
            retention_days=data.retention_days,
        )
        
        # Save to Postgres
        self.session.add(memory)
        await self.session.flush()
        await self._ensure_search_vector(
            memory_id=memory_id,
            title=data.title,
            content=data.content,
            tags=data.tags,
        )
        # No second flush needed — _ensure_search_vector uses session.execute()
        # which sends SQL immediately; the final db.commit() flushes the rest.

        # Save to Qdrant
        try:
            await QdrantService.upsert_memory(
                memory_id=vector_id,
                org_id=self.org_id,
                vector=embedding,
                payload={
                    "memory_id": memory_id,
                    "scope": data.scope,
                    "scope_id": data.scope_id,
                    # Denormalized for Qdrant filtering convenience
                    "team_id": data.scope_id if str(data.scope) == "team" else None,
                    "owner_id": self.user_id,
                    "tags": data.tags or [],
                    "classification": data.classification,
                    "memory_type": data.memory_type,
                    "created_at": write_ts.isoformat(),
                    "event_time": occurred_at.isoformat() if occurred_at else None,
                    "write_actor_id": writer_ctx["actor_id"],
                    "write_actor_type": writer_ctx["actor_type"],
                    "write_role": writer_ctx["role"],
                },
            )
        except Exception as exc:
            # Keep writes available even if vector infrastructure is degraded.
            logger.warning(
                "Qdrant upsert failed for memory_id=%s org_id=%s: %s",
                memory_id,
                self.org_id,
                exc,
            )
        
        # Audit log
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="create",
            success=True,
            details={
                "scope": data.scope,
                "classification": data.classification,
            },
        )
        
        return memory

    async def _ensure_search_vector(
        self,
        memory_id: str,
        title: Optional[str],
        content: str,
        tags: Optional[List[str]],
    ) -> None:
        """Fallback FTS indexing when database trigger is not present."""
        # Some unit tests use lightweight session doubles without SQL execution.
        if not hasattr(self.session, "execute"):
            return

        doc = " ".join(
            part for part in [title or "", content or "", " ".join(tags or [])] if part
        )
        if not doc:
            return

        try:
            await self.session.execute(
                text(
                    """
                    UPDATE memory_metadata
                    SET search_vector = to_tsvector('english', :doc)
                    WHERE id = :memory_id
                    """
                ),
                {"doc": doc, "memory_id": memory_id},
            )
        except SQLAlchemyError as exc:
            # Gracefully handle environments where the FTS migration is not applied yet.
            logger.warning(
                "Skipping search_vector update for memory_id=%s due to DB error: %s",
                memory_id,
                exc,
            )
    
    async def create_memory_smart(
        self,
        data: MemoryCreate,
        embedding: Optional[List[float]] = None,
        request_id: Optional[str] = None,
        force_long_term: bool = False,
        ttl: Optional[int] = None,
        actor_ctx: Optional[ResolvedActorContext] = None,
    ) -> Union[ShortTermMemory, MemoryMetadata]:
        """
        Create a memory using the hybrid architecture.
        
        By default, memories start as short-term in Redis and are
        automatically promoted to long-term when:
        - Accessed frequently (3+ times)
        - Content is detected as important (orders, preferences, etc.)
        
        Args:
            data: Memory creation data
            embedding: Optional pre-computed embedding vector
            request_id: Request ID for audit correlation
            force_long_term: If True, skip short-term and store directly in long-term
        
        Returns:
            ShortTermMemory (in Redis) or MemoryMetadata (if force_long_term or auto-promoted)
        """
        # Check permission
        permission_check = await self.permission_checker.check_permission(
            self.user_id, self.org_id, f"memory:create:{data.scope}"
        )
        
        if not permission_check.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id="",
                operation="create_smart",
                success=False,
                error_message=permission_check.reason,
            )
            raise PermissionError(permission_check.reason)
        
        # If explicitly requesting long-term via force flag, use traditional create
        # The smart endpoint defaults to short-term unless force_long_term=True
        if force_long_term:
            if embedding is None:
                embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
            return await self.create_memory(data, embedding, request_id, actor_ctx=actor_ctx)
        
        # Create short-term memory in Redis
        stm_service = ShortTermMemoryService(self.user_id, self.org_id)
        if actor_ctx is not None:
            writer_ctx = {
                "actor_id": actor_ctx.actor_id or "anonymous",
                "actor_type": actor_ctx.actor_type or "anonymous",
                "role": actor_ctx.role or "anonymous",
                "responsibility": "",
            }
        else:
            writer_ctx = normalize_actor_context(
                actor_id=None,
                actor_type=None,
                role=None,
                responsibility=None,
            )
        smart_metadata = dict(data.extra_metadata or {})
        write_ts = datetime.now(timezone.utc)
        occurred_at = self._normalize_utc_timestamp(data.occurred_at)
        smart_metadata.setdefault("written_at", write_ts.isoformat())
        smart_metadata.setdefault("write_actor_id", writer_ctx["actor_id"])
        smart_metadata.setdefault("write_actor_type", writer_ctx["actor_type"])
        smart_metadata.setdefault("write_role", writer_ctx["role"])
        smart_metadata.setdefault("write_responsibility", writer_ctx["responsibility"])
        if occurred_at is not None:
            smart_metadata.setdefault("event_time", occurred_at.isoformat())
            smart_metadata.setdefault("event_date", occurred_at.date().isoformat())
        
        stm = await stm_service.store(
            content=data.content,
            title=data.title,
            scope=data.scope,
            tags=data.tags,
            entities=data.entities,
            metadata=smart_metadata,
            ttl=ttl if ttl is not None else getattr(data, "ttl", None),
        )
        
        # Log creation
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=stm.id,
            operation="create_short_term",
            success=True,
            details={
                "scope": data.scope,
                "importance_score": stm.importance_score,
                "promotion_eligible": stm.promotion_eligible,
            },
        )
        
        # If immediately eligible for promotion, promote now
        if stm.promotion_eligible:
            promoter = MemoryPromoter(self.session, self.user_id, self.org_id)
            if embedding is None:
                embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
            promoted = await promoter.promote_memory(
                stm, 
                embedding=embedding,
                keep_in_cache=True,  # Keep in Redis for fast access
                promotion_reason="importance",
            )
            return promoted
        
        return stm
    
    async def get_memory_smart(
        self,
        memory_id: str,
        request_id: Optional[str] = None,
    ) -> Optional[Union[ShortTermMemory, MemoryMetadata]]:
        """
        Get a memory from either short-term (Redis) or long-term (PostgreSQL).
        
        Checks Redis first, then PostgreSQL. Accessing short-term memories
        may trigger automatic promotion if access count threshold is met.
        
        Args:
            memory_id: Memory UUID
            request_id: Request ID for audit correlation
        
        Returns:
            ShortTermMemory, MemoryMetadata, or None if not found
        """
        # Try short-term first (faster)
        stm_service = ShortTermMemoryService(self.user_id, self.org_id)
        stm = await stm_service.get(memory_id)
        
        if stm:
            # Check if now eligible for promotion due to access count
            if stm.promotion_eligible and stm.access_count >= ShortTermMemoryService.ACCESS_COUNT_THRESHOLD:
                promoter = MemoryPromoter(self.session, self.user_id, self.org_id)
                promoted = await promoter.check_and_promote(stm)
                if promoted:
                    return promoted  # Return the long-term version
            return stm
        
        # Fall back to long-term storage
        return await self.get_memory(memory_id, request_id)
    
    async def list_all_memories(
        self,
        include_short_term: bool = True,
        request_id: Optional[str] = None,
    ) -> dict:
        """
        List all memories for the user, from both short-term and long-term storage.
        
        Args:
            include_short_term: Whether to include short-term memories from Redis
            request_id: Request ID for audit correlation
        
        Returns:
            Dict with 'short_term' and 'long_term' memory lists
        """
        from sqlalchemy import select
        
        result = {
            "short_term": [],
            "long_term": [],
        }
        
        # Get short-term memories from Redis
        if include_short_term:
            stm_service = ShortTermMemoryService(self.user_id, self.org_id)
            result["short_term"] = await stm_service.list_user_memories()
        
        # Get long-term memories from PostgreSQL
        query = (
            select(MemoryMetadata)
            .where(MemoryMetadata.organization_id == self.org_id)
            .where(MemoryMetadata.is_active == True)
            .order_by(MemoryMetadata.created_at.desc())
            .limit(100)
        )
        db_result = await self.session.execute(query)
        result["long_term"] = list(db_result.scalars().all())
        
        return result

    async def list_memories(
        self,
        scope: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[MemoryMetadata], int, bool]:
        """
        List long-term memories with optional filters.

        Args:
            scope: Filter by scope
            tags: Filter by tags (must contain all)
            memory_type: Filter by memory_type
            page: Page number (1-indexed)
            page_size: Items per page

        Returns:
            (items, total, has_more)
        """
        from sqlalchemy import func

        base_query = (
            select(MemoryMetadata)
            .where(MemoryMetadata.organization_id == self.org_id)
            .where(MemoryMetadata.is_active == True)
        )

        if scope:
            base_query = base_query.where(MemoryMetadata.scope == scope)

        if memory_type:
            base_query = base_query.where(MemoryMetadata.memory_type == memory_type)

        if tags:
            base_query = base_query.where(MemoryMetadata.tags.contains(tags))

        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        offset = max(page - 1, 0) * page_size
        query = (
            base_query
            .order_by(MemoryMetadata.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )

        db_result = await self.session.execute(query)
        items = list(db_result.scalars().all())

        has_more = total > (offset + page_size)

        return items, total, has_more
    
    async def promote_memory(
        self,
        stm_id: str,
        embedding: Optional[List[float]] = None,
    ) -> Optional[MemoryMetadata]:
        """
        Manually promote a short-term memory to long-term storage.
        
        Args:
            stm_id: Short-term memory ID
            embedding: Optional embedding vector
        
        Returns:
            Created MemoryMetadata or None if STM not found
        """
        promoter = MemoryPromoter(self.session, self.user_id, self.org_id)
        return await promoter.promote_by_id(stm_id, embedding, reason="manual")

    # =========================================================================
    # Read
    # =========================================================================
    
    async def get_memory(
        self,
        memory_id: str,
        request_id: Optional[str] = None,
    ) -> Optional[MemoryMetadata]:
        """
        Get a memory by ID with permission checking.
        
        Args:
            memory_id: Memory UUID
            request_id: Request ID for audit correlation
        
        Returns:
            MemoryMetadata if found and authorized, None otherwise
        
        Raises:
            PermissionError: If user lacks read permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "read", self.clearance_level
        )
        
        # Log access attempt
        await self.audit_service.log_memory_access(
            user_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            action="read",
            authorized=access.allowed,
            authorization_method=access.method,
            denial_reason=access.reason if not access.allowed else None,
            request_id=request_id,
        )
        
        if not access.allowed:
            raise PermissionError(access.reason)
        
        # Get memory (RLS will filter)
        memory = await self.session.get(MemoryMetadata, memory_id)
        
        if memory:
            # Update access tracking
            memory.access_count += 1
            # last_accessed_at is a timestamp without timezone in Postgres.
            memory.last_accessed_at = datetime.utcnow()
        
        return memory
    
    async def explain_access(
        self,
        memory_id: str,
    ) -> dict:
        """
        Explain why user can or cannot access a memory.
        
        Args:
            memory_id: Memory UUID
        
        Returns:
            Detailed access explanation
        """
        return await self.permission_checker.explain_access(
            self.user_id, self.org_id, memory_id, self.clearance_level
        )
    
    # =========================================================================
    # Search
    # =========================================================================
    
    async def search_memories(
        self,
        query_embedding: List[float],
        request: MemorySearchRequest,
        request_id: Optional[str] = None,
    ) -> List[MemoryMetadata]:
        """
        Search memories using vector similarity.
        
        SECURITY: This method:
        1. Searches Qdrant with org filter
        2. Fetches results from Postgres (RLS filtered)
        3. Verifies each result against permission checker
        
        Args:
            query_embedding: Query vector
            request: Search parameters
            request_id: Request ID for audit
        
        Returns:
            List of authorized MemoryMetadata results
        """
        normalized_tags = [str(t).strip() for t in (request.tags or []) if str(t).strip()]
        date_from = self._normalize_utc_timestamp(getattr(request, "date_from", None))
        date_to = self._normalize_utc_timestamp(getattr(request, "date_to", None))

        qdrant_results = []
        graph_scores: dict[str, float] = {}
        lexical_scores: dict[str, float] = {}
        fallback_reason: str | None = None
        query_variants_used = 0
        multihop_subqueries_used = 0
        vector_tag_filter_relaxed = False

        ranking_meta = self.get_search_ranking_meta(request)
        decay_enabled = bool(ranking_meta.get("temporal_decay_enabled"))
        half_life_days = float(ranking_meta.get("temporal_decay_half_life_days") or 0.0)

        # Vector leg (Qdrant)
        scope_val = request.scope.value if hasattr(request.scope, "value") else request.scope
        complex_query = self._is_complex_query(request.query)
        overfetch_multiplier = 3 if complex_query else 2
        vector_limit = request.limit * overfetch_multiplier

        qdrant_error: Exception | None = None

        try:
            vector_tags: Optional[List[str]] = normalized_tags or None
            qdrant_results = await QdrantService.search(
                org_id=self.org_id,
                query_vector=query_embedding,
                limit=vector_limit,  # Over-fetch to account for RLS filtering
                score_threshold=request.score_threshold or 0.0,
                scope_filter=scope_val,
                team_id=request.team_id,
                tags=vector_tags,
            )

            # Core resilience: if strict tag payload filters produce zero vector hits,
            # retry vector retrieval without tag filters and enforce tag constraints
            # downstream in Postgres/RLS filtering.
            if not qdrant_results and normalized_tags:
                relaxed_results = await QdrantService.search(
                    org_id=self.org_id,
                    query_vector=query_embedding,
                    limit=vector_limit,
                    score_threshold=request.score_threshold or 0.0,
                    scope_filter=scope_val,
                    team_id=request.team_id,
                    tags=None,
                )
                if relaxed_results:
                    qdrant_results = relaxed_results
                    vector_tag_filter_relaxed = True
                    fallback_reason = "vector_tag_filter_relaxed_retry"
                    vector_tags = None

            # Query expansion (core, graph-independent): retry semantic search with
            # lightweight lexical rewrites and blend into candidate pool.
            if bool(getattr(settings, "SEARCH_QUERY_EXPANSION_ENABLED", True)):
                max_q_variants = int(getattr(settings, "SEARCH_QUERY_EXPANSION_MAX_VARIANTS", 3) or 3)
                variants = self._build_query_variants(request.query, max_q_variants)
                for idx, variant in enumerate(variants):
                    try:
                        variant_embedding = await EmbeddingService.embed(variant)
                        variant_results = await QdrantService.search(
                            org_id=self.org_id,
                            query_vector=variant_embedding,
                            limit=vector_limit,
                            score_threshold=max((request.score_threshold or 0.0) - 0.05, 0.0),
                            scope_filter=scope_val,
                            team_id=request.team_id,
                            tags=vector_tags,
                        )
                    except Exception:
                        continue

                    blend = max(0.8, 0.95 - (0.05 * idx))
                    for result in variant_results:
                        r_copy = dict(result)
                        r_copy["score"] = float(result.get("score") or 0.0) * blend
                        qdrant_results.append(r_copy)
                query_variants_used = len(variants)

            # Multi-hop subquery leg: decompose complex questions and fetch evidence
            # for each hop so downstream fusion can retain bridge facts.
            if complex_query and bool(getattr(settings, "SEARCH_MULTI_HOP_SUBQUERY_ENABLED", True)):
                max_subqueries = int(getattr(settings, "SEARCH_MULTI_HOP_MAX_SUBQUERIES", 3) or 3)
                subqueries = self._build_multihop_subqueries(request.query, max_subqueries)
                for idx, sq in enumerate(subqueries):
                    try:
                        sq_embedding = await EmbeddingService.embed(sq)
                        sq_results = await QdrantService.search(
                            org_id=self.org_id,
                            query_vector=sq_embedding,
                            limit=max(request.limit * 2, 10),
                            score_threshold=max((request.score_threshold or 0.0) - 0.1, 0.0),
                            scope_filter=scope_val,
                            team_id=request.team_id,
                            tags=vector_tags,
                        )
                    except Exception:
                        continue
                    blend = max(0.7, 0.92 - (0.08 * idx))
                    for result in sq_results:
                        r_copy = dict(result)
                        r_copy["score"] = float(result.get("score") or 0.0) * blend
                        qdrant_results.append(r_copy)
                multihop_subqueries_used = len(subqueries)

            # Graph-guided multi-query expansion:
            # - Seed from first-pass vector hits
            # - Expand neighbors from graph_relationships
            # - Build enriched query variants and re-query vectors
            graph_enabled = bool(getattr(request, "use_graph", False)) and bool(
                getattr(settings, "SEARCH_GRAPH_EXPANSION_ENABLED", True)
            )
            if graph_enabled:
                seed_limit = int(getattr(settings, "SEARCH_GRAPH_SEED_LIMIT", 12) or 12)
                neighbor_limit = int(getattr(settings, "SEARCH_GRAPH_NEIGHBOR_LIMIT", 32) or 32)
                max_variants = int(getattr(settings, "SEARCH_MULTI_QUERY_MAX_VARIANTS", 4) or 4)

                seed_ids = [
                    str(r.get("payload", {}).get("memory_id"))
                    for r in qdrant_results[:seed_limit]
                    if r.get("payload", {}).get("memory_id")
                ]
                seed_ids = [sid for sid in seed_ids if sid]

                if seed_ids:
                    graph_scores = await self._load_graph_neighbor_scores(seed_ids, neighbor_limit)
                    variants = await self._build_graph_query_variants(request.query, graph_scores, max_variants)

                    for idx, variant in enumerate(variants):
                        try:
                            variant_embedding = await EmbeddingService.embed(variant)
                            variant_results = await QdrantService.search(
                                org_id=self.org_id,
                                query_vector=variant_embedding,
                                limit=vector_limit,
                                score_threshold=max((request.score_threshold or 0.0) - 0.05, 0.0),
                                scope_filter=scope_val,
                                team_id=request.team_id,
                                tags=vector_tags,
                            )
                        except Exception:
                            continue

                        # Slightly discount later variants so the base query remains primary.
                        blend = max(0.75, 0.95 - (0.05 * idx))
                        for result in variant_results:
                            payload = result.get("payload") or {}
                            memory_id = str(payload.get("memory_id") or "")
                            if not memory_id:
                                continue
                            score = float(result.get("score") or 0.0) * blend
                            # Append for unified candidate set and keep best score downstream.
                            qdrant_results.append(result)
                            if memory_id in graph_scores:
                                graph_scores[memory_id] = max(graph_scores[memory_id], score)
                            else:
                                graph_scores[memory_id] = max(graph_scores.get(memory_id, 0.0), score * 0.8)
        except Exception as exc:
            logger.warning(
                "Qdrant search failed for org_id=%s query=%r: %s",
                self.org_id,
                request.query,
                exc,
            )
            qdrant_error = exc

        # Lexical leg (Postgres FTS) - opt-in via request.hybrid
        vector_scores: dict[str, float] = {}
        for result in qdrant_results:
            payload = result.get("payload") or {}
            memory_id = payload.get("memory_id")
            if not memory_id:
                continue
            memory_id = str(memory_id)
            score = float(result.get("score") or 0.0)
            vector_scores[memory_id] = max(vector_scores.get(memory_id, 0.0), score)

        lexical_enabled = self._should_auto_hybrid(vector_scores, request)
        if lexical_enabled:
            # Full-text search using pre-computed search_vector column with GIN index.
            # Uses BM25-style ranking via ts_rank_cd with normalization.
            # 
            # Normalization flags (see PostgreSQL docs):
            # 0 = default (ignores document length)
            # 1 = divides rank by 1 + log(document length)
            # 2 = divides rank by document length
            # 4 = divides rank by mean harmonic distance between extents
            # 8 = divides rank by number of unique words
            # 16 = divides rank by 1 + log(number of unique words)
            # 32 = divides rank by rank + 1
            # 
            # We use normalization=1 (BM25-like length normalization)
            normalization = 1
            
            # Build query using english stemming/stop-words so natural
            # language questions don't over-constrain FTS matches.
            tsq = func.plainto_tsquery("english", request.query)
            
            # Rank using ts_rank_cd (Cover Density ranking).
            # Use the 3-arg signature (vector, query, normalization) for broad
            # PostgreSQL compatibility; weighted variant requires explicit
            # float4[] typing and can fail on some deployments.
            rank = func.ts_rank_cd(
                MemoryMetadata.search_vector,
                tsq,
                normalization,
            )

            stmt = (
                select(MemoryMetadata.id, rank.label("rank"))
                .where(
                    MemoryMetadata.organization_id == self.org_id,
                    MemoryMetadata.is_active.is_(True),
                    MemoryMetadata.search_vector.op("@@")(tsq),
                )
                .order_by(rank.desc())
                .limit(request.limit * 2)
            )

            if scope_val:
                stmt = stmt.where(MemoryMetadata.scope == scope_val)
            if request.team_id:
                stmt = stmt.where(MemoryMetadata.scope == "team", MemoryMetadata.scope_id == request.team_id)
            if normalized_tags:
                stmt = stmt.where(MemoryMetadata.tags.contains(normalized_tags))
            if date_from is not None:
                stmt = stmt.where(MemoryMetadata.occurred_at >= date_from)
            if date_to is not None:
                stmt = stmt.where(MemoryMetadata.occurred_at <= date_to)

            lex_res = await self.session.execute(stmt)
            for row in lex_res.all():
                memory_id = str(row[0])
                lexical_scores[memory_id] = float(row[1] or 0.0)

            if not getattr(request, "hybrid", False):
                fallback_reason = "auto_hybrid_low_vector_confidence"

        if qdrant_error is not None:
            allow_lexical_fallback = bool(
                getattr(settings, "SEARCH_ALLOW_LEXICAL_FALLBACK_ON_VECTOR_ERROR", True)
            )
            if vector_scores or lexical_scores:
                fallback_reason = fallback_reason or "vector_partial_error_degraded"
            elif not (allow_lexical_fallback and lexical_scores):
                raise RuntimeError("Vector search unavailable") from qdrant_error
            else:
                fallback_reason = "vector_error_lexical_fallback"

        # Candidate IDs from both legs
        candidate_ids = list({*vector_scores.keys(), *lexical_scores.keys(), *graph_scores.keys()})
        graph_requested = bool(getattr(request, "use_graph", False))
        if not candidate_ids:
            if graph_requested:
                fallback_request = request.model_copy(update={"use_graph": False})
                retried_results = await self.search_memories(
                    query_embedding=query_embedding,
                    request=fallback_request,
                    request_id=request_id,
                )
                if retried_results:
                    retried_diag = dict(self.get_last_search_diagnostics() or {})
                    retried_diag["fallback_reason"] = "graph_empty_retry_without_graph"
                    retried_diag["graph_retry_without_graph"] = True
                    self._last_search_diagnostics = retried_diag
                    return retried_results[: request.limit]
            self._last_search_diagnostics = {
                "vector_hits": 0,
                "lexical_hits": 0,
                "graph_hits": 0,
                "candidate_ids": 0,
                "authorized_hits": 0,
                "fallback_reason": fallback_reason,
                "query_variants_used": query_variants_used,
                "multihop_subqueries_used": multihop_subqueries_used,
                "confidence_buckets": {"high": 0, "medium": 0, "low": 0},
                "graph_retry_without_graph": False,
            }
            return []

        # Fetch from Postgres (RLS will filter unauthorized)
        query = select(MemoryMetadata).where(
            and_(
                MemoryMetadata.id.in_(candidate_ids),
                MemoryMetadata.is_active.is_(True),
                # Defense-in-depth: even with RLS, constrain by org_id explicitly.
                MemoryMetadata.organization_id == self.org_id,
            )
        )

        if scope_val:
            query = query.where(MemoryMetadata.scope == scope_val)
        if request.team_id:
            query = query.where(MemoryMetadata.scope == "team", MemoryMetadata.scope_id == request.team_id)
        if normalized_tags:
            query = query.where(MemoryMetadata.tags.contains(normalized_tags))
        if date_from is not None:
            query = query.where(MemoryMetadata.occurred_at >= date_from)
        if date_to is not None:
            query = query.where(MemoryMetadata.occurred_at <= date_to)

        result = await self.session.execute(query)
        memories = result.scalars().all()

        # Optional: feedback-driven reranking (closed-loop retrieval).
        # Uses most recent per-user feedback of type "relevance" for each memory.
        feedback_payloads: dict[str, dict] = {}
        if bool(ranking_meta.get("feedback_rerank_enabled")) and candidate_ids:
            window_days = float(ranking_meta.get("feedback_rerank_window_days") or 90.0)
            pos_mult = float(ranking_meta.get("feedback_rerank_positive_multiplier") or 1.15)
            neg_mult = float(ranking_meta.get("feedback_rerank_negative_multiplier") or 0.5)
            if pos_mult <= 0:
                pos_mult = 1.15
            if neg_mult <= 0:
                neg_mult = 0.5

            stmt = select(
                MemoryFeedback.memory_id.label("memory_id"),
                MemoryFeedback.payload.label("payload"),
                func.row_number()
                .over(partition_by=MemoryFeedback.memory_id, order_by=MemoryFeedback.created_at.desc())
                .label("rn"),
            ).where(
                MemoryFeedback.organization_id == self.org_id,
                MemoryFeedback.actor_id == self.user_id,
                MemoryFeedback.feedback_type == "relevance",
                MemoryFeedback.memory_id.in_(candidate_ids),
            )
            if window_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
                stmt = stmt.where(MemoryFeedback.created_at >= cutoff)

            subq = stmt.subquery()
            latest_stmt = select(subq.c.memory_id, subq.c.payload).where(subq.c.rn == 1)
            fb_res = await self.session.execute(latest_stmt)
            for mid, payload in fb_res.all():
                if isinstance(payload, dict):
                    feedback_payloads[str(mid)] = payload

        # Normalize scores and compute a combined score
        max_vec = max(vector_scores.values(), default=0.0)
        max_lex = max(lexical_scores.values(), default=0.0)
        max_graph = max(graph_scores.values(), default=0.0)

        graph_enabled = bool(getattr(request, "use_graph", False)) and bool(
            getattr(settings, "SEARCH_GRAPH_EXPANSION_ENABLED", True)
        )
        if graph_enabled:
            vec_weight = 0.6
            lex_weight = 0.25
            graph_weight = 0.15
        else:
            vec_weight = 0.7
            lex_weight = 0.3
            graph_weight = 0.0

        # HNMS-inspired ranking mode selector.
        # Mode influences temporal decay weighting (recency bias).
        # (Computed once via get_search_ranking_meta.)
        
        # Verify each memory and collect authorized ones.
        # Use the batched permission path to avoid per-result N+1 checks on hot search traffic.
        authorized_ids = set(
            await self.permission_checker.filter_memory_ids_with_access(
                self.user_id,
                self.org_id,
                [m.id for m in memories],
                "read",
                self.clearance_level,
            )
        )
        authorized_memories: list[MemoryMetadata] = []
        normalized_similarities: dict[str, float] = {}
        dominant_session = self._infer_dominant_session(memories, vector_scores)
        query_hints = self._query_persona_hints(request.query)
        confidence_buckets = {"high": 0, "medium": 0, "low": 0}
        for memory in memories:
            if memory.id in authorized_ids:
                vec = vector_scores.get(memory.id, 0.0)
                lex = lexical_scores.get(memory.id, 0.0)
                graph = graph_scores.get(memory.id, 0.0)

                vec_norm = (vec / max_vec) if max_vec > 0 else 0.0
                lex_norm = (lex / max_lex) if max_lex > 0 else 0.0
                graph_norm = (graph / max_graph) if max_graph > 0 else 0.0

                normalized_similarities[str(memory.id)] = float(vec_norm)

                if getattr(request, "hybrid", False):
                    memory.score = (vec_weight * vec_norm) + (lex_weight * lex_norm) + (graph_weight * graph_norm)
                else:
                    memory.score = (vec_weight * vec_norm) + (graph_weight * graph_norm)

                # Perspective/session-aware reranking.
                # Boost evidence from the dominant session and likely speaker hints.
                session_key = self._memory_session_key(memory).lower()
                speaker_key = self._memory_speaker_key(memory)
                perspective_boost = 1.0
                if dominant_session:
                    perspective_boost *= 1.08 if session_key == dominant_session else 0.97
                if query_hints:
                    if any(h in speaker_key for h in query_hints):
                        perspective_boost *= 1.12
                    else:
                        perspective_boost *= 0.96
                memory.score = float(memory.score or 0.0) * perspective_boost

                # Lexical grounding boost: prioritize memories that actually cover
                # the question terms, especially under degraded vector recall.
                overlap = self._query_overlap_score(request.query, memory)
                if overlap >= 0.6:
                    overlap_mult = 1.22
                elif overlap >= 0.35:
                    overlap_mult = 1.10
                elif overlap == 0.0:
                    overlap_mult = 0.78
                else:
                    overlap_mult = 0.92
                memory.score = float(memory.score or 0.0) * overlap_mult

                # Optional: temporal decay weighting (HNMS-inspired).
                # Uses last_accessed_at if present, else updated_at, else created_at.
                if decay_enabled and half_life_days > 0:
                    anchor = (
                        getattr(memory, "last_accessed_at", None)
                        or getattr(memory, "updated_at", None)
                        or getattr(memory, "created_at", None)
                    )
                    if anchor is not None:
                        now = datetime.now(timezone.utc)
                        if getattr(anchor, "tzinfo", None) is None:
                            anchor = anchor.replace(tzinfo=timezone.utc)
                        age_days = max(0.0, (now - anchor).total_seconds() / 86400.0)
                        decay = math.pow(0.5, age_days / half_life_days)
                        memory.score = float(memory.score or 0.0) * float(decay)

                prior_score, prior_count = self._retrieval_learning_prior(memory)
                if prior_count > 0:
                    confidence = min(1.0, prior_count / 4.0)
                    prior_multiplier = 1.0 + (0.2 * prior_score * confidence)
                    if prior_multiplier > 0:
                        memory.score = float(memory.score or 0.0) * prior_multiplier

                # Optional: per-user relevance feedback reranking.
                # Expected payload shapes:
                # - {"value": 1} / {"value": -1}
                # - {"relevant": true} / {"relevant": false}
                # - {"relevance": 1.0} / {"relevance": -1.0}
                if feedback_payloads:
                    payload = feedback_payloads.get(str(memory.id))
                    if isinstance(payload, dict):
                        raw = payload.get("value")
                        if raw is None:
                            raw = payload.get("relevance")
                        if raw is None:
                            raw = payload.get("relevant")

                        v: float | None = None
                        if isinstance(raw, bool):
                            v = 1.0 if raw else -1.0
                        else:
                            try:
                                v = float(raw)  # type: ignore[arg-type]
                            except Exception:
                                v = None

                        if v is not None:
                            if v > 0:
                                memory.score = float(memory.score or 0.0) * pos_mult
                            elif v < 0:
                                memory.score = float(memory.score or 0.0) * neg_mult

                final_score = float(memory.score or 0.0)
                if final_score >= 0.75:
                    confidence_buckets["high"] += 1
                elif final_score >= 0.45:
                    confidence_buckets["medium"] += 1
                elif final_score > 0:
                    confidence_buckets["low"] += 1

                # Attach best-effort provenance for citations.
                # For now, the "source" is the memory itself (future: attachments/docs).
                # We use updated_at + content_hash as a stable version hint.
                memory.provenance = [
                    {
                        "kind": "memory",
                        "source_type": getattr(memory, "source_type", None),
                        "source_id": getattr(memory, "source_id", None),
                        "source_version": (
                            memory.updated_at.isoformat() if getattr(memory, "updated_at", None) else None
                        ),
                        "content_hash": getattr(memory, "content_hash", None),
                        "title": getattr(memory, "title", None),
                        "excerpt": getattr(memory, "content_preview", None),
                        "score": memory.score,
                        "meta": {
                            "memory_id": memory.id,
                            "vector_id": getattr(memory, "vector_id", None),
                            "created_at": (
                                memory.created_at.isoformat() if getattr(memory, "created_at", None) else None
                            ),
                            "updated_at": (
                                memory.updated_at.isoformat() if getattr(memory, "updated_at", None) else None
                            ),
                            "embedding_model": getattr(memory, "embedding_model", None),
                            "scope": getattr(memory, "scope", None),
                            "scope_id": getattr(memory, "scope_id", None),
                            "classification": getattr(memory, "classification", None),
                        },
                    }
                ]

                authorized_memories.append(memory)

                # Skip per-result audit writes in search path to keep retrieval resilient.
        
        # Activation scoring + explanation logging + async update tasks.
        # This keeps the synchronous request path fast (math + batched reads) and
        # pushes counter/coactivation writes into Celery.
        # Keep the request path side-effect free for reliable interactive demos.
        if False and authorized_memories:
            try:
                from app.services.memory_activation.retrieval import MemoryRetrievalService

                retrieval = MemoryRetrievalService(
                    session=self.session,
                    org_id=str(self.org_id),
                    user_id=str(self.user_id),
                )

                authorized_ids = [str(m.id) for m in authorized_memories]

                ranked_dicts, explanation_results = await retrieval.score_and_rank_results(
                    memory_ids=authorized_ids,
                    query=request.query,
                    similarities=normalized_similarities,
                    scope=scope_val,
                )

                activation_by_id = {str(r["id"]): float(r["activation_score"]) for r in ranked_dicts}
                ranked_ids = [str(r["id"]) for r in ranked_dicts]

                mem_by_id = {str(m.id): m for m in authorized_memories}
                authorized_memories = [mem_by_id[mid] for mid in ranked_ids if mid in mem_by_id]

                for mem in authorized_memories:
                    mem.score = activation_by_id.get(str(mem.id), 0.0)

                # Skip explanation writes in the request path to avoid DB flush side-effects.
                explanation_id = None

                # Best-effort enqueue of background updates (no-op in unit tests).
                try:
                    from app.core.celery_app import celery_app

                    broker = celery_app.conf.broker_url
                    if broker and not str(broker).startswith("memory://"):
                        from app.services.memory_activation.tasks import (
                            memory_access_update_task,
                            coactivation_update_task,
                        )

                        top_ids = [str(m.id) for m in authorized_memories[: request.limit]]
                        for mid in top_ids:
                            memory_access_update_task.apply_async(
                                kwargs={
                                    "memory_id": mid,
                                    "org_id": str(self.org_id),
                                    "user_id": str(self.user_id),
                                    "retrieval_explanation_id": explanation_id,
                                },
                                countdown=2,
                            )

                        if len(top_ids) > 1:
                            coactivation_update_task.apply_async(
                                kwargs={
                                    "primary_memory_id": top_ids[0],
                                    "coactivated_memory_ids": top_ids[1:],
                                    "org_id": str(self.org_id),
                                },
                                countdown=2,
                            )
                except Exception:
                    # Celery isn't required for serving search.
                    pass

            except Exception:
                # If activation scoring fails, fall back to the legacy combined score.
                pass

        # If graph-guided retrieval produced no authorized results, retry once
        # with graph expansion disabled to preserve baseline semantic/hybrid recall.
        graph_requested = bool(getattr(request, "use_graph", False))
        if graph_requested and not authorized_memories:
            fallback_request = request.model_copy(update={"use_graph": False})
            retried_results = await self.search_memories(
                query_embedding=query_embedding,
                request=fallback_request,
                request_id=request_id,
            )
            if retried_results:
                retried_diag = dict(self.get_last_search_diagnostics() or {})
                retried_diag["fallback_reason"] = "graph_empty_retry_without_graph"
                retried_diag["graph_retry_without_graph"] = True
                self._last_search_diagnostics = retried_diag
                return retried_results[: request.limit]

        # Sort by score and limit
        authorized_memories.sort(key=lambda m: float(m.score or 0.0), reverse=True)
        self._last_search_diagnostics = {
            "vector_hits": len(vector_scores),
            "lexical_hits": len(lexical_scores),
            "graph_hits": len(graph_scores),
            "candidate_ids": len(candidate_ids),
            "authorized_hits": len(authorized_memories),
            "fallback_reason": fallback_reason,
            "vector_tag_filter_relaxed": vector_tag_filter_relaxed,
            "query_variants_used": query_variants_used,
            "multihop_subqueries_used": multihop_subqueries_used,
            "dominant_session": dominant_session or None,
            "confidence_buckets": confidence_buckets,
            "lexical_mode": "forced" if getattr(request, "hybrid", False) else ("auto" if lexical_enabled else "off"),
            "graph_retry_without_graph": False,
        }
        return authorized_memories[: request.limit]

    def get_search_ranking_meta(self, request: MemorySearchRequest) -> dict[str, object]:
        """Compute effective ranking parameters for a search request.

        This is used for both search scoring and API response observability.
        """

        req_mode = getattr(request, "hnms_mode", None)
        allow_override = bool(getattr(settings, "SEARCH_HNMS_MODE_ALLOW_REQUEST_OVERRIDE", True))
        mode_source = "config"
        mode = req_mode if (allow_override and req_mode) else getattr(settings, "SEARCH_HNMS_MODE_DEFAULT", "balanced")
        if not isinstance(mode, str) and hasattr(mode, "value"):
            mode = mode.value
        if isinstance(req_mode, str) or hasattr(req_mode, "value"):
            if allow_override and req_mode:
                mode_source = "request"

        mode_str = str(mode or "balanced").lower()

        if mode_str == "performance":
            temporal_decay_enabled = True
            temporal_decay_half_life_days = float(
                getattr(settings, "SEARCH_HNMS_MODE_PERFORMANCE_HALF_LIFE_DAYS", 7.0) or 7.0
            )
        elif mode_str == "research":
            temporal_decay_enabled = True
            temporal_decay_half_life_days = float(
                getattr(settings, "SEARCH_HNMS_MODE_RESEARCH_HALF_LIFE_DAYS", 90.0) or 90.0
            )
        else:
            # balanced (or unknown) falls back to the base temporal decay knobs
            temporal_decay_enabled = bool(getattr(settings, "SEARCH_TEMPORAL_DECAY_ENABLED", False))
            temporal_decay_half_life_days = float(
                getattr(settings, "SEARCH_TEMPORAL_DECAY_HALF_LIFE_DAYS", 30.0) or 30.0
            )
            mode_str = "balanced"

        feedback_rerank_enabled = bool(getattr(settings, "SEARCH_FEEDBACK_RERANK_ENABLED", False))
        feedback_rerank_window_days = float(getattr(settings, "SEARCH_FEEDBACK_RERANK_WINDOW_DAYS", 90.0) or 90.0)
        feedback_rerank_positive_multiplier = float(
            getattr(settings, "SEARCH_FEEDBACK_RERANK_POSITIVE_MULTIPLIER", 1.15) or 1.15
        )
        feedback_rerank_negative_multiplier = float(
            getattr(settings, "SEARCH_FEEDBACK_RERANK_NEGATIVE_MULTIPLIER", 0.5) or 0.5
        )

        return {
            "hnms_mode_effective": mode_str,
            "hnms_mode_source": mode_source,
            "temporal_decay_enabled": temporal_decay_enabled,
            "temporal_decay_half_life_days": temporal_decay_half_life_days,
            "feedback_rerank_enabled": feedback_rerank_enabled,
            "feedback_rerank_window_days": feedback_rerank_window_days,
            "feedback_rerank_positive_multiplier": feedback_rerank_positive_multiplier,
            "feedback_rerank_negative_multiplier": feedback_rerank_negative_multiplier,
            "retrieval_prior_boost_enabled": True,
            "graph_expansion_enabled": bool(getattr(settings, "SEARCH_GRAPH_EXPANSION_ENABLED", True)),
            "graph_expansion_requested": bool(getattr(request, "use_graph", False)),
            "multi_query_max_variants": int(getattr(settings, "SEARCH_MULTI_QUERY_MAX_VARIANTS", 4) or 4),
            "query_expansion_enabled": bool(getattr(settings, "SEARCH_QUERY_EXPANSION_ENABLED", True)),
            "query_expansion_max_variants": int(getattr(settings, "SEARCH_QUERY_EXPANSION_MAX_VARIANTS", 3) or 3),
            "auto_hybrid_enabled": bool(getattr(settings, "SEARCH_AUTO_HYBRID_ENABLED", True)),
            "auto_hybrid_min_vector_hits": int(getattr(settings, "SEARCH_AUTO_HYBRID_MIN_VECTOR_HITS", 4) or 4),
            "auto_hybrid_max_top_score": float(getattr(settings, "SEARCH_AUTO_HYBRID_MAX_TOP_SCORE", 0.55) or 0.55),
            "multi_hop_subquery_enabled": bool(getattr(settings, "SEARCH_MULTI_HOP_SUBQUERY_ENABLED", True)),
            "multi_hop_max_subqueries": int(getattr(settings, "SEARCH_MULTI_HOP_MAX_SUBQUERIES", 3) or 3),
            "vector_error_lexical_fallback_enabled": bool(
                getattr(settings, "SEARCH_ALLOW_LEXICAL_FALLBACK_ON_VECTOR_ERROR", True)
            ),
        }
    
    # =========================================================================
    # Update
    # =========================================================================
    
    async def update_memory(
        self,
        memory_id: str,
        data: MemoryUpdate,
        new_embedding: Optional[List[float]] = None,
        request_id: Optional[str] = None,
    ) -> MemoryMetadata:
        """
        Update a memory.
        
        Args:
            memory_id: Memory UUID
            data: Update data
            new_embedding: New embedding if content changed
            request_id: Request ID for audit
        
        Returns:
            Updated MemoryMetadata
        
        Raises:
            PermissionError: If user lacks write permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "write", self.clearance_level
        )
        
        if not access.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id=memory_id,
                operation="update",
                success=False,
                error_message=access.reason,
            )
            raise PermissionError(access.reason)
        
        # Get memory
        memory = await self.session.get(MemoryMetadata, memory_id)
        if not memory:
            raise ValueError("Memory not found")
        
        # Track changes
        changes = {}

        if data.content is not None:
            changes["content"] = {
                "old_preview": memory.content_preview,
                "new_preview": data.content[:2000],
            }
            memory.content_preview = data.content[:2000]
            memory.content_hash = hashlib.sha256(data.content.encode("utf-8")).hexdigest()

        # Update fields
        if data.title is not None:
            changes["title"] = {"old": memory.title, "new": data.title}
            memory.title = data.title
        
        if data.tags is not None:
            changes["tags"] = {"old": memory.tags, "new": data.tags}
            memory.tags = data.tags
        
        if data.classification is not None:
            changes["classification"] = {"old": memory.classification, "new": data.classification}
            memory.classification = data.classification
        
        if data.extra_metadata is not None:
            changes["extra_metadata"] = {"old": memory.extra_metadata, "new": data.extra_metadata}
            memory.extra_metadata = data.extra_metadata

        if data.retention_days is not None:
            changes["retention_days"] = {"old": memory.retention_days, "new": data.retention_days}
            memory.retention_days = data.retention_days

        if data.content is not None or data.title is not None or data.tags is not None:
            await self._ensure_search_vector(
                memory_id=memory_id,
                title=memory.title,
                content=data.content if data.content is not None else (memory.content_preview or ""),
                tags=memory.tags,
            )

        # Update embedding in Qdrant if provided
        if new_embedding:
            await QdrantService.upsert_memory(
                memory_id=memory.vector_id,
                org_id=self.org_id,
                vector=new_embedding,
                payload={
                    "memory_id": memory_id,
                    "scope": memory.scope,
                    "scope_id": memory.scope_id,
                    "team_id": memory.scope_id if str(memory.scope) == "team" else None,
                    "owner_id": memory.owner_id,
                    "tags": memory.tags,
                    "classification": memory.classification,
                    "memory_type": memory.memory_type,
                    "created_at": memory.created_at.isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        
        # Audit log
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="update",
            success=True,
            details={"changes": changes},
        )
        
        return memory
    
    # =========================================================================
    # Delete
    # =========================================================================
    
    async def delete_memory(
        self,
        memory_id: str,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Soft-delete a memory.
        
        Args:
            memory_id: Memory UUID
            request_id: Request ID for audit
        
        Returns:
            True if deleted
        
        Raises:
            PermissionError: If user lacks delete permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "delete", self.clearance_level
        )
        
        if not access.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id=memory_id,
                operation="delete",
                success=False,
                error_message=access.reason,
            )
            raise PermissionError(access.reason)
        
        # Get memory
        memory = await self.session.get(MemoryMetadata, memory_id)
        if not memory:
            raise ValueError("Memory not found")
        
        # Check legal hold
        if memory.legal_hold:
            raise PermissionError("Memory is under legal hold and cannot be deleted")
        
        # Soft delete
        memory.is_active = False
        
        # Remove from Qdrant (best-effort; soft-delete in DB is authoritative)
        try:
            if memory.vector_id:
                await QdrantService.delete_memory(memory.vector_id, self.org_id)
        except Exception:
            pass
        
        # Audit log
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="delete",
            success=True,
        )
        
        return True
    
    # =========================================================================
    # Sharing
    # =========================================================================
    
    async def share_memory(
        self,
        memory_id: str,
        request: MemoryShareRequest,
        request_id: Optional[str] = None,
    ) -> MemorySharing:
        """
        Share a memory with a user or group.
        
        Args:
            memory_id: Memory UUID
            request: Share request with target and permission
            request_id: Request ID for audit
        
        Returns:
            Created MemorySharing record
        
        Raises:
            PermissionError: If user lacks share permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "share", self.clearance_level
        )
        
        if not access.allowed:
            raise PermissionError(access.reason)
        
        # Create share record
        share = MemorySharing(
            memory_id=memory_id,
            organization_id=self.org_id,
            share_type=request.share_type,
            target_id=request.target_id,
            permission=request.permission,
            expires_at=request.expires_at,
            shared_by=self.user_id,
            share_reason=request.reason,
        )
        
        self.session.add(share)
        await self.session.flush()
        
        # Invalidate target's permission cache
        if request.share_type == "user":
            await self.permission_checker.invalidate_user_cache(
                request.target_id, self.org_id
            )
        
        # Audit log
        await self.audit_service.log_memory_share(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            share_type=request.share_type,
            target_id=request.target_id,
            permission=request.permission,
            expires_at=request.expires_at,
        )
        
        return share
