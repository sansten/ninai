"""Agent runner service.

Runs an agent against a memory reference and persists an AgentRun row.

Key properties:
- Idempotent per (org_id, memory_id, agent_name, agent_version)
- Computes inputs_hash for observability and safe retries
- Sets tenant DB session vars in workers (RLS)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
import contextvars
import json
import time
from typing import Optional
import itertools

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.registry import get_agent
from app.agents.types import AgentResult
from app.agents.utils import compute_inputs_hash
from app.agents.llm.tool_events import ToolEvent, ToolEventSink
from app.core.config import settings
from app.core.database import get_tenant_session
from app.models.agent_run import AgentRun
from app.models.agent_run_event import AgentRunEvent
from app.models.memory import MemoryMetadata
from app.models.memory_feedback import MemoryFeedback
from app.services.agent_result_cache_service import AgentResultCacheService
from app.services.audit_service import AuditService
from app.services.feedback_learning_config_service import FeedbackLearningConfigService
from app.services.graph_edge_service import GraphEdgeService
from app.services.logseq_export_persistence_service import LogseqExportPersistenceService
from app.services.short_term_memory import ShortTermMemoryService
from app.services.topic_service import TopicService
from app.services.pattern_service import PatternService


_TOOL_EVENT_SINK_VAR: contextvars.ContextVar[ToolEventSink | None] = contextvars.ContextVar(
    "agent_runner_tool_event_sink",
    default=None,
)


async def _emit_tool_event(*, event_type: str, summary_text: str, payload: dict) -> None:
    sink = _TOOL_EVENT_SINK_VAR.get()
    if sink is None:
        return
    try:
        await sink(
            ToolEvent(
                event_type=event_type,
                summary_text=summary_text,
                payload=payload,
            )
        )
    except Exception:
        return


@dataclass(frozen=True)
class PipelineContext:
    org_id: str
    memory_id: str
    initiator_user_id: Optional[str] = None
    trace_id: Optional[str] = None
    storage: str = "long_term"  # long_term|short_term


class AgentRunner:
    def __init__(self, *, service_user_id: Optional[str] = None):
        self.service_user_id = service_user_id

    @staticmethod
    def _stable_json(v: object) -> str:
        try:
            return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return str(v)

    def _cache_enabled(self) -> bool:
        return bool(getattr(settings, "AGENT_CACHE_ENABLED", False))

    def _cache_strategy(self, agent_name: str) -> str:
        # Keep runner logic simple: caching is only for LLM strategy.
        # Per-agent overrides can still force heuristic, but then caching is bypassed.
        v = str(getattr(settings, "AGENT_STRATEGY", "llm") or "llm").strip().lower()
        return v if v in {"llm", "heuristic"} else "llm"

    def _cache_model(self) -> str:
        m = str(getattr(settings, "OLLAMA_MODEL", "") or "").strip()
        return m or "default"

    def _should_cache_agent(self, agent_name: str) -> bool:
        # Only cache deterministic enrichment agents.
        return agent_name in {
            "ClassificationAgent",
            "MetadataExtractionAgent",
            "TopicModelingAgent",
            "PatternDetectionAgent",
            "GraphLinkingAgent",
        }

    def _compute_cache_key(
        self,
        *,
        agent_name: str,
        agent_version: str,
        strategy: str,
        model: str,
        ctx: PipelineContext,
        content: str,
        existing_classification: Optional[str],
        scope: str,
        scope_id: Optional[str],
        enrichment: dict,
        pending_feedback_fingerprint: str,
    ) -> str:
        # Intentionally excludes memory_id to enable cross-memory reuse.
        return compute_inputs_hash(
            [
                agent_name,
                agent_version,
                strategy,
                model,
                ctx.org_id,
                ctx.storage,
                content,
                existing_classification or "",
                scope or "",
                scope_id or "",
                self._stable_json(enrichment or {}),
                pending_feedback_fingerprint or "",
            ]
        )

    def _create_tool_event_sink(
        self,
        *,
        session: AsyncSession,
        run_row: AgentRun,
    ) -> ToolEventSink:
        """Create an async sink for tool_call/tool_result events.

        Uses a per-run in-memory counter for step_index to keep events ordered
        without extra DB reads.
        """

        counter = itertools.count(10)

        async def sink(event: ToolEvent) -> None:
            try:
                event_type = str(event.get("event_type") or "").strip() or "tool_call"
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                summary_text = str(event.get("summary_text") or "").strip()

                await self._append_trajectory_event(
                    session=session,
                    org_id=run_row.organization_id,
                    agent_run_id=run_row.id,
                    memory_id=run_row.memory_id,
                    event_type=event_type,
                    step_index=int(next(counter)),
                    payload=payload,
                    summary_text=summary_text,
                    trace_id=run_row.trace_id,
                    created_at=datetime.now(timezone.utc),
                )
            except Exception:
                return

        return sink

    async def run_agent(
        self,
        *,
        ctx: PipelineContext,
        agent_name: str,
        attempt: int = 1,
        max_attempts: int = 5,
    ) -> AgentResult:
        agent = get_agent(agent_name)

        started_at = datetime.now(timezone.utc)

        if agent is None:
            effective_user_id = self.service_user_id or ctx.initiator_user_id or ""
            buffered_tool_events: list[ToolEvent] = []

            async def _buffer_sink(event: ToolEvent) -> None:
                buffered_tool_events.append(event)

            token = _TOOL_EVENT_SINK_VAR.set(_buffer_sink)
            try:
                await _emit_tool_event(
                    event_type="tool_call",
                    summary_text="get_tenant_session call",
                    payload={
                        "tool": "get_tenant_session",
                        "roles_len": 0,
                        "clearance_level": 0,
                        "justification": "agent_pipeline",
                        "has_user_id": bool(effective_user_id),
                        "has_org_id": bool(ctx.org_id),
                    },
                )
                start = time.perf_counter()

                async with contextlib.AsyncExitStack() as stack:
                    try:
                        session = await stack.enter_async_context(
                            get_tenant_session(
                                user_id=effective_user_id,
                                org_id=ctx.org_id,
                                roles="",
                                clearance_level=0,
                                justification="agent_pipeline",
                            )
                        )
                    except Exception as e:
                        duration_ms = (time.perf_counter() - start) * 1000.0
                        await _emit_tool_event(
                            event_type="tool_result",
                            summary_text="get_tenant_session failed",
                            payload={
                                "tool": "get_tenant_session",
                                "ok": False,
                                "duration_ms": round(duration_ms, 3),
                                "error": str(e)[:2000],
                            },
                        )
                        raise

                    duration_ms = (time.perf_counter() - start) * 1000.0
                    await _emit_tool_event(
                        event_type="tool_result",
                        summary_text="get_tenant_session ok",
                        payload={
                            "tool": "get_tenant_session",
                            "ok": True,
                            "duration_ms": round(duration_ms, 3),
                        },
                    )

                    finished_at = datetime.now(timezone.utc)
                    result = AgentResult(
                        agent_name=agent_name,
                        agent_version="0",
                        memory_id=ctx.memory_id,
                        status="skipped",
                        confidence=0.0,
                        outputs={},
                        warnings=[f"Agent '{agent_name}' not implemented"],
                        errors=[],
                        started_at=started_at,
                        finished_at=finished_at,
                        trace_id=ctx.trace_id,
                    )

                    inputs_hash = compute_inputs_hash([agent_name, "0", ctx.org_id, ctx.memory_id, ctx.storage])
                    await _emit_tool_event(
                        event_type="tool_call",
                        summary_text="compute_inputs_hash computed",
                        payload={"tool": "compute_inputs_hash", "inputs_hash_len": len(inputs_hash), "agent": agent_name},
                    )

                    run_row = await self._get_or_create_run_row(
                        session=session,
                        org_id=ctx.org_id,
                        memory_id=ctx.memory_id,
                        agent_name=agent_name,
                        agent_version="0",
                        inputs_hash=inputs_hash,
                        trace_id=ctx.trace_id,
                        started_at=started_at,
                    )

                    tool_event_sink = self._create_tool_event_sink(session=session, run_row=run_row)
                    _TOOL_EVENT_SINK_VAR.set(tool_event_sink)

                    if buffered_tool_events:
                        for ev in buffered_tool_events:
                            try:
                                await tool_event_sink(ev)
                            except Exception:
                                pass
                        buffered_tool_events.clear()

                    await self._persist_result(session, run_row, result, inputs_hash)
                    return result
            finally:
                try:
                    _TOOL_EVENT_SINK_VAR.reset(token)
                except Exception:
                    pass

        effective_user_id = self.service_user_id or ctx.initiator_user_id or ""
        roles = ""  # workers run as system; keep minimal unless needed

        buffered_tool_events: list[ToolEvent] = []

        async def _buffer_sink(event: ToolEvent) -> None:
            buffered_tool_events.append(event)

        token = _TOOL_EVENT_SINK_VAR.set(_buffer_sink)
        try:
            await _emit_tool_event(
                event_type="tool_call",
                summary_text="get_tenant_session call",
                payload={
                    "tool": "get_tenant_session",
                    "roles_len": len(roles),
                    "clearance_level": 0,
                    "justification": "agent_pipeline",
                    "has_user_id": bool(effective_user_id),
                    "has_org_id": bool(ctx.org_id),
                },
            )
            start = time.perf_counter()

            async with contextlib.AsyncExitStack() as stack:
                try:
                    session = await stack.enter_async_context(
                        get_tenant_session(
                            user_id=effective_user_id,
                            org_id=ctx.org_id,
                            roles=roles,
                            clearance_level=0,
                            justification="agent_pipeline",
                        )
                    )
                except Exception as e:
                    duration_ms = (time.perf_counter() - start) * 1000.0
                    await _emit_tool_event(
                        event_type="tool_result",
                        summary_text="get_tenant_session failed",
                        payload={
                            "tool": "get_tenant_session",
                            "ok": False,
                            "duration_ms": round(duration_ms, 3),
                            "error": str(e)[:2000],
                        },
                    )
                    raise

                duration_ms = (time.perf_counter() - start) * 1000.0
                await _emit_tool_event(
                    event_type="tool_result",
                    summary_text="get_tenant_session ok",
                    payload={
                        "tool": "get_tenant_session",
                        "ok": True,
                        "duration_ms": round(duration_ms, 3),
                    },
                )

                audit = AuditService(session)

                content, existing_classification, scope, scope_id = await self._load_memory_inputs(session, ctx)
                enrichment = await self._load_prior_enrichment(session, ctx.org_id, ctx.memory_id)

                pending_feedback_fingerprint = ""
                if agent.name == "FeedbackLearningAgent":
                    pending_feedback_fingerprint = await self._load_pending_feedback_fingerprint(session, ctx.org_id, ctx.memory_id)

                inputs_hash = compute_inputs_hash(
                    [
                        agent.name,
                        agent.version,
                        ctx.org_id,
                        ctx.memory_id,
                        ctx.storage,
                        content,
                        existing_classification or "",
                        scope or "",
                        scope_id or "",
                        self._stable_json(enrichment or {}),
                        pending_feedback_fingerprint,
                    ]
                )

                await _emit_tool_event(
                    event_type="tool_call",
                    summary_text="compute_inputs_hash computed",
                    payload={"tool": "compute_inputs_hash", "inputs_hash_len": len(inputs_hash), "agent": agent.name},
                )

                run_row = await self._get_or_create_run_row(
                    session=session,
                    org_id=ctx.org_id,
                    memory_id=ctx.memory_id,
                    agent_name=agent.name,
                    agent_version=agent.version,
                    inputs_hash=inputs_hash,
                    trace_id=ctx.trace_id,
                    started_at=started_at,
                )

                tool_event_sink = self._create_tool_event_sink(session=session, run_row=run_row)
                _TOOL_EVENT_SINK_VAR.set(tool_event_sink)

                if buffered_tool_events:
                    for ev in buffered_tool_events:
                        try:
                            await tool_event_sink(ev)
                        except Exception:
                            pass
                    buffered_tool_events.clear()

                async def _emit_tool_event_local(*, event_type: str, summary_text: str, payload: dict) -> None:
                    await _emit_tool_event(event_type=event_type, summary_text=summary_text, payload=payload)

                # If already succeeded with same inputs, return stored outputs.
                if run_row.status == "success" and run_row.inputs_hash == inputs_hash:
                    finished_at = datetime.now(timezone.utc)
                    return AgentResult(
                        agent_name=run_row.agent_name,
                        agent_version=run_row.agent_version,
                        memory_id=run_row.memory_id,
                        status="success",
                        confidence=float(run_row.confidence or 0.0),
                        outputs=run_row.outputs or {},
                        warnings=run_row.warnings or [],
                        errors=run_row.errors or [],
                        started_at=run_row.started_at.replace(tzinfo=timezone.utc),
                        finished_at=run_row.finished_at.replace(tzinfo=timezone.utc),
                        trace_id=run_row.trace_id,
                    )

                # Cross-memory cache lookup (best-effort).
                cache_hit: Optional[dict] = None
                cache_confidence: float = 0.0
                strategy = self._cache_strategy(agent.name)
                model = self._cache_model()
                if (
                    self._cache_enabled()
                    and strategy == "llm"
                    and self._should_cache_agent(agent.name)
                    and content
                ):
                    try:
                        cache_key = self._compute_cache_key(
                            agent_name=agent.name,
                            agent_version=agent.version,
                            strategy=strategy,
                            model=model,
                            ctx=ctx,
                            content=content,
                            existing_classification=existing_classification,
                            scope=scope or "personal",
                            scope_id=scope_id,
                            enrichment=enrichment or {},
                            pending_feedback_fingerprint=pending_feedback_fingerprint,
                        )
                        await _emit_tool_event_local(
                            event_type="tool_call",
                            summary_text="AgentRunner._compute_cache_key computed",
                            payload={
                                "tool": "AgentRunner._compute_cache_key",
                                "agent": agent.name,
                                "inputs_hash_len": len(inputs_hash),
                                "cache_key_len": len(cache_key),
                            },
                        )
                        cache_svc = AgentResultCacheService(session)
                        await _emit_tool_event_local(
                            event_type="tool_call",
                            summary_text="AgentResultCacheService.get call",
                            payload={
                                "tool": "AgentResultCacheService.get",
                                "agent": agent.name,
                                "strategy": strategy,
                                "model": model,
                            },
                        )
                        start = time.perf_counter()
                        cached = await cache_svc.get(
                            organization_id=ctx.org_id,
                            agent_name=agent.name,
                            agent_version=agent.version,
                            strategy=strategy,
                            model=model,
                            cache_key=cache_key,
                        )
                        duration_ms = (time.perf_counter() - start) * 1000.0
                        await _emit_tool_event_local(
                            event_type="tool_result",
                            summary_text="AgentResultCacheService.get ok",
                            payload={
                                "tool": "AgentResultCacheService.get",
                                "agent": agent.name,
                                "ok": True,
                                "cache_hit": cached is not None,
                                "duration_ms": round(duration_ms, 3),
                            },
                        )
                        if cached is not None:
                            cache_hit = cached.outputs
                            cache_confidence = cached.confidence
                    except Exception as e:
                        await _emit_tool_event_local(
                            event_type="tool_result",
                            summary_text="AgentResultCacheService.get failed",
                            payload={
                                "tool": "AgentResultCacheService.get",
                                "agent": agent.name,
                                "ok": False,
                                "error": str(e)[:2000],
                            },
                        )
                        cache_hit = None

                if cache_hit is not None:
                    finished_at = datetime.now(timezone.utc)
                    cached_result = AgentResult(
                        agent_name=agent.name,
                        agent_version=agent.version,
                        memory_id=ctx.memory_id,
                        status="success",
                        confidence=float(cache_confidence or 0.0),
                        outputs=cache_hit or {},
                        warnings=["cache_hit"],
                        errors=[],
                        started_at=started_at,
                        finished_at=finished_at,
                        trace_id=ctx.trace_id,
                    )

                    await self._materialize_side_effects(
                        session=session,
                        ctx=ctx,
                        agent_name=agent.name,
                        result=cached_result,
                        scope=scope or "personal",
                        scope_id=scope_id,
                        tool_event_sink=tool_event_sink,
                    )
                    await self._persist_result(session, run_row, cached_result, inputs_hash)
                    return cached_result

                agent_ctx = {
                    "tenant": {"org_id": ctx.org_id, "org_slug": None},
                    "actor": {"user_id": ctx.initiator_user_id or "", "roles": []},
                    "memory": {
                        "id": ctx.memory_id,
                        "storage": ctx.storage,
                        "content": content,
                        "classification": existing_classification or "internal",
                        "scope": scope or "personal",
                        "scope_id": scope_id,
                        "enrichment": enrichment,
                    },
                    "runtime": {
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "job_id": ctx.trace_id or "",
                    },
                    "tool_event_sink": tool_event_sink,
                }

                try:
                    result = await agent.run(ctx.memory_id, agent_ctx)
                    agent.validate_outputs(result)

                    # Materialize agent side-effects in the same transaction.
                    await self._materialize_side_effects(
                        session=session,
                        ctx=ctx,
                        agent_name=agent.name,
                        result=result,
                        scope=scope or "personal",
                        scope_id=scope_id,
                        tool_event_sink=tool_event_sink,
                    )

                    # Best-effort cache write for successful LLM runs.
                    if (
                        self._cache_enabled()
                        and result.status == "success"
                        and strategy == "llm"
                        and self._should_cache_agent(agent.name)
                        and content
                    ):
                        try:
                            cache_key = self._compute_cache_key(
                                agent_name=agent.name,
                                agent_version=agent.version,
                                strategy=strategy,
                                model=model,
                                ctx=ctx,
                                content=content,
                                existing_classification=existing_classification,
                                scope=scope or "personal",
                                scope_id=scope_id,
                                enrichment=enrichment or {},
                                pending_feedback_fingerprint=pending_feedback_fingerprint,
                            )
                            await _emit_tool_event_local(
                                event_type="tool_call",
                                summary_text="AgentRunner._compute_cache_key computed",
                                payload={
                                    "tool": "AgentRunner._compute_cache_key",
                                    "agent": agent.name,
                                    "inputs_hash_len": len(inputs_hash),
                                    "cache_key_len": len(cache_key),
                                },
                            )
                            cache_svc = AgentResultCacheService(session)
                            await _emit_tool_event_local(
                                event_type="tool_call",
                                summary_text="AgentResultCacheService.upsert call",
                                payload={
                                    "tool": "AgentResultCacheService.upsert",
                                    "agent": agent.name,
                                    "strategy": strategy,
                                    "model": model,
                                    "outputs_key_count": len((result.outputs or {}).keys()),
                                },
                            )
                            start = time.perf_counter()
                            await cache_svc.upsert(
                                organization_id=ctx.org_id,
                                agent_name=agent.name,
                                agent_version=agent.version,
                                strategy=strategy,
                                model=model,
                                cache_key=cache_key,
                                outputs=result.outputs or {},
                                confidence=float(result.confidence or 0.0),
                                ttl_seconds=getattr(settings, "AGENT_CACHE_TTL_SECONDS", None),
                            )
                            duration_ms = (time.perf_counter() - start) * 1000.0
                            await _emit_tool_event_local(
                                event_type="tool_result",
                                summary_text="AgentResultCacheService.upsert ok",
                                payload={
                                    "tool": "AgentResultCacheService.upsert",
                                    "agent": agent.name,
                                    "ok": True,
                                    "duration_ms": round(duration_ms, 3),
                                },
                            )
                        except Exception as e:
                            await _emit_tool_event_local(
                                event_type="tool_result",
                                summary_text="AgentResultCacheService.upsert failed",
                                payload={
                                    "tool": "AgentResultCacheService.upsert",
                                    "agent": agent.name,
                                    "ok": False,
                                    "error": str(e)[:2000],
                                },
                            )
                            pass

                    await self._persist_result(session, run_row, result, inputs_hash)
                    return result
                except ValueError as e:
                    # validation error: no retries
                    finished_at = datetime.now(timezone.utc)
                    failed = AgentResult(
                        agent_name=agent.name,
                        agent_version=agent.version,
                        memory_id=ctx.memory_id,
                        status="failed",
                        confidence=0.0,
                        outputs={},
                        warnings=[],
                        errors=[str(e)],
                        started_at=started_at,
                        finished_at=finished_at,
                        trace_id=ctx.trace_id,
                    )
                    await self._persist_result(session, run_row, failed, inputs_hash)
                    await _emit_tool_event_local(
                        event_type="tool_call",
                        summary_text="AuditService.log_event call",
                        payload={
                            "tool": "AuditService.log_event",
                            "event_type": "system.agent_failed",
                            "severity": "error",
                        },
                    )
                    start = time.perf_counter()
                    try:
                        await audit.log_event(
                            event_type="system.agent_failed",
                            actor_id=effective_user_id or None,
                            actor_type="system",
                            organization_id=ctx.org_id,
                            resource_type="memory",
                            resource_id=ctx.memory_id,
                            success=False,
                            error_message=str(e),
                            details={"agent": agent.name, "version": agent.version, "attempt": attempt},
                            request_id=ctx.trace_id,
                            severity="error",
                        )
                        duration_ms = (time.perf_counter() - start) * 1000.0
                        await _emit_tool_event_local(
                            event_type="tool_result",
                            summary_text="AuditService.log_event ok",
                            payload={
                                "tool": "AuditService.log_event",
                                "ok": True,
                                "duration_ms": round(duration_ms, 3),
                            },
                        )
                    except Exception as audit_exc:
                        duration_ms = (time.perf_counter() - start) * 1000.0
                        await _emit_tool_event_local(
                            event_type="tool_result",
                            summary_text="AuditService.log_event failed",
                            payload={
                                "tool": "AuditService.log_event",
                                "ok": False,
                                "duration_ms": round(duration_ms, 3),
                                "error": str(audit_exc)[:2000],
                            },
                        )
                    raise
                except Exception as e:
                    finished_at = datetime.now(timezone.utc)
                    failed = AgentResult(
                        agent_name=agent.name,
                        agent_version=agent.version,
                        memory_id=ctx.memory_id,
                        status="retry" if attempt < max_attempts else "failed",
                        confidence=0.0,
                        outputs={},
                        warnings=[],
                        errors=[str(e)],
                        started_at=started_at,
                        finished_at=finished_at,
                        trace_id=ctx.trace_id,
                    )
                    await self._persist_result(session, run_row, failed, inputs_hash)
                    await _emit_tool_event_local(
                        event_type="tool_call",
                        summary_text="AuditService.log_event call",
                        payload={
                            "tool": "AuditService.log_event",
                            "event_type": "system.agent_failed",
                            "severity": "error",
                        },
                    )
                    start = time.perf_counter()
                    try:
                        await audit.log_event(
                            event_type="system.agent_failed",
                            actor_id=effective_user_id or None,
                            actor_type="system",
                            organization_id=ctx.org_id,
                            resource_type="memory",
                            resource_id=ctx.memory_id,
                            success=False,
                            error_message=str(e),
                            details={"agent": agent.name, "version": agent.version, "attempt": attempt},
                            request_id=ctx.trace_id,
                            severity="error",
                        )
                        duration_ms = (time.perf_counter() - start) * 1000.0
                        await _emit_tool_event_local(
                            event_type="tool_result",
                            summary_text="AuditService.log_event ok",
                            payload={
                                "tool": "AuditService.log_event",
                                "ok": True,
                                "duration_ms": round(duration_ms, 3),
                            },
                        )
                    except Exception as audit_exc:
                        duration_ms = (time.perf_counter() - start) * 1000.0
                        await _emit_tool_event_local(
                            event_type="tool_result",
                            summary_text="AuditService.log_event failed",
                            payload={
                                "tool": "AuditService.log_event",
                                "ok": False,
                                "duration_ms": round(duration_ms, 3),
                                "error": str(audit_exc)[:2000],
                            },
                        )
                    raise
        finally:
            try:
                _TOOL_EVENT_SINK_VAR.reset(token)
            except Exception:
                pass

    async def _materialize_side_effects(
        self,
        *,
        session: AsyncSession,
        ctx: PipelineContext,
        agent_name: str,
        result: AgentResult,
        scope: str,
        scope_id: Optional[str],
        tool_event_sink: ToolEventSink | None = None,
    ) -> None:
        if result.status != "success":
            return

        async def _emit_tool_event(*, event_type: str, summary_text: str, payload: dict) -> None:
            if tool_event_sink is None:
                return
            try:
                await tool_event_sink(
                    ToolEvent(
                        event_type=event_type,
                        summary_text=summary_text,
                        payload=payload,
                    )
                )
            except Exception:
                # Never let observability break materialization.
                return

        async def _audit_log_event(*, event_type: str, severity: str, kwargs: dict) -> None:
            await _emit_tool_event(
                event_type="tool_call",
                summary_text="AuditService.log_event call",
                payload={"tool": "AuditService.log_event", "event_type": event_type, "severity": severity},
            )
            start = time.perf_counter()
            try:
                audit = AuditService(session)
                await audit.log_event(**kwargs)
                duration_ms = (time.perf_counter() - start) * 1000.0
                await _emit_tool_event(
                    event_type="tool_result",
                    summary_text="AuditService.log_event ok",
                    payload={"tool": "AuditService.log_event", "ok": True, "duration_ms": round(duration_ms, 3)},
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000.0
                await _emit_tool_event(
                    event_type="tool_result",
                    summary_text="AuditService.log_event failed",
                    payload={
                        "tool": "AuditService.log_event",
                        "ok": False,
                        "duration_ms": round(duration_ms, 3),
                        "error": str(e)[:2000],
                    },
                )

        async def _run_side_effect(*, tool_name: str, call_payload: dict, op_coro):
            await _emit_tool_event(
                event_type="tool_call",
                summary_text=f"{tool_name} call",
                payload={"tool": tool_name, **call_payload},
            )

            start = time.perf_counter()
            try:
                out = await op_coro
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000.0
                await _emit_tool_event(
                    event_type="tool_result",
                    summary_text=f"{tool_name} failed",
                    payload={
                        "tool": tool_name,
                        "ok": False,
                        "duration_ms": round(duration_ms, 3),
                        "error": str(e)[:2000],
                        **call_payload,
                    },
                )
                raise

            duration_ms = (time.perf_counter() - start) * 1000.0
            await _emit_tool_event(
                event_type="tool_result",
                summary_text=f"{tool_name} ok",
                payload={
                    "tool": tool_name,
                    "ok": True,
                    "duration_ms": round(duration_ms, 3),
                    **call_payload,
                },
            )
            return out

        if agent_name == "GraphLinkingAgent":
            svc = GraphEdgeService(session)
            await _run_side_effect(
                tool_name="GraphEdgeService.upsert_edges_for_memory",
                call_payload={
                    "agent": agent_name,
                    "memory_id": ctx.memory_id,
                    "outputs_key_count": len((result.outputs or {}).keys()),
                },
                op_coro=svc.upsert_edges_for_memory(
                    organization_id=ctx.org_id,
                    memory_id=ctx.memory_id,
                    outputs=result.outputs,
                    created_by="agent",
                ),
            )

        if agent_name == "TopicModelingAgent":
            svc = TopicService(session)
            await _run_side_effect(
                tool_name="TopicService.upsert_topics_for_memory",
                call_payload={
                    "agent": agent_name,
                    "memory_id": ctx.memory_id,
                    "scope": scope or "personal",
                    "scope_id": scope_id,
                    "outputs_key_count": len((result.outputs or {}).keys()),
                },
                op_coro=svc.upsert_topics_for_memory(
                    organization_id=ctx.org_id,
                    memory_id=ctx.memory_id,
                    scope=scope or "personal",
                    scope_id=scope_id,
                    outputs=result.outputs,
                    created_by="agent",
                ),
            )

        if agent_name == "PatternDetectionAgent":
            svc = PatternService(session)
            await _run_side_effect(
                tool_name="PatternService.upsert_patterns_for_memory",
                call_payload={
                    "agent": agent_name,
                    "memory_id": ctx.memory_id,
                    "scope": scope or "personal",
                    "scope_id": scope_id,
                    "outputs_key_count": len((result.outputs or {}).keys()),
                },
                op_coro=svc.upsert_patterns_for_memory(
                    organization_id=ctx.org_id,
                    memory_id=ctx.memory_id,
                    scope=scope or "personal",
                    scope_id=scope_id,
                    outputs=result.outputs,
                    created_by="agent",
                ),
            )

        if agent_name == "FeedbackLearningAgent":
            svc = FeedbackLearningConfigService(session)
            applied = await _run_side_effect(
                tool_name="FeedbackLearningConfigService.apply_from_agent_outputs",
                call_payload={
                    "agent": agent_name,
                    "memory_id": ctx.memory_id,
                    "outputs_key_count": len((result.outputs or {}).keys()),
                },
                op_coro=svc.apply_from_agent_outputs(
                    organization_id=ctx.org_id,
                    source_memory_id=ctx.memory_id,
                    outputs=result.outputs,
                    updated_by_user_id=ctx.initiator_user_id or None,
                    agent_version=result.agent_version,
                    trace_id=result.trace_id,
                ),
            )
            if applied.get("config_updated"):
                await _audit_log_event(
                    event_type="policy.feedback_learning_config_updated",
                    severity="info",
                    kwargs={
                        "event_type": "policy.feedback_learning_config_updated",
                        "actor_id": ctx.initiator_user_id or None,
                        "actor_type": "agent",
                        "organization_id": ctx.org_id,
                        "resource_type": "organization",
                        "resource_id": ctx.org_id,
                        "success": True,
                        "details": {
                            "agent": "FeedbackLearningAgent",
                            "agent_version": result.agent_version,
                            "source_memory_id": ctx.memory_id,
                            **{k: applied.get(k) for k in ["stopwords_added", "thresholds_updated", "weights_updated"] if k in applied},
                        },
                        "request_id": result.trace_id,
                        "severity": "info",
                    },
                )

        if agent_name == "LogseqExportAgent":
            svc = LogseqExportPersistenceService(session)
            persisted = await _run_side_effect(
                tool_name="LogseqExportPersistenceService.upsert_export_for_memory",
                call_payload={
                    "agent": agent_name,
                    "memory_id": ctx.memory_id,
                    "outputs_key_count": len((result.outputs or {}).keys()),
                },
                op_coro=svc.upsert_export_for_memory(
                    organization_id=ctx.org_id,
                    memory_id=ctx.memory_id,
                    outputs=result.outputs,
                    updated_by_user_id=ctx.initiator_user_id or None,
                    agent_version=result.agent_version,
                    trace_id=result.trace_id,
                    created_by="agent",
                ),
            )
            if persisted.get("persisted"):
                await _audit_log_event(
                    event_type="logseq.export_materialized",
                    severity="info",
                    kwargs={
                        "event_type": "logseq.export_materialized",
                        "actor_id": ctx.initiator_user_id or None,
                        "actor_type": "agent",
                        "organization_id": ctx.org_id,
                        "resource_type": "memory",
                        "resource_id": ctx.memory_id,
                        "success": True,
                        "details": {
                            "agent": "LogseqExportAgent",
                            "agent_version": result.agent_version,
                            "item_count": persisted.get("item_count"),
                        },
                        "request_id": result.trace_id,
                        "severity": "info",
                    },
                )

    async def _load_pending_feedback_fingerprint(self, session: AsyncSession, org_id: str, memory_id: str) -> str:
        """Fingerprint unapplied feedback for idempotency.

        Without this, FeedbackLearningAgent could be incorrectly skipped when
        new feedback arrives but the memory content/enrichment is unchanged.
        """

        await _emit_tool_event(
            event_type="tool_call",
            summary_text="Pending feedback fingerprint call",
            payload={"tool": "MemoryFeedback.pending_fingerprint", "memory_id": memory_id},
        )
        start = time.perf_counter()

        stmt = (
            select(
                func.count(MemoryFeedback.id),
                func.max(MemoryFeedback.created_at),
            )
            .where(
                MemoryFeedback.organization_id == org_id,
                MemoryFeedback.memory_id == memory_id,
                MemoryFeedback.is_applied.is_(False),
            )
        )
        try:
            res = await session.execute(stmt)
            row = res.one_or_none()
            if not row:
                duration_ms = (time.perf_counter() - start) * 1000.0
                await _emit_tool_event(
                    event_type="tool_result",
                    summary_text="Pending feedback fingerprint ok",
                    payload={
                        "tool": "MemoryFeedback.pending_fingerprint",
                        "ok": True,
                        "duration_ms": round(duration_ms, 3),
                        "pending_count": 0,
                    },
                )
                return "0:"
            pending_count, max_created = row
            max_s = max_created.isoformat() if max_created is not None else ""
            try:
                c = int(pending_count or 0)
            except Exception:
                c = 0
            duration_ms = (time.perf_counter() - start) * 1000.0
            await _emit_tool_event(
                event_type="tool_result",
                summary_text="Pending feedback fingerprint ok",
                payload={
                    "tool": "MemoryFeedback.pending_fingerprint",
                    "ok": True,
                    "duration_ms": round(duration_ms, 3),
                    "pending_count": c,
                },
            )
            return f"{c}:{max_s}"
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            await _emit_tool_event(
                event_type="tool_result",
                summary_text="Pending feedback fingerprint failed",
                payload={
                    "tool": "MemoryFeedback.pending_fingerprint",
                    "ok": False,
                    "duration_ms": round(duration_ms, 3),
                    "error": str(e)[:2000],
                },
            )
            raise

    async def _load_memory_inputs(self, session: AsyncSession, ctx: PipelineContext) -> tuple[str, Optional[str], str, Optional[str]]:
        if ctx.storage == "short_term" and ctx.initiator_user_id:
            stm_service = ShortTermMemoryService(ctx.initiator_user_id, ctx.org_id)
            await _emit_tool_event(
                event_type="tool_call",
                summary_text="Short term memory fetch call",
                payload={"tool": "ShortTermMemoryService.get", "memory_id": ctx.memory_id},
            )
            start = time.perf_counter()
            try:
                stm = await stm_service.get(ctx.memory_id)
                duration_ms = (time.perf_counter() - start) * 1000.0
                found = stm is not None and bool(getattr(stm, "content", None))
                await _emit_tool_event(
                    event_type="tool_result",
                    summary_text="Short term memory fetch ok",
                    payload={
                        "tool": "ShortTermMemoryService.get",
                        "ok": True,
                        "duration_ms": round(duration_ms, 3),
                        "found": found,
                        "content_len": len(getattr(stm, "content", "") or "") if stm is not None else 0,
                    },
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000.0
                await _emit_tool_event(
                    event_type="tool_result",
                    summary_text="Short term memory fetch failed",
                    payload={
                        "tool": "ShortTermMemoryService.get",
                        "ok": False,
                        "duration_ms": round(duration_ms, 3),
                        "error": str(e)[:2000],
                    },
                )
                raise
            if stm is not None and stm.content:
                return stm.content, None, getattr(stm, "scope", "personal") or "personal", None

        # Fallback to LTM preview.
        await _emit_tool_event(
            event_type="tool_call",
            summary_text="Long term memory fetch call",
            payload={"tool": "MemoryMetadata.get", "memory_id": ctx.memory_id},
        )
        start = time.perf_counter()
        try:
            memory = await session.get(MemoryMetadata, ctx.memory_id)
            duration_ms = (time.perf_counter() - start) * 1000.0
            await _emit_tool_event(
                event_type="tool_result",
                summary_text="Long term memory fetch ok",
                payload={
                    "tool": "MemoryMetadata.get",
                    "ok": True,
                    "duration_ms": round(duration_ms, 3),
                    "found": memory is not None,
                    "content_preview_len": len((getattr(memory, "content_preview", "") or "")) if memory is not None else 0,
                },
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            await _emit_tool_event(
                event_type="tool_result",
                summary_text="Long term memory fetch failed",
                payload={
                    "tool": "MemoryMetadata.get",
                    "ok": False,
                    "duration_ms": round(duration_ms, 3),
                    "error": str(e)[:2000],
                },
            )
            raise
        if memory is None:
            return "", None, "personal", None
        return memory.content_preview or "", memory.classification, memory.scope or "personal", memory.scope_id

    async def _load_prior_enrichment(self, session: AsyncSession, org_id: str, memory_id: str) -> dict:
        """Load prior successful agent outputs to feed downstream agents."""

        await _emit_tool_event(
            event_type="tool_call",
            summary_text="Prior enrichment load call",
            payload={"tool": "AgentRun.load_prior_enrichment", "memory_id": memory_id},
        )
        start = time.perf_counter()

        stmt = select(AgentRun).where(
            AgentRun.organization_id == org_id,
            AgentRun.memory_id == memory_id,
            AgentRun.status == "success",
            AgentRun.agent_name.in_(
                [
                    "ClassificationAgent",
                    "MetadataExtractionAgent",
                    "TopicModelingAgent",
                    "PatternDetectionAgent",
                ]
            ),
        )
        try:
            res = await session.execute(stmt)
            rows = list(res.scalars().all())
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            await _emit_tool_event(
                event_type="tool_result",
                summary_text="Prior enrichment load failed",
                payload={
                    "tool": "AgentRun.load_prior_enrichment",
                    "ok": False,
                    "duration_ms": round(duration_ms, 3),
                    "error": str(e)[:2000],
                },
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000.0
        await _emit_tool_event(
            event_type="tool_result",
            summary_text="Prior enrichment load ok",
            payload={
                "tool": "AgentRun.load_prior_enrichment",
                "ok": True,
                "duration_ms": round(duration_ms, 3),
                "row_count": len(rows),
            },
        )

        out: dict = {}
        for r in rows:
            if r.agent_name == "ClassificationAgent":
                out["classification"] = r.outputs or {}
            elif r.agent_name == "MetadataExtractionAgent":
                out["metadata"] = r.outputs or {}
            if r.agent_name == "TopicModelingAgent":
                out["topics"] = r.outputs or {}
            if r.agent_name == "PatternDetectionAgent":
                out["patterns"] = r.outputs or {}

        return out

    async def _get_or_create_run_row(
        self,
        *,
        session: AsyncSession,
        org_id: str,
        memory_id: str,
        agent_name: str,
        agent_version: str,
        inputs_hash: str,
        trace_id: Optional[str],
        started_at: datetime,
    ) -> AgentRun:
        stmt = select(AgentRun).where(
            AgentRun.organization_id == org_id,
            AgentRun.memory_id == memory_id,
            AgentRun.agent_name == agent_name,
            AgentRun.agent_version == agent_version,
        )
        res = await session.execute(stmt)
        existing = res.scalar_one_or_none()
        if existing is not None:
            # If we've already succeeded for this exact input, keep the stored row unchanged.
            if existing.status == "success" and existing.inputs_hash == inputs_hash:
                return existing

            # Otherwise update inputs_hash/trace and reset timestamps for a new attempt.
            existing.inputs_hash = inputs_hash
            existing.trace_id = trace_id
            existing.started_at = started_at
            existing.finished_at = started_at

            await self._append_trajectory_event(
                session=session,
                org_id=org_id,
                agent_run_id=existing.id,
                memory_id=memory_id,
                event_type="run_started",
                step_index=0,
                payload={
                    "agent_name": agent_name,
                    "agent_version": agent_version,
                    "inputs_hash": inputs_hash,
                },
                summary_text=f"{agent_name}@{agent_version} started",
                trace_id=trace_id,
                created_at=started_at,
            )
            return existing

        row = AgentRun(
            organization_id=org_id,
            memory_id=memory_id,
            agent_name=agent_name,
            agent_version=agent_version,
            inputs_hash=inputs_hash,
            status="retry",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=started_at,
            trace_id=trace_id,
        )
        session.add(row)
        await session.flush()

        await self._append_trajectory_event(
            session=session,
            org_id=org_id,
            agent_run_id=row.id,
            memory_id=memory_id,
            event_type="run_started",
            step_index=0,
            payload={
                "agent_name": agent_name,
                "agent_version": agent_version,
                "inputs_hash": inputs_hash,
            },
            summary_text=f"{agent_name}@{agent_version} started",
            trace_id=trace_id,
            created_at=started_at,
        )
        return row

    async def _persist_result(self, session: AsyncSession, row: AgentRun, result: AgentResult, inputs_hash: str) -> None:
        row.inputs_hash = inputs_hash
        row.status = result.status
        row.confidence = float(result.confidence or 0.0)
        row.outputs = result.outputs or {}
        row.warnings = result.warnings or []
        row.errors = result.errors or []
        row.started_at = result.started_at
        row.finished_at = result.finished_at
        row.trace_id = result.trace_id
        row.provenance = list(getattr(result, "provenance", []) or [])
        await session.flush()

        await self._append_trajectory_event(
            session=session,
            org_id=row.organization_id,
            agent_run_id=row.id,
            memory_id=row.memory_id,
            event_type="run_result",
            step_index=1,
            payload={
                "status": result.status,
                "confidence": float(result.confidence or 0.0),
                "warnings_count": len(result.warnings or []),
                "errors": list(result.errors or []),
                "outputs_keys": sorted(list((result.outputs or {}).keys())),
            },
            summary_text=f"{row.agent_name}@{row.agent_version} {result.status} confidence={float(result.confidence or 0.0):.3f}",
            trace_id=result.trace_id,
            created_at=result.finished_at,
        )


    async def _append_trajectory_event(
        self,
        *,
        session: AsyncSession,
        org_id: str,
        agent_run_id: str,
        memory_id: str,
        event_type: str,
        step_index: int,
        payload: dict,
        summary_text: str,
        trace_id: Optional[str],
        created_at: datetime,
    ) -> None:
        """Best-effort event append.

        Must never fail agent execution.
        """

        try:
            session.add(
                AgentRunEvent(
                    organization_id=org_id,
                    agent_run_id=agent_run_id,
                    memory_id=memory_id,
                    event_type=event_type,
                    step_index=step_index,
                    payload=payload or {},
                    summary_text=summary_text or "",
                    created_at=created_at,
                    trace_id=trace_id,
                )
            )
            await session.flush()
        except Exception:
            return
