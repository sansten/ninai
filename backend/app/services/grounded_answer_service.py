from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from app.services.cognitive_gateway_service import CognitiveGatewayService


@dataclass
class GroundedAnswerResult:
    answer: str
    grounded: bool
    confidence: float
    answer_source: str
    support: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    evidence_quality: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    used_llm: bool = False
    context_turns: int = 0
    llm_error: str | None = None
    llm_failure_mode: str | None = None
    llm_endpoint: str | None = None
    uncertainty_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundedAnswerService:
    """Answer questions from a structured evidence package instead of flat snippets."""

    _MAX_PROMPT_CHARS = 1600
    _MAX_FACT_LINES = 5
    _MAX_MEMORY_LINES = 5
    _MAX_TIMELINE_LINES = 3
    _MAX_SEMANTIC_LINES = 2
    _MAX_GRAPH_LINES = 2
    _MAX_GOAL_LINES = 2
    _MAX_ENTITY_LINES = 6

    def __init__(self, gateway: CognitiveGatewayService | None = None) -> None:
        self.gateway = gateway or CognitiveGatewayService()

    async def answer(
        self,
        *,
        question: str,
        evidence_package: dict[str, Any] | None,
        memories: list[dict[str, Any]] | None = None,
        model: str | None = None,
        num_ctx: int = 32768,
        timeout_seconds: float | None = None,
        keep_alive: int | None = None,
    ) -> GroundedAnswerResult:
        package = dict(evidence_package or {})
        memory_hits = list(package.get("memory_hits") or memories or [])
        facts = list(package.get("facts") or [])
        contradictions = list(package.get("contradictions") or [])
        evidence_quality = dict(package.get("evidence_quality") or {})
        question_profile = self._question_profile(question, package)

        grounded = bool(memory_hits or facts)
        if not grounded:
            return GroundedAnswerResult(
                answer="I don't have enough grounded evidence to answer that yet.",
                grounded=False,
                confidence=0.0,
                answer_source="grounded_empty",
                support=[],
                contradictions=contradictions,
                evidence_quality=evidence_quality,
                uncertainty_reason="no_grounded_evidence",
            )

        deterministic = self._deterministic_answer(
            question=question,
            package=package,
            memory_hits=memory_hits,
            facts=facts,
            contradictions=contradictions,
            evidence_quality=evidence_quality,
            question_profile=question_profile,
        )
        if deterministic is not None:
            return deterministic

        prompt = self._build_prompt(question=question, evidence_package=package)
        gateway_result = await self.gateway.answer(
            question=question,
            memories=[],
            model=model,
            num_ctx=min(int(num_ctx or 32768), 1536),
            prompt_override=prompt,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
        )
        final_answer = str(gateway_result.answer or "").strip()
        final_source = str(getattr(gateway_result, "answer_source", "") or "gateway_empty")
        final_model = gateway_result.model
        final_used_llm = bool(gateway_result.used_llm)
        final_error = gateway_result.llm_error
        final_failure_mode = getattr(gateway_result, "llm_failure_mode", None)
        final_endpoint = getattr(gateway_result, "llm_endpoint", None)

        if self._should_compress_answer(question_profile, final_answer):
            compressed = await self._compress_answer(
                question=question,
                profile=question_profile,
                draft_answer=final_answer,
                evidence_package=package,
                model=model,
                num_ctx=min(int(num_ctx or 32768), 8192),
                timeout_seconds=min(float(timeout_seconds or 60.0), 60.0),
                keep_alive=keep_alive,
            )
            compressed_answer = str(compressed.answer or "").strip()
            if compressed_answer:
                final_answer = compressed_answer
                final_source = f"{final_source}_compressed"
                final_model = compressed.model
                final_used_llm = bool(compressed.used_llm)
                final_error = compressed.llm_error
                final_failure_mode = getattr(compressed, "llm_failure_mode", None)
                final_endpoint = getattr(compressed, "llm_endpoint", None)

        confidence = self._estimate_confidence(
            evidence_quality=evidence_quality,
            fact_count=len(facts),
            contradiction_count=len(contradictions),
        )
        support = self._support_lines(package)

        uncertainty_reason = None
        if contradictions:
            uncertainty_reason = "contradictory_evidence_present"

        return GroundedAnswerResult(
            answer=final_answer,
            grounded=grounded,
            confidence=confidence,
            answer_source=final_source,
            support=support,
            contradictions=contradictions,
            evidence_quality=evidence_quality,
            model=final_model,
            used_llm=final_used_llm,
            context_turns=int(gateway_result.context_turns or 0),
            llm_error=final_error,
            llm_failure_mode=final_failure_mode,
            llm_endpoint=final_endpoint,
            uncertainty_reason=uncertainty_reason,
        )

    def _build_prompt(self, *, question: str, evidence_package: dict[str, Any]) -> str:
        memory_hits = list(evidence_package.get("memory_hits") or [])
        facts = list(evidence_package.get("facts") or [])
        contradictions = list(evidence_package.get("contradictions") or [])
        temporal_reasoning = dict(evidence_package.get("temporal_reasoning") or {})
        semantic_nodes = list(evidence_package.get("semantic_nodes") or [])
        graph_neighbors = list(evidence_package.get("graph_neighbors") or [])
        goal_context = dict(evidence_package.get("goal_context") or {})
        entity_context = dict(evidence_package.get("entity_context") or {})
        multi_hop_trace = list(
            evidence_package.get("multi_hop_trace")
            or (evidence_package.get("planner_context") or {}).get("multi_hop_trace")
            or []
        )
        question_profile = self._question_profile(question, evidence_package)
        relevant_facts = self._rank_facts(facts, question_profile)[: self._MAX_FACT_LINES]
        relevant_memories = self._rank_memories(memory_hits, question_profile)[: self._MAX_MEMORY_LINES]
        relevant_timeline = self._rank_timeline_events(
            list(temporal_reasoning.get("timeline") or []),
            question_profile,
        )[: self._MAX_TIMELINE_LINES]
        semantic_lines = self._semantic_lines(semantic_nodes)
        graph_lines = self._graph_lines(graph_neighbors)
        goal_lines = self._goal_lines(goal_context)
        entity_lines = self._entity_context_lines(entity_context)
        answer_hints = self._answer_hints(
            profile=question_profile,
            facts=relevant_facts,
            memories=relevant_memories,
            timeline=relevant_timeline,
        )

        memory_lines = []
        for hit in relevant_memories:
            preview = str(hit.get("content_preview") or hit.get("content") or "").strip()
            memory_id = str(hit.get("memory_id") or "").strip()
            preview = re.sub(r"\s+", " ", preview)
            if preview:
                memory_lines.append(f"- MEMORY {memory_id or '?'}: {preview[:160]}")

        fact_lines = []
        for fact in relevant_facts:
            fact_lines.append(
                f"- FACT: {fact.get('subject')} | {fact.get('predicate')} | {fact.get('object')}"
            )

        contradiction_lines = []
        for item in contradictions[:2]:
            contradiction_lines.append(
                f"- CONFLICT: {str(item.get('reason') or '').strip()[:140]}"
            )

        timeline_lines = []
        for event in relevant_timeline:
            occurred_at = event.get("occurred_at") or "unknown_time"
            label = event.get("title") or event.get("content_preview") or event.get("memory_id")
            if label:
                timeline_lines.append(f"- TIME: {occurred_at} | {str(label).strip()[:90]}")

        hop_lines = []
        for hop in multi_hop_trace[:2]:
            hop_lines.append(
                f"- HOP: {str(hop.get('query') or '').strip()[:120]}"
            )

        sections = [
            "You are Ninai's grounded answer engine.",
            "Answer using only the evidence below.",
            (
                f"QUESTION PROFILE: mode={question_profile['mode']}, "
                f"primary_subject={question_profile.get('primary_subject') or 'none'}, "
                f"distractors={', '.join(question_profile.get('distractors') or []) or 'none'}"
            ),
            "Rules:",
            "- Prefer FACT lines over narrative memory text.",
            "- Answer about the PRIMARY SUBJECT, not a distractor.",
            "- For temporal questions, return only the resolved date or time phrase.",
            "- For direct questions, return only the exact person, place, date, attribute, or object phrase.",
            "- For action, plan, research, or realization questions, return the object or outcome, not the quoted sentence.",
            "- If evidence conflicts, use the strongest support and keep the answer brief.",
            "- Return only the answer text.",
            "ANSWER HINTS:",
            "\n".join(answer_hints) or "- none",
            "FACTS:",
            "\n".join(fact_lines) or "- none",
            "MEMORY HITS:",
            "\n".join(memory_lines) or "- none",
            "TIMELINE:",
            "\n".join(timeline_lines) or "- none",
            "ENTITY RESOLUTION:",
            "\n".join(entity_lines) or "- none",
            "SEMANTIC NODES:",
            "\n".join(semantic_lines) or "- none",
            "GRAPH SIGNALS:",
            "\n".join(graph_lines) or "- none",
            "GOAL CONTEXT:",
            "\n".join(goal_lines) or "- none",
            "MULTI-HOP TRACE:",
            "\n".join(hop_lines) or "- none",
        ]
        if contradiction_lines:
            sections.extend([
                "CONTRADICTIONS:",
                "\n".join(contradiction_lines),
            ])
        sections.extend([
            f"QUESTION: {question}",
            "ANSWER:",
        ])
        prompt = "\n".join(sections)
        if len(prompt) <= self._MAX_PROMPT_CHARS:
            return prompt
        return self._truncate_prompt(
            prompt=prompt,
            question=question,
            question_profile=question_profile,
            answer_hints=answer_hints,
            fact_lines=fact_lines,
            memory_lines=memory_lines,
            timeline_lines=timeline_lines,
            entity_lines=entity_lines,
            semantic_lines=semantic_lines,
            graph_lines=graph_lines,
            goal_lines=goal_lines,
            hop_lines=hop_lines,
            contradiction_lines=contradiction_lines,
        )

    def _truncate_prompt(
        self,
        *,
        prompt: str,
        question: str,
        question_profile: dict[str, Any],
        answer_hints: list[str],
        fact_lines: list[str],
        memory_lines: list[str],
        timeline_lines: list[str],
        entity_lines: list[str],
        semantic_lines: list[str],
        graph_lines: list[str],
        goal_lines: list[str],
        hop_lines: list[str],
        contradiction_lines: list[str],
    ) -> str:
        sections = [
            "You are Ninai's grounded answer engine.",
            (
                f"mode={question_profile['mode']}; "
                f"primary_subject={question_profile.get('primary_subject') or 'none'}; "
                f"distractors={', '.join(question_profile.get('distractors') or []) or 'none'}"
            ),
            "Use only the evidence below and return only the final answer text.",
            "ANSWER HINTS:",
            "\n".join(answer_hints[:2]) or "- none",
            "FACTS:",
            "\n".join(fact_lines[:3]) or "- none",
            "MEMORY HITS:",
            "\n".join(memory_lines[:2]) or "- none",
        ]
        if timeline_lines:
            sections.extend(["TIMELINE:", "\n".join(timeline_lines[:2])])
        if entity_lines:
            sections.extend(["ENTITY RESOLUTION:", "\n".join(entity_lines[:2])])
        if semantic_lines:
            sections.extend(["SEMANTIC NODES:", "\n".join(semantic_lines[:1])])
        if graph_lines:
            sections.extend(["GRAPH SIGNALS:", "\n".join(graph_lines[:1])])
        if goal_lines:
            sections.extend(["GOAL CONTEXT:", "\n".join(goal_lines[:1])])
        if hop_lines:
            sections.extend(["MULTI-HOP TRACE:", "\n".join(hop_lines[:1])])
        if contradiction_lines:
            sections.extend(["CONTRADICTIONS:", "\n".join(contradiction_lines[:1])])
        sections.extend([f"QUESTION: {question}", "ANSWER:"])
        compact = "\n".join(sections)
        if len(compact) <= self._MAX_PROMPT_CHARS:
            return compact
        return compact[: self._MAX_PROMPT_CHARS].rstrip()

    async def _compress_answer(
        self,
        *,
        question: str,
        profile: dict[str, Any],
        draft_answer: str,
        evidence_package: dict[str, Any],
        model: str | None,
        num_ctx: int,
        timeout_seconds: float | None,
        keep_alive: int | None,
    ):
        return await self.gateway.answer(
            question=question,
            memories=[],
            model=model,
            num_ctx=num_ctx,
            prompt_override=self._compression_prompt(
                question=question,
                profile=profile,
                draft_answer=draft_answer,
                evidence_package=evidence_package,
            ),
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
        )

    def _deterministic_answer(
        self,
        *,
        question: str,
        package: dict[str, Any],
        memory_hits: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
        evidence_quality: dict[str, Any],
        question_profile: dict[str, Any],
    ) -> GroundedAnswerResult | None:
        ranked_facts = self._rank_facts(facts, question_profile)
        ranked_timeline = self._rank_timeline_events(
            list((package.get("temporal_reasoning") or {}).get("timeline") or []),
            question_profile,
        )
        if contradictions:
            return None
        confidence = self._estimate_confidence(
            evidence_quality=evidence_quality,
            fact_count=len(facts),
            contradiction_count=len(contradictions),
        )
        uncertainty_reason = "contradictory_evidence_present" if contradictions else None
        support = self._support_lines(package)
        mode = str(question_profile.get("mode") or "single_value")

        direct_answer = ""
        answer_source = ""
        direct_fact: dict[str, Any] | None = None

        if mode == "temporal":
            direct_fact = self._best_temporal_fact(ranked_facts)
            if direct_fact is not None:
                direct_answer = str(direct_fact.get("object") or "").strip()
            elif ranked_timeline:
                direct_answer = str(ranked_timeline[0].get("occurred_at") or "").strip()
            answer_source = self._fact_answer_source(direct_fact, default="timeline_direct")
        elif mode == "single_value":
            direct_fact = self._best_direct_fact(ranked_facts, question_profile)
            if direct_fact is not None:
                direct_answer = str(direct_fact.get("object") or "").strip()
                answer_source = self._fact_answer_source(direct_fact)
        elif mode == "list":
            values = self._best_list_values(ranked_facts, question_profile)
            if values:
                direct_answer = ", ".join(values)
                answer_source = "fact_list_direct"

        if not direct_answer:
            return None

        return GroundedAnswerResult(
            answer=direct_answer,
            grounded=True,
            confidence=confidence,
            answer_source=answer_source or "fact_direct",
            support=support,
            contradictions=contradictions,
            evidence_quality=evidence_quality,
            model=None,
            used_llm=False,
            context_turns=0,
            llm_error=None,
            llm_failure_mode=None,
            llm_endpoint=None,
            uncertainty_reason=uncertainty_reason,
        )

    def _support_lines(self, evidence_package: dict[str, Any]) -> list[str]:
        support: list[str] = []
        for fact in list(evidence_package.get("facts") or [])[:3]:
            support.append(
                f"fact:{fact.get('subject')} {fact.get('predicate')} {fact.get('object')}"
            )
        for hit in list(evidence_package.get("memory_hits") or [])[:2]:
            title = str(hit.get("title") or "").strip()
            preview = str(hit.get("content_preview") or "").strip()
            label = title or preview[:80]
            if label:
                support.append(f"memory:{label}")
        for event in list((evidence_package.get("temporal_reasoning") or {}).get("timeline") or [])[:2]:
            label = str(event.get("title") or event.get("content_preview") or "").strip()
            if label:
                support.append(f"timeline:{label}")
        for entity in list((evidence_package.get("entity_context") or {}).get("entities") or [])[:1]:
            name = str(entity.get("canonical_name") or "").strip()
            if name:
                support.append(f"entity:{name}")
        return support

    def _estimate_confidence(
        self,
        *,
        evidence_quality: dict[str, Any],
        fact_count: int,
        contradiction_count: int,
    ) -> float:
        avg_memory_score = self._safe_float(evidence_quality.get("avg_memory_score"))
        avg_semantic_quality = self._safe_float(evidence_quality.get("avg_semantic_quality"))
        avg_feedback_signal = self._safe_float(evidence_quality.get("avg_feedback_signal"))

        confidence = 0.25
        confidence += min(0.25, avg_memory_score * 0.3)
        confidence += min(0.2, avg_semantic_quality * 0.25)
        confidence += min(0.1, max(0.0, avg_feedback_signal) * 0.1)
        if fact_count > 0:
            confidence += 0.15
        if contradiction_count > 0:
            confidence -= min(0.25, contradiction_count * 0.08)
        return round(max(0.0, min(0.95, confidence)), 4)

    @classmethod
    def _question_profile(cls, question: str, evidence_package: dict[str, Any]) -> dict[str, Any]:
        lowered = f" {str(question or '').lower()} "
        entities: list[str] = []
        for item in list((evidence_package.get("query_intelligence") or {}).get("extracted_entities") or []):
            text = str(item).strip()
            if text and text not in entities:
                entities.append(text)
        for fact in list(evidence_package.get("facts") or [])[:10]:
            subject = str(fact.get("subject") or "").strip()
            if subject and subject not in entities:
                entities.append(subject)
            if len(entities) >= 4:
                break
        proper_names = [
            match.group(0).strip()
            for match in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+){0,2}\b", str(question or ""))
        ]
        for name in proper_names:
            if name not in entities:
                entities.append(name)
            if len(entities) >= 4:
                break

        mode = "single_value"
        if any(token in lowered for token in (" when ", " what date ", " what time ", " how long ")):
            mode = "temporal"
        elif any(token in lowered for token in (" compare ", " difference ", " versus ", " vs ", " contrast ")):
            mode = "comparison"
        elif any(token in lowered for token in (" why ", " because ", " due to ", " explain ", " how does ", " how do ", " how did ")):
            mode = "explanation"
        elif any(token in lowered for token in (" list ", " all ", " activities ", " events ", " things ", " types ", " kinds ")):
            mode = "list"
        elif any(token in lowered for token in (" what were ",)):
            mode = "list"

        compositional = any(
            token in lowered
            for token in (" before ", " after ", " still ", " because ", " due to ", " while ", " during ", " then ", " and ")
        )
        return {
            "mode": mode,
            "compositional": compositional,
            "entities": entities[:4],
            "primary_subject": entities[0] if entities else None,
            "distractors": entities[1:4],
            "tokens": cls._query_tokens(question, limit=16),
        }

    @classmethod
    def _should_compress_answer(cls, profile: dict[str, Any], answer: str) -> bool:
        text = str(answer or "").strip()
        if not text:
            return False
        mode = str(profile.get("mode") or "single_value")
        words = re.findall(r"\b\w+\b", text)
        lowered = text.lower()
        if mode == "temporal":
            return len(words) > 5 or any(
                phrase in lowered
                for phrase in ("attended", "went", "happened", "occurred", "ran ", "planned", "on ")
            )
        if mode == "single_value":
            return len(words) > 6 or any(
                phrase in lowered
                for phrase in (" is ", " are ", " was ", " were ", " identifies as ", " realized ", " moved from ")
            )
        return False

    @classmethod
    def _compression_prompt(
        cls,
        *,
        question: str,
        profile: dict[str, Any],
        draft_answer: str,
        evidence_package: dict[str, Any],
    ) -> str:
        facts = cls._rank_facts(list(evidence_package.get("facts") or []), profile)[:5]
        memories = cls._rank_memories(list(evidence_package.get("memory_hits") or []), profile)[:4]
        timeline = cls._rank_timeline_events(
            list((evidence_package.get("temporal_reasoning") or {}).get("timeline") or []),
            profile,
        )[:4]

        fact_lines = [
            f"- FACT: {fact.get('subject')} {fact.get('predicate')} {fact.get('object')}"
            for fact in facts
        ]
        memory_lines = []
        for memory in memories:
            title = str(memory.get("title") or memory.get("memory_id") or "").strip()
            preview = str(memory.get("content_preview") or memory.get("content") or "").strip()
            if title or preview:
                memory_lines.append(f"- MEMORY: {title} | {preview[:180]}".strip())
        timeline_lines = [
            f"- TIME: {item.get('occurred_at')} | {item.get('title') or item.get('memory_id')}"
            for item in timeline
        ]

        answer_shape = "a short exact value"
        if profile.get("mode") == "temporal":
            answer_shape = "the exact resolved date or time phrase only"
        elif profile.get("mode") == "single_value":
            answer_shape = "the exact person, place, attribute, or object phrase only"

        sections = [
            "Rewrite the draft answer into the shortest grounded answer that directly satisfies the question.",
            f"Return {answer_shape}.",
            "Do not include narration, hedging, or a full sentence when a short value is available.",
            "If the draft answer is already the shortest grounded answer, return it unchanged.",
            "Return only the final answer text.",
            "",
            f"MODE: {profile.get('mode')}",
            f"QUESTION: {question}",
            f"DRAFT ANSWER: {draft_answer}",
            "",
            "FACTS:",
            "\n".join(fact_lines) or "- none",
            "",
            "MEMORIES:",
            "\n".join(memory_lines) or "- none",
            "",
            "TIMELINE:",
            "\n".join(timeline_lines) or "- none",
            "",
            "FINAL ANSWER:",
        ]
        return "\n".join(sections)

    @classmethod
    def _best_direct_fact(
        cls,
        ranked_facts: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        subject_focus = str(profile.get("primary_subject") or "").strip().lower()
        for fact in ranked_facts[:4]:
            status = str(fact.get("status") or "active").strip().lower()
            obj = str(fact.get("object") or "").strip()
            subject = str(fact.get("subject") or "").strip().lower()
            predicate = str(fact.get("predicate") or "").strip().lower()
            if status not in {"active", ""} or not obj:
                continue
            if subject_focus and subject and subject != subject_focus and subject_focus not in obj.lower():
                continue
            if predicate in {"said", "asked", "mentioned", "talked_about"}:
                continue
            return fact
        return None

    @classmethod
    def _best_temporal_fact(cls, ranked_facts: list[dict[str, Any]]) -> dict[str, Any] | None:
        for fact in ranked_facts[:4]:
            if cls._looks_temporal(str(fact.get("object") or "")):
                return fact
        return None

    @classmethod
    def _best_list_values(
        cls,
        ranked_facts: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> list[str]:
        subject_focus = str(profile.get("primary_subject") or "").strip().lower()
        values: list[str] = []
        seen: set[str] = set()
        for fact in ranked_facts[:6]:
            status = str(fact.get("status") or "active").strip().lower()
            subject = str(fact.get("subject") or "").strip().lower()
            obj = str(fact.get("object") or "").strip()
            if status not in {"active", ""} or not obj:
                continue
            if subject_focus and subject and subject != subject_focus:
                continue
            key = obj.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(obj)
        return values[:4]

    @staticmethod
    def _fact_answer_source(fact: dict[str, Any] | None, default: str = "fact_direct") -> str:
        if not isinstance(fact, dict):
            return default
        source_type = str(fact.get("source_type") or "").strip().lower()
        source_memory_id = str(fact.get("source_memory_id") or "").strip().lower()
        if source_type == "state_space" or source_memory_id.startswith("state::"):
            return "state_space_direct"
        return default

    def _semantic_lines(self, semantic_nodes: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for node in semantic_nodes[: self._MAX_SEMANTIC_LINES]:
            content = str(node.get("content") or "").strip()
            if content:
                lines.append(f"- SEMANTIC: {content[:140]}")
        return lines

    def _graph_lines(self, graph_neighbors: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for edge in graph_neighbors[: self._MAX_GRAPH_LINES]:
            source_type = str(edge.get("source_type") or "").strip()
            target_type = str(edge.get("target_type") or "").strip()
            similarity = self._safe_float(edge.get("similarity"))
            if source_type or target_type:
                lines.append(f"- GRAPH: {source_type} -> {target_type} ({similarity:.2f})")
        return lines

    def _goal_lines(self, goal_context: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        for goal in list(goal_context.get("active_goals") or [])[:1]:
            title = str(goal.get("title") or "").strip()
            if title:
                lines.append(f"- GOAL: {title}")
        for gap in list(goal_context.get("knowledge_gaps") or [])[:1]:
            description = str(gap.get("description") or "").strip()
            if description:
                lines.append(f"- GAP: {description[:120]}")
        return lines[: self._MAX_GOAL_LINES]

    def _entity_context_lines(self, entity_context: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        entities = list(entity_context.get("entities") or [])
        for entity in entities[:2]:
            canonical = str(entity.get("canonical_name") or "").strip()
            aliases = [str(item).strip() for item in list(entity.get("aliases") or []) if str(item).strip()]
            links = list(entity.get("entity_links") or [])
            facts = list(entity.get("facts") or [])
            if not canonical:
                continue
            alias_text = ", ".join(aliases[:3]) if aliases else "none"
            role = "primary" if entity.get("is_primary_subject") else "related"
            lines.append(f"- ENTITY {role}: {canonical} | aliases={alias_text}")
            if links:
                rendered_links = ", ".join(
                    f"{str(link.get('predicate') or '').strip()}->{str(link.get('entity') or '').strip()}"
                    for link in links[:2]
                    if str(link.get("predicate") or "").strip() and str(link.get("entity") or "").strip()
                )
                if rendered_links:
                    lines.append(f"- LINKS: {rendered_links}")
            if facts:
                top_fact = facts[0]
                lines.append(
                    f"- ENTITY FACT: {top_fact.get('subject')} | {top_fact.get('predicate')} | {top_fact.get('object')}"
                )
            memory_mentions = list(entity.get("memory_mentions") or [])
            if memory_mentions:
                mention = memory_mentions[0]
                preview = str(mention.get("content_preview") or "").strip()
                if preview:
                    lines.append(f"- ENTITY MEMORY: {preview[:120]}")
        return lines[: self._MAX_ENTITY_LINES]

    @classmethod
    def _rank_facts(cls, facts: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        subject_focus = str(profile.get("primary_subject") or "").strip().lower()
        distractors = [str(item).strip().lower() for item in (profile.get("distractors") or []) if str(item).strip()]
        for fact in facts:
            subject = str(fact.get("subject") or "").strip().lower()
            predicate = str(fact.get("predicate") or "").strip().lower()
            obj = str(fact.get("object") or "").strip().lower()
            score = cls._safe_float(fact.get("confidence"))
            if subject_focus:
                if subject_focus == subject:
                    score += 1.5
                elif subject_focus in obj:
                    score += 0.8
            if distractors and subject_focus and subject != subject_focus and any(item == subject for item in distractors):
                score -= 0.45
            for entity in profile["entities"]:
                lowered = entity.lower()
                if lowered == subject:
                    score += 1.2
                elif lowered in obj:
                    score += 0.7
            for token in profile["tokens"]:
                if token == predicate:
                    score += 0.8
                elif token in predicate:
                    score += 0.4
                elif token in obj or token in subject:
                    score += 0.25
            if profile["mode"] == "temporal" and cls._looks_temporal(obj):
                score += 0.9
            scored.append((score, fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    @classmethod
    def _rank_memories(cls, memories: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        subject_focus = str(profile.get("primary_subject") or "").strip().lower()
        distractors = [str(item).strip().lower() for item in (profile.get("distractors") or []) if str(item).strip()]
        for memory in memories:
            title = str(memory.get("title") or "").strip().lower()
            preview = str(memory.get("content_preview") or memory.get("content") or "").strip().lower()
            combined = f"{title} {preview}"
            score = cls._safe_float(memory.get("score"))
            if subject_focus:
                if subject_focus in combined:
                    score += 1.2
                elif any(token in combined for token in re.findall(r"\b[a-z0-9_]{3,}\b", subject_focus)):
                    score += 0.5
            if distractors and subject_focus and subject_focus not in combined:
                distractor_hits = sum(1 for item in distractors if item in combined)
                score -= min(0.45, distractor_hits * 0.2)
            for entity in profile["entities"]:
                lowered = entity.lower()
                if lowered and lowered in combined:
                    score += 0.9
            for token in profile["tokens"]:
                if token and token in combined:
                    score += 0.18
            if profile["mode"] == "temporal" and cls._looks_temporal(combined):
                score += 0.6
            if profile["mode"] == "list" and "," in preview:
                score += 0.1
            scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    @classmethod
    def _rank_timeline_events(cls, timeline: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        subject_focus = str(profile.get("primary_subject") or "").strip().lower()
        for event in timeline:
            label = " ".join(
                str(event.get(key) or "").strip().lower()
                for key in ("title", "content_preview", "memory_id")
            )
            score = 0.2
            occurred_at = str(event.get("occurred_at") or "").strip()
            if occurred_at:
                score += 0.4
            if subject_focus and subject_focus in label:
                score += 1.0
            for entity in profile["entities"]:
                lowered = entity.lower()
                if lowered and lowered in label:
                    score += 0.8
            for token in profile["tokens"]:
                if token and token in label:
                    score += 0.2
            scored.append((score, event))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    @classmethod
    def _answer_hints(
        cls,
        *,
        profile: dict[str, Any],
        facts: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        timeline: list[dict[str, Any]],
    ) -> list[str]:
        hints: list[str] = []
        for fact in facts[:4]:
            hints.append(
                f"- PRIORITY FACT: {fact.get('subject')} {fact.get('predicate')} {fact.get('object')}"
            )
        if profile["mode"] == "temporal":
            for event in timeline[:3]:
                hints.append(
                    f"- TIME CANDIDATE: {event.get('occurred_at')} | {event.get('title') or event.get('memory_id')}"
                )
        for memory in memories[:2]:
            title = str(memory.get("title") or "").strip()
            preview = str(memory.get("content_preview") or memory.get("content") or "").strip()
            parts = [part for part in [title, preview[:120]] if part]
            if parts:
                hints.append(f"- PRIORITY MEMORY: {' | '.join(parts)}")
        return hints[:8]

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _query_tokens(cls, text: str, *, limit: int = 16) -> list[str]:
        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "how",
            "with",
            "from",
            "that",
            "this",
            "were",
            "they",
            "them",
            "their",
            "about",
            "have",
            "does",
            "did",
            "into",
            "while",
        }
        tokens: list[str] = []
        for token in re.findall(r"\b[a-z0-9_]{3,}\b", str(text or "").lower()):
            if token in stopwords or token in tokens:
                continue
            tokens.append(token)
            if len(tokens) >= max(1, int(limit or 1)):
                break
        return tokens

    @staticmethod
    def _looks_temporal(text: str) -> bool:
        value = str(text or "").lower()
        if any(
            token in value
            for token in (
                "january",
                "february",
                "march",
                "april",
                "may ",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            )
        ):
            return True
        return bool(re.search(r"\b\d{4}\b|\b\d{1,2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b", value))
