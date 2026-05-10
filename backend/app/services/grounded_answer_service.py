from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    uncertainty_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundedAnswerService:
    """Answer questions from a structured evidence package instead of flat snippets."""

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

        prompt = self._build_prompt(question=question, evidence_package=package)
        gateway_result = await self.gateway.answer(
            question=question,
            memories=list(memories or memory_hits),
            model=model,
            num_ctx=num_ctx,
            prompt_override=prompt,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
        )

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
            answer=gateway_result.answer,
            grounded=grounded,
            confidence=confidence,
            answer_source="grounded_answer_service",
            support=support,
            contradictions=contradictions,
            evidence_quality=evidence_quality,
            model=gateway_result.model,
            used_llm=bool(gateway_result.used_llm),
            context_turns=int(gateway_result.context_turns or 0),
            llm_error=gateway_result.llm_error,
            uncertainty_reason=uncertainty_reason,
        )

    def _build_prompt(self, *, question: str, evidence_package: dict[str, Any]) -> str:
        memory_hits = list(evidence_package.get("memory_hits") or [])
        facts = list(evidence_package.get("facts") or [])
        contradictions = list(evidence_package.get("contradictions") or [])
        episodes = list(evidence_package.get("episodes") or [])
        semantic_nodes = list(evidence_package.get("semantic_nodes") or [])
        temporal_reasoning = dict(evidence_package.get("temporal_reasoning") or {})
        multi_hop_trace = list(
            evidence_package.get("multi_hop_trace")
            or (evidence_package.get("planner_context") or {}).get("multi_hop_trace")
            or []
        )
        goal_context = dict(evidence_package.get("goal_context") or {})
        world_state = dict(goal_context.get("world_state") or {})

        memory_lines = []
        for hit in memory_hits[:6]:
            preview = str(hit.get("content_preview") or "").strip()
            title = str(hit.get("title") or "").strip()
            memory_id = str(hit.get("memory_id") or "").strip()
            parts = [part for part in [title, preview] if part]
            if parts:
                memory_lines.append(f"- [{memory_id}] {' | '.join(parts)}")

        fact_lines = []
        for fact in facts[:8]:
            fact_lines.append(
                f"- FACT: {fact.get('subject')} {fact.get('predicate')} {fact.get('object')} "
                f"(status={fact.get('status')}, confidence={self._safe_float(fact.get('confidence')):.2f})"
            )

        contradiction_lines = []
        for item in contradictions[:5]:
            contradiction_lines.append(
                f"- CONFLICT: {item.get('reason')} "
                f"(severity={item.get('severity')}, fact_a={item.get('fact_a_object')}, fact_b={item.get('fact_b_object')})"
            )

        episode_lines = []
        for episode in episodes[:4]:
            label = episode.get("title") or episode.get("episode_id")
            summary = episode.get("summary")
            if label or summary:
                episode_lines.append(f"- EPISODE: {label} | {summary}")

        semantic_lines = []
        for node in semantic_nodes[:4]:
            content = str(node.get("content") or "").strip()
            if content:
                semantic_lines.append(f"- NODE: {content}")

        timeline_lines = []
        for event in (temporal_reasoning.get("timeline") or [])[:5]:
            occurred_at = event.get("occurred_at") or "unknown_time"
            label = event.get("title") or event.get("content_preview") or event.get("memory_id")
            if label:
                timeline_lines.append(f"- EVENT: {occurred_at} | {label}")

        hop_lines = []
        for hop in multi_hop_trace[:4]:
            hop_lines.append(
                f"- HOP {hop.get('step_index')}: {hop.get('query')} "
                f"(count={hop.get('memory_count')}, confidence={self._safe_float(hop.get('confidence')):.2f})"
            )

        goal_lines = []
        for goal in (goal_context.get("active_goals") or [])[:4]:
            goal_lines.append(f"- GOAL: {goal.get('title')} (status={goal.get('status')}, urgency={goal.get('urgency')})")
        for gap in (goal_context.get("knowledge_gaps") or [])[:4]:
            goal_lines.append(f"- GAP: {gap.get('description')} (type={gap.get('gap_type')})")

        world_lines = []
        for change in (world_state.get("recent_changes") or [])[:4]:
            world_lines.append(f"- CHANGE: {change.get('entity')} -> {change.get('change_type')} | {change.get('description')}")

        sections = [
            "You are Ninai's grounded answer engine.",
            "Answer the question using only the structured evidence below.",
            "If evidence is missing or contradictory, say so briefly instead of guessing.",
            "Return only the answer text. No markdown, no bullet list, no extra explanation.",
            "",
            "MEMORY HITS:",
            "\n".join(memory_lines) or "- none",
            "",
            "FACTS:",
            "\n".join(fact_lines) or "- none",
            "",
            "CONTRADICTIONS:",
            "\n".join(contradiction_lines) or "- none",
            "",
            "EPISODES:",
            "\n".join(episode_lines) or "- none",
            "",
            "SEMANTIC NODES:",
            "\n".join(semantic_lines) or "- none",
            "",
            "TIMELINE:",
            "\n".join(timeline_lines) or "- none",
            "",
            "MULTI-HOP TRACE:",
            "\n".join(hop_lines) or "- none",
            "",
            "GOALS AND GAPS:",
            "\n".join(goal_lines) or "- none",
            "",
            "WORLD STATE:",
            "\n".join(world_lines) or "- none",
            "",
            f"QUESTION: {question}",
            "ANSWER:",
        ]
        return "\n".join(sections)

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

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
