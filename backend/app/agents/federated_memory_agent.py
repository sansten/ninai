"""Federated Memory & Collective Intelligence — Phase 44.

Privacy-preserving cross-organization knowledge synthesis.

Builds on existing PR-10 federated foundations (services/models/endpoints) and
provides an AGI enrichment agent that converts federated signals into
actionable, tenant-safe outputs for downstream planning and reporting.

Inputs (via context["memory"]["enrichment"]):
  - federated_candidates: list[dict]
      Each candidate can include:
      {domain, summary_type, quality_score, count_contributing_orgs,
       content, success_rate}
  - org_metrics: dict[str, float]
      Organization metrics for benchmarking (e.g., response_time, quality)
  - peer_metrics: dict[str, list[float]]
      Peer-group metric distributions
  - benchmark_metric: str (optional)
  - target_percentile: float 0..1 (optional, default 0.75)
  - min_contributors: int (optional, default 5)
  - privacy_budget_epsilon: float (optional, default 1.0)

Outputs:
  - federated_summary: dict
  - benchmark_insight: dict
  - sharing_recommendation: str
  - privacy_risk_score: float 0..1
  - federated_confidence: float 0..1
  - confidence: float 0..1
  - rationale: "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.llm_breaker import create_llm_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings
from app.services.differential_privacy_service import DifferentialPrivacyService


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def filter_k_anonymous_candidates(
    candidates: list[dict[str, Any]],
    *,
    min_contributors: int,
) -> list[dict[str, Any]]:
    """Keep only federated candidates satisfying k-anonymity threshold."""
    filtered: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        contributor_count = int(item.get("count_contributing_orgs") or 0)
        if contributor_count >= max(1, min_contributors):
            filtered.append(item)
    return filtered


def rank_federated_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort federated candidates by quality and contributor depth."""
    return sorted(
        [c for c in candidates if isinstance(c, dict)],
        key=lambda c: (
            float(c.get("quality_score") or 0.0),
            int(c.get("count_contributing_orgs") or 0),
        ),
        reverse=True,
    )


def summarize_federated_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Build compact summary over filtered federated candidates."""
    if not candidates:
        return {
            "candidate_count": 0,
            "top_domains": [],
            "avg_quality": 0.0,
            "avg_success_rate": 0.0,
            "top_patterns": [],
        }

    domains: dict[str, int] = {}
    quality_scores: list[float] = []
    success_scores: list[float] = []

    for c in candidates:
        domain = str(c.get("domain") or "general").strip().lower()
        domains[domain] = domains.get(domain, 0) + 1
        quality_scores.append(_clip01(float(c.get("quality_score") or 0.0)))
        if "success_rate" in c:
            success_scores.append(_clip01(float(c.get("success_rate") or 0.0)))

    ranked = rank_federated_candidates(candidates)
    top_patterns: list[str] = []
    for c in ranked[:3]:
        content = c.get("content")
        if isinstance(content, dict):
            description = str(content.get("description") or content.get("title") or "").strip()
            if description:
                top_patterns.append(description)
                continue
        top_patterns.append(f"{c.get('summary_type', 'pattern')}:{c.get('domain', 'general')}")

    top_domains = [k for k, _ in sorted(domains.items(), key=lambda item: item[1], reverse=True)[:3]]

    return {
        "candidate_count": len(candidates),
        "top_domains": top_domains,
        "avg_quality": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.0,
        "avg_success_rate": round(sum(success_scores) / len(success_scores), 4) if success_scores else 0.0,
        "top_patterns": top_patterns,
    }


def percentile_rank(value: float, peers: list[float]) -> float:
    """Compute percentile rank of value against peer sample."""
    if not peers:
        return 0.0
    valid = [float(v) for v in peers]
    less_or_equal = sum(1 for v in valid if v <= float(value))
    return round(less_or_equal / max(1, len(valid)), 4)


def build_benchmark_insight(
    *,
    org_metrics: dict[str, float],
    peer_metrics: dict[str, list[float]],
    benchmark_metric: str,
    target_percentile: float,
    dp_service: DifferentialPrivacyService,
) -> dict[str, Any]:
    """Build privacy-aware benchmark insight for one metric."""
    metric = str(benchmark_metric or "").strip() or "memory_quality"
    org_value = float(org_metrics.get(metric) or 0.0)
    peers = list(peer_metrics.get(metric) or [])

    if not peers:
        return {
            "metric": metric,
            "org_value": org_value,
            "peer_percentile": 0.0,
            "target_percentile": _clip01(target_percentile),
            "gap_to_target": 0.0,
            "peer_count": 0,
            "private_peer_mean": 0.0,
            "status": "insufficient_peer_data",
        }

    peer_percentile = percentile_rank(org_value, peers)
    private_peer_mean = float(
        dp_service.aggregate_with_privacy(
            org_data_dict={f"peer_{idx}": [float(v)] for idx, v in enumerate(peers)},
            aggregation_type="mean",
            epsilon=dp_service.epsilon,
        )
    )

    return {
        "metric": metric,
        "org_value": org_value,
        "peer_percentile": peer_percentile,
        "target_percentile": _clip01(target_percentile),
        "gap_to_target": round(_clip01(target_percentile) - peer_percentile, 4),
        "peer_count": len(peers),
        "private_peer_mean": round(private_peer_mean, 4),
        "status": "on_track" if peer_percentile >= _clip01(target_percentile) else "below_target",
    }


def build_sharing_recommendation(
    *,
    privacy_risk_score: float,
    candidate_count: int,
    avg_quality: float,
) -> str:
    """Recommend sharing scope based on privacy risk and federated quality."""
    risk = _clip01(privacy_risk_score)
    quality = _clip01(avg_quality)

    if candidate_count == 0:
        return "hold: insufficient federated evidence"
    if risk >= 0.6:
        return "restrict: peer_group_only"
    if quality >= 0.7 and risk <= 0.3:
        return "share: public_best_practice"
    return "share: controlled_peer_exchange"


def compute_federated_confidence(
    *,
    candidate_count: int,
    avg_quality: float,
    peer_count: int,
    privacy_risk_score: float,
) -> float:
    """Confidence that federated output is reliable and safely shareable."""
    score = 0.35
    if candidate_count >= 3:
        score += 0.20
    if candidate_count >= 8:
        score += 0.10
    score += 0.20 * _clip01(avg_quality)
    if peer_count >= 5:
        score += 0.10
    if peer_count >= 15:
        score += 0.05
    score += 0.10 * (1.0 - _clip01(privacy_risk_score))
    return round(_clip01(score), 4)


class FederatedMemoryAgent(BaseAgent):
    """Phase 44: Federated memory synthesis with privacy-aware confidence."""

    name = "FederatedMemoryAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["PlaybookAgent", "CredibilityAgent", "AdaptiveEnrichmentBudgetAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("federated_summary"), dict):
            raise ValueError("federated_summary must be a dict")
        if not isinstance(outputs.get("benchmark_insight"), dict):
            raise ValueError("benchmark_insight must be a dict")
        if not isinstance(outputs.get("sharing_recommendation"), str):
            raise ValueError("sharing_recommendation must be a str")

        privacy = outputs.get("privacy_risk_score")
        if not isinstance(privacy, (int, float)) or not (0.0 <= float(privacy) <= 1.0):
            raise ValueError("privacy_risk_score must be a float in [0, 1]")

        federated_conf = outputs.get("federated_confidence")
        if not isinstance(federated_conf, (int, float)) or not (0.0 <= float(federated_conf) <= 1.0):
            raise ValueError("federated_confidence must be a float in [0, 1]")

        conf = outputs.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

        rationale = outputs.get("rationale")
        if rationale not in {"heuristic", "llm"}:
            raise ValueError("rationale must be heuristic|llm")

    def _heuristic(
        self,
        *,
        candidates: list[dict[str, Any]],
        org_metrics: dict[str, float],
        peer_metrics: dict[str, list[float]],
        benchmark_metric: str,
        target_percentile: float,
        min_contributors: int,
        privacy_budget_epsilon: float,
    ) -> dict[str, Any]:
        dp_service = DifferentialPrivacyService(epsilon=max(0.01, float(privacy_budget_epsilon)))

        filtered = filter_k_anonymous_candidates(
            candidates,
            min_contributors=max(1, int(min_contributors)),
        )
        federated_summary = summarize_federated_candidates(filtered)

        # Privacy risk uses effective contributing-org count and data dimensionality.
        org_count_estimate = 0
        if filtered:
            org_count_estimate = min(
                int(c.get("count_contributing_orgs") or 0) for c in filtered if isinstance(c, dict)
            )
        org_count_estimate = max(org_count_estimate, 1)

        data_dimension = max(1, len(org_metrics) + len(peer_metrics))
        privacy_risk_score = round(
            dp_service.estimate_reidentification_risk(
                org_count=org_count_estimate,
                data_dimension=data_dimension,
                epsilon=dp_service.epsilon,
            ),
            4,
        )

        benchmark_insight = build_benchmark_insight(
            org_metrics=org_metrics,
            peer_metrics=peer_metrics,
            benchmark_metric=benchmark_metric,
            target_percentile=target_percentile,
            dp_service=dp_service,
        )

        sharing_recommendation = build_sharing_recommendation(
            privacy_risk_score=privacy_risk_score,
            candidate_count=int(federated_summary.get("candidate_count") or 0),
            avg_quality=float(federated_summary.get("avg_quality") or 0.0),
        )

        federated_confidence = compute_federated_confidence(
            candidate_count=int(federated_summary.get("candidate_count") or 0),
            avg_quality=float(federated_summary.get("avg_quality") or 0.0),
            peer_count=int(benchmark_insight.get("peer_count") or 0),
            privacy_risk_score=privacy_risk_score,
        )

        return {
            "federated_summary": federated_summary,
            "benchmark_insight": benchmark_insight,
            "sharing_recommendation": sharing_recommendation,
            "privacy_risk_score": privacy_risk_score,
            "federated_confidence": federated_confidence,
            "confidence": federated_confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        candidates = list(enrichment.get("federated_candidates") or [])
        org_metrics = dict(enrichment.get("org_metrics") or {})
        peer_metrics = dict(enrichment.get("peer_metrics") or {})
        benchmark_metric = str(enrichment.get("benchmark_metric") or "memory_quality")
        target_percentile = float(enrichment.get("target_percentile") or 0.75)
        min_contributors = int(enrichment.get("min_contributors") or 5)
        privacy_budget_epsilon = float(enrichment.get("privacy_budget_epsilon") or 1.0)

        strategy = getattr(settings, "FEDERATED_MEMORY_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]
        if strategy == "heuristic":
            outputs = self._heuristic(
                candidates=candidates,
                org_metrics=org_metrics,
                peer_metrics=peer_metrics,
                benchmark_metric=benchmark_metric,
                target_percentile=target_percentile,
                min_contributors=min_contributors,
                privacy_budget_epsilon=privacy_budget_epsilon,
            )
        else:
            heuristic = self._heuristic(
                candidates=candidates,
                org_metrics=org_metrics,
                peer_metrics=peer_metrics,
                benchmark_metric=benchmark_metric,
                target_percentile=target_percentile,
                min_contributors=min_contributors,
                privacy_budget_epsilon=privacy_budget_epsilon,
            )
            prompt = (
                "You are a federated memory intelligence agent. Output JSON only.\n"
                "Given privacy-safe aggregates, produce concise federated guidance with fields:\n"
                "federated_summary (object), benchmark_insight (object), sharing_recommendation (string),\n"
                "privacy_risk_score (0..1), federated_confidence (0..1), confidence (0..1), rationale.\n"
                "Do not invent tenant-specific data.\n\n"
                f"HEURISTIC_BASELINE: {heuristic}\n"
            )
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_llm_model("agents")),
                timeout_seconds=float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2)),
            )
            response = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(response, dict)
                and isinstance(response.get("federated_summary"), dict)
                and isinstance(response.get("benchmark_insight"), dict)
                and isinstance(response.get("sharing_recommendation"), str)
            ):
                outputs = {
                    **response,
                    "privacy_risk_score": _clip01(float(response.get("privacy_risk_score") or 0.0)),
                    "federated_confidence": _clip01(float(response.get("federated_confidence") or 0.0)),
                    "confidence": _clip01(float(response.get("confidence") or 0.0)),
                    "rationale": "llm",
                }
            else:
                outputs = heuristic

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.5)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
