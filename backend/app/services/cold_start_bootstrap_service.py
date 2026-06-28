"""Cold Start Bootstrap Service — Phase 95.

Solves the blank-slate problem: a new tenant has no memories, no playbooks,
and no interaction history. Without seeding, the first few weeks of Ninai usage
produce generic, unhelpful responses because there is nothing to retrieve.

Approach:
  1. Detect the tenant's domain from metadata (industry, role keywords, etc.).
  2. Select a matching domain template (or the closest if none matches exactly).
  3. Seed a small set of starter memories and playbooks for that domain.
  4. Record which template was applied so it can be refined later.

Domain detection is heuristic — it scores keyword overlap between the tenant
profile and domain keyword lists. For production, this can be wired to a
classifier, but keyword matching is sufficient for the 5 enterprise domains
and produces deterministic, testable results.

Template contents per domain (configurable — override via custom_seeds):
  - sales:       lead handling, CRM terminology, qualification frameworks
  - engineering: incident runbooks, deployment checklists, code review norms
  - support:     escalation paths, response templates, SLA definitions
  - research:    literature tracking, hypothesis templates, citation norms
  - operations:  process standards, compliance checklist, vendor SLAs

Usage::

    svc = ColdStartBootstrapService()

    result = svc.bootstrap(
        org_id="org-123",
        tenant_profile={
            "industry": "SaaS",
            "team_role": "Customer Success",
            "keywords": ["renewal", "churn", "onboarding", "upsell"],
        },
    )
    # result.domain → "sales"
    # result.seeds_created → 12
    # result.memory_seeds → list of dicts ready for memory ingest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain keyword vocabulary
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "sales": [
        "sales", "revenue", "pipeline", "lead", "prospect", "deal", "quota",
        "crm", "account", "renewal", "churn", "upsell", "cross-sell", "close",
        "forecast", "opportunity", "customer success", "onboarding",
    ],
    "engineering": [
        "engineering", "developer", "code", "deploy", "incident", "bug",
        "release", "sprint", "backlog", "pull request", "review", "ci", "cd",
        "devops", "sre", "infrastructure", "kubernetes", "docker", "monitoring",
        "slo", "sla", "postmortem", "runbook",
    ],
    "support": [
        "support", "ticket", "helpdesk", "customer", "escalation", "resolution",
        "sla", "response time", "csat", "nps", "agent", "triage", "issue",
        "complaint", "refund", "troubleshoot",
    ],
    "research": [
        "research", "analysis", "hypothesis", "experiment", "paper", "citation",
        "literature", "study", "finding", "dataset", "model", "benchmark",
        "evaluation", "peer review", "publication", "academic",
    ],
    "operations": [
        "operations", "ops", "process", "compliance", "vendor", "procurement",
        "logistics", "supply chain", "workflow", "audit", "approval", "policy",
        "sop", "standard", "checklist", "budget", "finance",
    ],
}

# ---------------------------------------------------------------------------
# Starter memory templates
# ---------------------------------------------------------------------------

_DOMAIN_MEMORY_SEEDS: dict[str, list[dict[str, Any]]] = {
    "sales": [
        {
            "title": "Lead Qualification Framework",
            "content": "Qualify leads using BANT: Budget, Authority, Need, Timeline. "
                       "Disqualify if no clear budget or decision timeline beyond 6 months.",
            "tags": ["lead", "qualification", "bant"],
            "memory_type": "playbook",
        },
        {
            "title": "Discovery Call Checklist",
            "content": "1. Confirm pain point. 2. Identify stakeholders. 3. Map buying process. "
                       "4. Set clear next step with a specific date before ending the call.",
            "tags": ["discovery", "call", "sales"],
            "memory_type": "playbook",
        },
        {
            "title": "Churn Signal Response",
            "content": "If a customer misses 3 consecutive check-ins or NPS drops below 6, "
                       "flag for immediate CSM outreach within 48h and create a save plan.",
            "tags": ["churn", "retention", "customer-success"],
            "memory_type": "policy",
        },
        {
            "title": "CRM Data Hygiene Standard",
            "content": "Update deal stage, close date, and next action within 24h of each customer "
                       "interaction. Stale records (>7 days untouched) trigger a hygiene alert.",
            "tags": ["crm", "data-hygiene", "sales"],
            "memory_type": "policy",
        },
        {
            "title": "Renewal Kickoff Timing",
            "content": "Begin renewal conversations 90 days before contract end. Send auto-reminder "
                       "at 90, 60, and 30 days. Escalate to AE if no response after 60-day touch.",
            "tags": ["renewal", "contract", "timing"],
            "memory_type": "fact",
        },
    ],
    "engineering": [
        {
            "title": "Incident Severity Definitions",
            "content": "SEV1: complete outage, all hands. SEV2: degraded service >10% users affected. "
                       "SEV3: isolated bug, workaround exists. SEV4: cosmetic or low-impact.",
            "tags": ["incident", "severity", "sre"],
            "memory_type": "fact",
        },
        {
            "title": "Deployment Checklist",
            "content": "Before deploying to production: run full test suite, review diff with peer, "
                       "confirm feature flags off for partial rollouts, alert on-call, schedule rollback window.",
            "tags": ["deploy", "checklist", "production"],
            "memory_type": "playbook",
        },
        {
            "title": "Postmortem Template",
            "content": "Sections: timeline, impact, root cause, contributing factors, "
                       "action items (owner + due date). Blameless tone. Publish within 72h of resolution.",
            "tags": ["postmortem", "incident", "retrospective"],
            "memory_type": "playbook",
        },
        {
            "title": "On-Call Escalation Path",
            "content": "L1: primary on-call (15 min). L2: secondary + service owner (30 min). "
                       "L3: engineering manager (45 min). Notify stakeholders at SEV1/SEV2 within 10 min.",
            "tags": ["on-call", "escalation", "incident"],
            "memory_type": "policy",
        },
        {
            "title": "Code Review SLA",
            "content": "All PRs must receive at least one approval within 24h on business days. "
                       "Hotfixes: 2h. Block merge on open security findings or failing CI.",
            "tags": ["code-review", "pr", "sla"],
            "memory_type": "policy",
        },
    ],
    "support": [
        {
            "title": "Ticket Triage Priority Matrix",
            "content": "P1: production down, respond in 1h. P2: major feature broken, 4h. "
                       "P3: minor issue with workaround, 8h. P4: question/cosmetic, 2 business days.",
            "tags": ["triage", "priority", "sla"],
            "memory_type": "fact",
        },
        {
            "title": "Escalation Criteria",
            "content": "Escalate to L2 if: issue unresolved after 2 contact attempts, customer is "
                       "enterprise tier, data loss involved, or customer requests manager directly.",
            "tags": ["escalation", "l2", "support"],
            "memory_type": "policy",
        },
        {
            "title": "First Response Template",
            "content": "Thank the customer, acknowledge the issue, provide expected resolution "
                       "timeline and a ticket reference number. Avoid generic 'we are looking into it'.",
            "tags": ["response", "template", "communication"],
            "memory_type": "playbook",
        },
        {
            "title": "Refund Policy",
            "content": "Pro-rata refunds within 14 days of billing for documented billing errors. "
                       "No refunds for usage-based overages unless a system error is confirmed by engineering.",
            "tags": ["refund", "billing", "policy"],
            "memory_type": "policy",
        },
        {
            "title": "CSAT Follow-Up Workflow",
            "content": "If CSAT < 4/5: manager reviews within 24h, agent follows up personally. "
                       "If CSAT = 1–2: automatic escalation to support lead + case flagged for QA.",
            "tags": ["csat", "quality", "follow-up"],
            "memory_type": "playbook",
        },
    ],
    "research": [
        {
            "title": "Hypothesis Documentation Standard",
            "content": "Each hypothesis must state: proposed mechanism, expected outcome, "
                       "falsification criteria, and required evidence. File in the experiment log before testing.",
            "tags": ["hypothesis", "experiment", "documentation"],
            "memory_type": "policy",
        },
        {
            "title": "Literature Review Checklist",
            "content": "Search at minimum: Google Scholar, Semantic Scholar, arXiv (if applicable). "
                       "Include publication date and venue. Note conflicting findings explicitly.",
            "tags": ["literature", "review", "research"],
            "memory_type": "playbook",
        },
        {
            "title": "Experiment Reproducibility Requirements",
            "content": "Code, data, environment config, and random seeds must be archived before "
                       "submitting results. Peer must be able to reproduce within 4h given the archive.",
            "tags": ["reproducibility", "experiment", "code"],
            "memory_type": "policy",
        },
        {
            "title": "Citation and Attribution Policy",
            "content": "Cite primary sources; avoid citing surveys as the sole support for a claim. "
                       "If a result is from an unpublished preprint, flag it as provisional.",
            "tags": ["citation", "attribution", "integrity"],
            "memory_type": "policy",
        },
        {
            "title": "Dataset Versioning Convention",
            "content": "Tag datasets with YYYYMMDD of creation and a one-line schema fingerprint. "
                       "Never overwrite raw data. Derived datasets live in /processed/, not /raw/.",
            "tags": ["dataset", "versioning", "data"],
            "memory_type": "fact",
        },
    ],
    "operations": [
        {
            "title": "Vendor Onboarding Checklist",
            "content": "Require SOC 2 report or equivalent, data processing agreement, "
                       "insurance certificate, and bank details before first payment. Legal reviews contracts.",
            "tags": ["vendor", "onboarding", "compliance"],
            "memory_type": "playbook",
        },
        {
            "title": "Budget Approval Thresholds",
            "content": "< $5K: team lead. $5K–$25K: department head. $25K–$100K: VP approval. "
                       "> $100K: CFO sign-off required. All require a cost justification memo.",
            "tags": ["budget", "approval", "finance"],
            "memory_type": "policy",
        },
        {
            "title": "Change Management SOP",
            "content": "Proposed changes to core processes require: 2-week notice, written impact "
                       "assessment, and sign-off from affected team leads before implementation.",
            "tags": ["change-management", "process", "sop"],
            "memory_type": "policy",
        },
        {
            "title": "Compliance Audit Schedule",
            "content": "Annual: SOC 2, ISO 27001 re-certification. Quarterly: access rights review, "
                       "vendor SLA review. Monthly: security patch audit. Weekly: anomaly log review.",
            "tags": ["compliance", "audit", "schedule"],
            "memory_type": "fact",
        },
        {
            "title": "Incident to Operations Escalation",
            "content": "Operations is notified when engineering incidents affect: vendor SLAs, "
                       "payroll systems, financial reporting pipelines, or any PII data store.",
            "tags": ["incident", "escalation", "operations"],
            "memory_type": "policy",
        },
    ],
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    org_id: str
    domain: str
    domain_confidence: float              # 0.0–1.0
    matched_keywords: list[str]
    memory_seeds: list[dict[str, Any]]    # ready for memory ingest
    seeds_created: int
    template_applied: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class DomainScore:
    domain: str
    score: float
    matched_keywords: list[str]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ColdStartBootstrapService:
    """Domain-aware memory seeder for new tenants.

    Parameters
    ----------
    custom_seeds:
        Optional domain → list[seed_dict] override. Merged with built-in
        templates so only the domains supplied are replaced.
    min_confidence:
        Minimum domain detection confidence to apply a non-default template.
        Below this, falls back to a generic cross-domain seed set.
    """

    _FALLBACK_DOMAIN = "operations"

    def __init__(
        self,
        custom_seeds: dict[str, list[dict[str, Any]]] | None = None,
        min_confidence: float = 0.10,
    ) -> None:
        self._min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self._seeds: dict[str, list[dict[str, Any]]] = dict(_DOMAIN_MEMORY_SEEDS)
        if custom_seeds:
            self._seeds.update(custom_seeds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_domain(self, tenant_profile: dict[str, Any]) -> DomainScore:
        """Score all domains against the tenant profile and return the best match."""
        text_blob = self._profile_to_text(tenant_profile).lower()
        scores: list[DomainScore] = []

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in text_blob]
            raw_score = len(hits) / max(len(keywords), 1)
            scores.append(DomainScore(domain=domain, score=raw_score, matched_keywords=hits))

        scores.sort(key=lambda s: s.score, reverse=True)
        best = scores[0]
        total = sum(s.score for s in scores)
        confidence = (best.score / total) if total > 0 else 0.0
        return DomainScore(
            domain=best.domain,
            score=confidence,
            matched_keywords=best.matched_keywords,
        )

    def bootstrap(
        self,
        org_id: str,
        tenant_profile: dict[str, Any],
        *,
        custom_seeds: list[dict[str, Any]] | None = None,
        max_seeds: int = 10,
    ) -> BootstrapResult:
        """Detect domain and produce a set of starter memory seeds.

        Parameters
        ----------
        org_id:
            Tenant identifier (for provenance tagging).
        tenant_profile:
            Dict containing any combination of: industry, team_role, keywords,
            description, name, tags. More text = better domain detection.
        custom_seeds:
            Extra seeds to include beyond the template (merged, not replaced).
        max_seeds:
            Cap on total seeds returned (template + custom combined).
        """
        domain_score = self.detect_domain(tenant_profile)
        warnings: list[str] = []

        if domain_score.score < self._min_confidence:
            domain = self._FALLBACK_DOMAIN
            warnings.append(
                f"Domain confidence {domain_score.score:.2f} below threshold "
                f"{self._min_confidence:.2f}; fell back to '{domain}'."
            )
        else:
            domain = domain_score.domain

        template_seeds = list(self._seeds.get(domain, []))
        extra = list(custom_seeds or [])
        all_seeds = (template_seeds + extra)[:max_seeds]

        # Tag each seed with provenance
        tagged = []
        for seed in all_seeds:
            s = dict(seed)
            s["org_id"] = org_id
            s["bootstrap_domain"] = domain
            s.setdefault("tags", [])
            if "bootstrap" not in s["tags"]:
                s["tags"] = list(s["tags"]) + ["bootstrap"]
            tagged.append(s)

        template_name = f"{domain}_v1"
        logger.info(
            "ColdStart: org=%s domain=%s confidence=%.2f seeds=%d",
            org_id, domain, domain_score.score, len(tagged),
        )

        return BootstrapResult(
            org_id=org_id,
            domain=domain,
            domain_confidence=domain_score.score,
            matched_keywords=domain_score.matched_keywords,
            memory_seeds=tagged,
            seeds_created=len(tagged),
            template_applied=template_name,
            warnings=warnings,
        )

    def available_domains(self) -> list[str]:
        """Return list of domains with seed templates."""
        return sorted(self._seeds.keys())

    def add_domain_seeds(self, domain: str, seeds: list[dict[str, Any]]) -> None:
        """Register or replace seeds for a domain at runtime."""
        self._seeds[domain.lower()] = list(seeds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_to_text(profile: dict[str, Any]) -> str:
        """Flatten profile dict values into a single searchable string."""
        parts: list[str] = []
        for v in profile.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, (list, tuple)):
                parts.extend(str(item) for item in v)
            elif isinstance(v, dict):
                parts.extend(str(item) for item in v.values())
            else:
                parts.append(str(v))
        return " ".join(parts)
