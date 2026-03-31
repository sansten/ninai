# Phase 54: Proof Layer Productization Playbook

This playbook standardizes how teams move from pilot to production using the Phase 54 proof layer.

## 1. Scope

This document covers:
- Packaging and capability tiering for proof workflows
- Reference integrations for Python, Node, and Java/Spring
- Rollout checklist with reproducibility and compatibility gates

Core proof endpoints:
- `POST /api/v1/proof/scorecard`
- `POST /api/v1/proof/monthly-impact`

## 2. Capability Tiering

| Tier | Intended Stage | Required Capabilities | Exit Criteria |
|---|---|---|---|
| Tier P1: Pilot Baseline | Initial value discovery | Proof endpoints enabled, baseline metrics defined, reproducibility hash captured | First 30-day report generated with deterministic replay |
| Tier P2: Pilot Expansion | Multi-team validation | 3+ tenant cohorts, monthly impact tracking, API compatibility checks in CI | Lead-time gain >= 25% trend and positive SLA avoidance trend |
| Tier P3: Production Proof | Operationalized value reporting | Automated monthly reports, integration runbooks, rollback and audit controls | Pilot-to-production conversion >= 60% |

## 3. Reference Integrations

### Python SDK (recommended)

The Python SDK now exposes a typed proof resource:

```python
from ninai import NinaiClient

client = NinaiClient(api_key="nai_your_key", base_url="https://api.example.com/api/v1")

scorecard = client.proof.scorecard(
    records=[
        {
            "incident_id": "inc-1",
            "lead_time_hours": 7.0,
            "mttr_hours": 5.0,
            "avoided_sla_breach": True,
            "false_escalation": False,
        }
    ],
    baseline={
        "lead_time_hours": 10.0,
        "mttr_hours": 8.0,
        "false_escalation_rate": 0.25,
    },
)

print(scorecard.score, scorecard.reproducibility_hash)
```

### Node (reference HTTP client)

```javascript
const res = await fetch("https://api.example.com/api/v1/proof/monthly-impact", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${process.env.NINAI_TOKEN}`,
  },
  body: JSON.stringify({
    month: "2026-03",
    records: [
      {
        incident_id: "inc-1",
        lead_time_hours: 7.0,
        mttr_hours: 5.0,
        avoided_sla_breach: true,
        false_escalation: false,
      },
    ],
    baseline: {
      lead_time_hours: 10.0,
      mttr_hours: 8.0,
      false_escalation_rate: 0.25,
      sla_penalty_per_breach: 1500.0,
    },
    monthly_operating_cost: 3000.0,
  }),
});

const report = await res.json();
console.log(report.net_impact, report.roi_pct, report.reproducibility_hash);
```

### Java/Spring (reference WebClient)

```java
WebClient client = WebClient.builder()
    .baseUrl("https://api.example.com/api/v1")
    .defaultHeader("Authorization", "Bearer " + token)
    .build();

Map<String, Object> payload = Map.of(
    "records", List.of(Map.of(
        "incident_id", "inc-1",
        "lead_time_hours", 7.0,
        "mttr_hours", 5.0,
        "avoided_sla_breach", true,
        "false_escalation", false
    )),
    "baseline", Map.of(
        "lead_time_hours", 10.0,
        "mttr_hours", 8.0,
        "false_escalation_rate", 0.25
    )
);

Map<?, ?> scorecard = client.post()
    .uri("/proof/scorecard")
    .bodyValue(payload)
    .retrieve()
    .bodyToMono(Map.class)
    .block();

System.out.println(scorecard.get("score"));
```

## 4. Rollout Checklist

### DX onboarding gate (<4 hours)
- API key provisioning documented
- One working proof endpoint call per integration stack
- Deterministic hash observed across replay of same payload

### Pilot evidence gate (3 tenants)
- Each tenant has pre/post baseline records
- Monthly impact report generated for at least 30 days
- Evidence package includes scorecard + monthly-impact JSON + reproducibility hash

### API compatibility gate
- Backward-compatible response contract validated in CI
- No breaking field removals for proof responses
- Versioned schema change notes for any additive fields

### Production readiness gate
- Monthly report runbook owned by on-call team
- Alerting on proof endpoint failures and latency
- Rollback plan tested for integration clients

## 5. Operating Rhythm

Monthly proof review cadence:
1. Generate and archive monthly-impact reports for active pilots.
2. Compare reproducibility hashes for sampled replays.
3. Review trend deltas (lead time, MTTR, SLA avoidance, false escalations).
4. Approve promotion from P2 to P3 only when all gates pass for two consecutive cycles.

## 6. Notes

- Proof metrics are advisory for rollout governance and should be paired with tenant-level qualitative evidence.
- Keep raw records used for score generation immutable once a monthly report is finalized.
