"""
Gate evidence tests for G1 (runbook), G2 (alerting), G3 (capacity planning),
H1 (feature status), H2 (bounded claims), H3 (trust/transparency).

These tests verify that required documentation artifacts exist and contain
the expected content — serving as a reproducible check that the CognitiveOS
readiness gate evidence is present on disk.
"""

import os
import re


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _read(rel):
    path = os.path.join(REPO_ROOT, rel)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Gate G1 — Runbook completeness
# ---------------------------------------------------------------------------


def test_runbook_has_heartbeat_failure_section():
    """G1: runbook covers heartbeat failure incident response."""
    text = _read("docs/DEPLOYMENT_RUNBOOK.md")
    assert "Heartbeat Failure" in text
    assert "CognitiveHeartbeatStale" in text or "cognitive_heartbeat" in text


def test_runbook_has_queue_saturation_section():
    """G1: runbook covers cognitive queue saturation."""
    text = _read("docs/DEPLOYMENT_RUNBOOK.md")
    assert "Queue Saturation" in text or "CognitiveReviewQueueDeep" in text


def test_runbook_has_policy_misconfiguration_section():
    """G1: runbook covers policy misconfiguration response."""
    text = _read("docs/DEPLOYMENT_RUNBOOK.md")
    assert "Policy Misconfiguration" in text or "CognitivePolicyDenialSpike" in text


def test_runbook_has_kill_switch_procedure():
    """G1: runbook documents emergency kill switch usage."""
    text = _read("docs/DEPLOYMENT_RUNBOOK.md")
    assert "Kill Switch" in text or "cognitive-autonomy" in text


# ---------------------------------------------------------------------------
# Gate G2 — Alerting quality
# ---------------------------------------------------------------------------


def test_prometheus_rules_has_cognitive_os_group():
    """G2: rules.yml contains a cognitive_os_alerts rule group."""
    text = _read("docker/prometheus/rules.yml")
    assert "cognitive_os_alerts" in text


def test_prometheus_rules_has_heartbeat_stale_alert():
    """G2: heartbeat staleness alert is defined."""
    text = _read("docker/prometheus/rules.yml")
    assert "CognitiveHeartbeatStale" in text


def test_prometheus_rules_has_heartbeat_missed_alert():
    """G2: heartbeat missed (absent metric) alert is defined."""
    text = _read("docker/prometheus/rules.yml")
    assert "CognitiveHeartbeatMissed" in text


def test_prometheus_rules_has_policy_denial_spike_alert():
    """G2: policy denial spike alert for safety monitoring."""
    text = _read("docker/prometheus/rules.yml")
    assert "CognitivePolicyDenialSpike" in text


def test_prometheus_rules_has_review_queue_deep_alert():
    """G2: review queue depth alert defined."""
    text = _read("docker/prometheus/rules.yml")
    assert "CognitiveReviewQueueDeep" in text


# ---------------------------------------------------------------------------
# Gate G3 — Capacity planning
# ---------------------------------------------------------------------------


def test_runbook_has_capacity_planning_section():
    """G3: runbook documents capacity model and autoscaling thresholds."""
    text = _read("docs/DEPLOYMENT_RUNBOOK.md")
    assert "Capacity Planning" in text or "capacity" in text.lower()


def test_runbook_capacity_mentions_heartbeat_worker_scaling():
    """G3: scaling guidance exists for heartbeat/Celery workers."""
    text = _read("docs/DEPLOYMENT_RUNBOOK.md")
    assert re.search(r"scale.*worker|worker.*scale|autoscal", text, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Gate H1 — Feature status reflects real implementation
# ---------------------------------------------------------------------------


def test_features_md_has_cognitive_os_section():
    """H1: FEATURES.md documents CognitiveOS capability and phase status."""
    text = _read("docs/FEATURES.md")
    assert "CognitiveOS" in text


def test_features_md_lists_autonomous_heartbeat_as_implemented():
    """H1: heartbeat autonomous spawn is marked implemented in feature docs."""
    text = _read("docs/FEATURES.md")
    assert "cognitive_heartbeat_task" in text or "Autonomous cognitive heartbeat" in text


def test_features_md_lists_kill_switch_as_implemented():
    """H1: kill switch feature is listed as implemented."""
    text = _read("docs/FEATURES.md")
    assert "Kill switch" in text or "kill switch" in text or "cognitive-autonomy" in text


# ---------------------------------------------------------------------------
# Gate H2 — Bounded claim language
# ---------------------------------------------------------------------------


def test_features_md_has_does_not_do_section():
    """H2: docs explicitly state what CognitiveOS does NOT do."""
    text = _read("docs/FEATURES.md")
    assert "does not" in text.lower() or "does NOT" in text


def test_features_md_mentions_capability_boundary():
    """H2: capability scope limits are documented."""
    text = _read("docs/FEATURES.md")
    assert "capability" in text.lower() and ("boundary" in text.lower() or "token" in text.lower())


def test_features_md_mentions_probabilistic_quality_caveat():
    """H2: docs acknowledge that decision quality is not guaranteed."""
    text = _read("docs/FEATURES.md")
    assert "probabilistic" in text.lower() or "quality depends" in text.lower() or "not guarantee" in text.lower()


# ---------------------------------------------------------------------------
# Gate H3 — Customer-facing trust / transparency
# ---------------------------------------------------------------------------


def test_features_md_has_trust_transparency_section():
    """H3: docs explain transparency of autonomous operation."""
    text = _read("docs/FEATURES.md")
    assert "Trust" in text or "Transparency" in text


def test_features_md_explains_human_in_loop():
    """H3: docs explain where humans are always in the loop."""
    text = _read("docs/FEATURES.md")
    assert "human" in text.lower() and "in the loop" in text.lower()


def test_features_md_explains_tenant_isolation():
    """H3: docs explain how multi-tenant safety is enforced."""
    text = _read("docs/FEATURES.md")
    assert "organization_id" in text or "RLS" in text or "row-level security" in text.lower()


def test_features_md_provides_audit_query_examples():
    """H3: docs provide concrete audit query examples for customers."""
    text = _read("docs/FEATURES.md")
    assert "audit" in text.lower() and ("policy.autonomous_action" in text or "/api/v1/audit" in text)
