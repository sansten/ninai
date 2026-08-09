"""Phase 95 — ColdStartBootstrapService tests."""
from __future__ import annotations

import pytest

from app.services.cold_start_bootstrap_service import (
    BootstrapResult,
    ColdStartBootstrapService,
    DomainScore,
    _DOMAIN_MEMORY_SEEDS,
)


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

class TestDomainDetection:
    def test_sales_keywords_detected(self):
        svc = ColdStartBootstrapService()
        profile = {"industry": "SaaS", "team_role": "Sales", "keywords": ["pipeline", "lead", "quota"]}
        score = svc.detect_domain(profile)
        assert score.domain == "sales"

    def test_engineering_keywords_detected(self):
        svc = ColdStartBootstrapService()
        profile = {"role": "DevOps", "description": "deploy incidents sre kubernetes docker"}
        score = svc.detect_domain(profile)
        assert score.domain == "engineering"

    def test_support_keywords_detected(self):
        svc = ColdStartBootstrapService()
        profile = {"team": "Customer Support", "tags": ["ticket", "escalation", "sla", "helpdesk"]}
        score = svc.detect_domain(profile)
        assert score.domain == "support"

    def test_research_keywords_detected(self):
        svc = ColdStartBootstrapService()
        profile = {"description": "hypothesis experiment literature citation paper benchmark"}
        score = svc.detect_domain(profile)
        assert score.domain == "research"

    def test_operations_keywords_detected(self):
        svc = ColdStartBootstrapService()
        profile = {"description": "procurement compliance vendor audit budget sop"}
        score = svc.detect_domain(profile)
        assert score.domain == "operations"

    def test_confidence_between_zero_and_one(self):
        svc = ColdStartBootstrapService()
        score = svc.detect_domain({"team": "sales lead pipeline"})
        assert 0.0 <= score.score <= 1.0

    def test_matched_keywords_subset_of_input(self):
        svc = ColdStartBootstrapService()
        profile = {"text": "deploy incident sre runbook"}
        score = svc.detect_domain(profile)
        assert all(isinstance(k, str) for k in score.matched_keywords)

    def test_empty_profile_returns_some_domain(self):
        svc = ColdStartBootstrapService()
        score = svc.detect_domain({})
        assert score.domain in svc.available_domains()

    def test_domain_score_is_domain_score_type(self):
        svc = ColdStartBootstrapService()
        result = svc.detect_domain({"k": "v"})
        assert isinstance(result, DomainScore)

    def test_mixed_keywords_picks_dominant_domain(self):
        svc = ColdStartBootstrapService()
        # Heavily weighted towards engineering
        profile = {"text": "deploy incident kubernetes sre docker ci cd runbook postmortem"}
        score = svc.detect_domain(profile)
        assert score.domain == "engineering"


# ---------------------------------------------------------------------------
# Bootstrap — result structure
# ---------------------------------------------------------------------------

class TestBootstrapResultStructure:
    def test_returns_bootstrap_result(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("org-123", {"team": "sales pipeline lead"})
        assert isinstance(result, BootstrapResult)

    def test_org_id_preserved(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("org-abc", {"team": "support"})
        assert result.org_id == "org-abc"

    def test_domain_is_string(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "sales"})
        assert isinstance(result.domain, str)

    def test_seeds_created_equals_len_memory_seeds(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "sales"})
        assert result.seeds_created == len(result.memory_seeds)

    def test_seeds_are_dicts(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "engineering deploy"})
        assert all(isinstance(s, dict) for s in result.memory_seeds)

    def test_seeds_have_org_id(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("org-xyz", {"team": "support"})
        for seed in result.memory_seeds:
            assert seed["org_id"] == "org-xyz"

    def test_seeds_tagged_with_bootstrap(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "research"})
        for seed in result.memory_seeds:
            assert "bootstrap" in seed["tags"]

    def test_template_applied_field_set(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "sales"})
        assert result.template_applied
        assert "sales" in result.template_applied

    def test_domain_confidence_is_float(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "operations vendor compliance"})
        assert isinstance(result.domain_confidence, float)

    def test_matched_keywords_list(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "lead pipeline sales quota"})
        assert isinstance(result.matched_keywords, list)


# ---------------------------------------------------------------------------
# Bootstrap — seeds count and content
# ---------------------------------------------------------------------------

class TestBootstrapSeedContent:
    def test_sales_seeds_have_titles(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "sales lead pipeline quota"})
        for seed in result.memory_seeds:
            assert "title" in seed

    def test_seeds_have_content_field(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "engineering deploy runbook"})
        for seed in result.memory_seeds:
            assert "content" in seed
            assert len(seed["content"]) > 10

    def test_max_seeds_respected(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "sales"}, max_seeds=2)
        assert result.seeds_created <= 2

    def test_custom_seeds_appended(self):
        svc = ColdStartBootstrapService()
        custom = [{"title": "Custom Policy", "content": "Custom rule", "tags": ["custom"]}]
        result = svc.bootstrap("o1", {"team": "support ticket"}, custom_seeds=custom)
        titles = [s["title"] for s in result.memory_seeds]
        assert "Custom Policy" in titles

    def test_bootstrap_domain_tag_on_seeds(self):
        svc = ColdStartBootstrapService()
        result = svc.bootstrap("o1", {"team": "research hypothesis"})
        for seed in result.memory_seeds:
            assert "bootstrap_domain" in seed
            assert seed["bootstrap_domain"] == result.domain

    def test_no_seeds_lost_below_max(self):
        svc = ColdStartBootstrapService()
        domain_seed_count = len(_DOMAIN_MEMORY_SEEDS["engineering"])
        result = svc.bootstrap("o1", {"team": "engineering deploy incident"}, max_seeds=100)
        assert result.seeds_created == domain_seed_count


# ---------------------------------------------------------------------------
# Confidence fallback
# ---------------------------------------------------------------------------

class TestConfidenceFallback:
    def test_low_confidence_profile_gets_fallback_domain(self):
        svc = ColdStartBootstrapService(min_confidence=0.99)
        result = svc.bootstrap("o1", {"name": "acme corp"})
        # No real keywords → confidence 0 → fallback
        assert len(result.warnings) > 0
        assert "fell back" in result.warnings[0].lower() or "below" in result.warnings[0].lower()

    def test_fallback_domain_still_produces_seeds(self):
        svc = ColdStartBootstrapService(min_confidence=0.99)
        result = svc.bootstrap("o1", {"name": "mystery company"})
        assert result.seeds_created > 0

    def test_high_confidence_threshold_triggers_fallback(self):
        svc = ColdStartBootstrapService(min_confidence=0.95)
        result = svc.bootstrap("o1", {"team": "sales"})
        # Even "sales" keywords might not hit 0.95 confidence
        assert result.seeds_created > 0


# ---------------------------------------------------------------------------
# Domain management
# ---------------------------------------------------------------------------

class TestDomainManagement:
    def test_available_domains_includes_built_in(self):
        svc = ColdStartBootstrapService()
        domains = svc.available_domains()
        assert "sales" in domains
        assert "engineering" in domains
        assert "support" in domains
        assert "research" in domains
        assert "operations" in domains

    def test_available_domains_sorted(self):
        svc = ColdStartBootstrapService()
        domains = svc.available_domains()
        assert domains == sorted(domains)

    def test_add_custom_domain(self):
        svc = ColdStartBootstrapService()
        svc.add_domain_seeds("legal", [{"title": "Litigation Hold", "content": "...", "tags": []}])
        assert "legal" in svc.available_domains()

    def test_custom_seeds_override_template(self):
        custom = {"sales": [{"title": "Custom Sales Seed", "content": "override", "tags": []}]}
        svc = ColdStartBootstrapService(custom_seeds=custom)
        result = svc.bootstrap("o1", {"team": "sales lead pipeline"})
        assert any("Custom Sales Seed" in s.get("title", "") for s in result.memory_seeds)

    def test_add_domain_replaces_existing(self):
        svc = ColdStartBootstrapService()
        svc.add_domain_seeds("sales", [{"title": "Only Seed", "content": "x", "tags": []}])
        result = svc.bootstrap("o1", {"team": "sales pipeline lead quota"})
        titles = [s["title"] for s in result.memory_seeds]
        assert "Only Seed" in titles
        # Original sales seeds should be replaced
        assert "Lead Qualification Framework" not in titles

    def test_profile_to_text_flattens_nested(self):
        svc = ColdStartBootstrapService()
        profile = {"keywords": ["sales", "lead"], "meta": {"role": "Account Executive"}}
        text = svc._profile_to_text(profile)
        assert "sales" in text
        assert "Account Executive" in text
