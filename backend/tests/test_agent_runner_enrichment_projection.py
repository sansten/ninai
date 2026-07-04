"""Tests for AgentRunner's generalized enrichment projection.

_load_prior_enrichment used to query only a 4-agent whitelist
(ClassificationAgent, MetadataExtractionAgent, TopicModelingAgent,
PatternDetectionAgent), so any agent that declared a dependency on a
different upstream agent's output (OrgAttentionAgent reading
propagation_signals from SiloPropagationAgent, ProactiveMemoryPushAgent
reading attention_signals from OrgAttentionAgent, etc.) always saw an empty
default even when the upstream agent had actually run successfully. These
tests cover the pure projection function directly — no database needed.
"""
from __future__ import annotations

from app.services.agent_runner import project_agent_outputs


class TestLegacyNestedProjections:
    def test_classification_nests_under_classification_key(self):
        out: dict = {}
        project_agent_outputs(out, "ClassificationAgent", {"sensitivity": "high"})
        assert out["classification"] == {"sensitivity": "high"}

    def test_metadata_topics_patterns_still_nest_as_before(self):
        out: dict = {}
        project_agent_outputs(out, "MetadataExtractionAgent", {"summary": "x"})
        project_agent_outputs(out, "TopicModelingAgent", {"topics": ["a"]})
        project_agent_outputs(out, "PatternDetectionAgent", {"patterns": ["p"]})
        assert out["metadata"] == {"summary": "x"}
        assert out["topics"] == {"topics": ["a"]}
        assert out["patterns"] == {"patterns": ["p"]}


class TestFlatProjections:
    def test_semantic_normalization_projects_business_domain_and_intent(self):
        out: dict = {}
        project_agent_outputs(
            out, "SemanticNormalizationAgent",
            {"business_domain": "engineering", "intent": "status_update", "other": "ignored"},
        )
        assert out["business_domain"] == "engineering"
        assert out["intent"] == "status_update"
        assert "other" not in out

    def test_context_amplifier_projects_context_bundle(self):
        out: dict = {}
        project_agent_outputs(
            out, "ContextAmplifierAgent", {"context_bundle": [{"entity": "Acme"}]}
        )
        assert out["context_bundle"] == [{"entity": "Acme"}]

    def test_silo_propagation_projects_propagation_signals(self):
        out: dict = {}
        project_agent_outputs(
            out, "SiloPropagationAgent", {"propagation_signals": [{"entity": "Acme", "target_domain": "sales"}]}
        )
        assert out["propagation_signals"] == [{"entity": "Acme", "target_domain": "sales"}]

    def test_org_attention_projects_attention_signals_and_overlap_alerts(self):
        out: dict = {}
        project_agent_outputs(
            out, "OrgAttentionAgent",
            {
                "attention_signals": [{"team": "eng", "topic": "migration"}],
                "overlap_alerts": [{"topic": "migration", "teams": ["eng", "ops"]}],
            },
        )
        assert out["attention_signals"] == [{"team": "eng", "topic": "migration"}]
        assert out["overlap_alerts"] == [{"topic": "migration", "teams": ["eng", "ops"]}]

    def test_missing_output_key_does_not_create_enrichment_key(self):
        out: dict = {}
        project_agent_outputs(out, "SemanticNormalizationAgent", {})
        assert "business_domain" not in out
        assert "intent" not in out


class TestEntityResolutionDerivedProjection:
    def test_projects_resolved_and_unresolved_entities(self):
        out: dict = {}
        project_agent_outputs(
            out, "EntityResolutionAgent",
            {
                "resolved_entities": [{"canonical": "Acme Corp", "entity_type": "org"}],
                "unresolved_entities": ["mystery co"],
            },
        )
        assert out["resolved_entities"] == [{"canonical": "Acme Corp", "entity_type": "org"}]
        assert out["unresolved_entities"] == ["mystery co"]

    def test_derives_flat_entities_list_from_canonical_names(self):
        """OrgAttentionAgent/ProactiveMemoryPushAgent expect entities as a flat
        list[str], but EntityResolutionAgent emits list[dict] with 'canonical'."""
        out: dict = {}
        project_agent_outputs(
            out, "EntityResolutionAgent",
            {
                "resolved_entities": [
                    {"canonical": "Acme Corp", "entity_type": "org"},
                    {"canonical": "Bob Smith", "entity_type": "person"},
                    {"entity_type": "org"},  # no canonical — skipped
                ],
            },
        )
        assert out["entities"] == ["Acme Corp", "Bob Smith"]

    def test_no_entities_key_when_no_canonical_names(self):
        out: dict = {}
        project_agent_outputs(out, "EntityResolutionAgent", {"resolved_entities": []})
        assert "entities" not in out


class TestUnknownAgent:
    def test_unrecognized_agent_is_a_safe_no_op(self):
        out: dict = {}
        project_agent_outputs(out, "SomeFutureAgent", {"whatever": "value"})
        assert out == {}

    def test_none_outputs_does_not_raise(self):
        out: dict = {}
        project_agent_outputs(out, "ClassificationAgent", None)  # type: ignore[arg-type]
        assert out["classification"] == {}


class TestMultipleAgentsAccumulate:
    def test_projections_from_different_agents_coexist(self):
        out: dict = {}
        project_agent_outputs(out, "ClassificationAgent", {"sensitivity": "low"})
        project_agent_outputs(
            out, "EntityResolutionAgent",
            {"resolved_entities": [{"canonical": "Acme", "entity_type": "org"}]},
        )
        project_agent_outputs(out, "ContextAmplifierAgent", {"context_bundle": [{"entity": "Acme"}]})
        assert out["classification"] == {"sensitivity": "low"}
        assert out["entities"] == ["Acme"]
        assert out["context_bundle"] == [{"entity": "Acme"}]
