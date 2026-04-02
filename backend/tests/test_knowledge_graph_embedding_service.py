from __future__ import annotations

from app.services.knowledge_graph_embedding_service import KnowledgeGraphEmbeddingService


def _sample_edges() -> list[dict]:
    return [
        {"source_id": "hub", "target_id": "a", "weight": 0.9},
        {"source_id": "hub", "target_id": "b", "weight": 0.8},
        {"source_id": "hub", "target_id": "c", "weight": 0.7},
        {"source_id": "a", "target_id": "hub", "weight": 0.6},
        {"source_id": "b", "target_id": "hub", "weight": 0.5},
        {"source_id": "c", "target_id": "hub", "weight": 0.4},
        {"source_id": "a", "target_id": "leaf", "weight": 0.3},
    ]


class TestBuildAdjacency:
    def test_build_adjacency_three_edges(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = svc.build_adjacency(
            edges=[
                {"source_id": "n1", "target_id": "n2", "weight": 1.0},
                {"source_id": "n1", "target_id": "n3", "weight": 0.5},
                {"source_id": "n2", "target_id": "n3", "weight": 0.7},
            ]
        )
        assert adjacency["n1"] == [("n2", 1.0), ("n3", 0.5)]
        assert adjacency["n2"] == [("n3", 0.7)]
        assert adjacency["n3"] == []

    def test_empty_edges_returns_empty_adjacency(self):
        svc = KnowledgeGraphEmbeddingService()
        assert svc.build_adjacency(edges=[]) == {}

    def test_invalid_edges_are_skipped(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = svc.build_adjacency(
            edges=[
                {"source_id": "a", "target_id": "b", "weight": 1.0},
                {"source_id": None, "target_id": "c", "weight": 1.0},
                {"source_id": "d", "target_id": None, "weight": 1.0},
            ]
        )
        assert adjacency == {"a": [("b", 1.0)], "b": []}

    def test_default_weight_when_missing(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = svc.build_adjacency(edges=[{"source_id": "a", "target_id": "b"}])
        assert adjacency["a"] == [("b", 1.0)]

    def test_non_numeric_weight_falls_back_default(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = svc.build_adjacency(
            edges=[{"source_id": "a", "target_id": "b", "weight": "bad"}]
        )
        assert adjacency["a"] == [("b", 1.0)]


class TestComputeDegreeFeatures:
    def test_leaf_node_total_degree_one(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = svc.build_adjacency(edges=[{"source_id": "a", "target_id": "leaf", "weight": 1.0}])
        features = svc.compute_degree_features(adjacency=adjacency)
        assert features["a"]["total_degree"] == 1
        assert features["a"]["is_leaf"] is True

    def test_hub_node_flagged_true(self):
        svc = KnowledgeGraphEmbeddingService()
        features = svc.compute_degree_features(adjacency=svc.build_adjacency(edges=_sample_edges()))
        assert features["hub"]["is_hub"] is True

    def test_feature_contains_expected_keys(self):
        svc = KnowledgeGraphEmbeddingService()
        features = svc.compute_degree_features(adjacency=svc.build_adjacency(edges=_sample_edges()))
        node = features["hub"]
        assert set(node) == {
            "in_degree",
            "out_degree",
            "total_degree",
            "avg_neighbor_weight",
            "max_neighbor_weight",
            "is_hub",
            "is_leaf",
        }

    def test_average_and_max_neighbor_weight(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = {"x": [("a", 0.2), ("b", 0.8)], "a": [], "b": []}
        features = svc.compute_degree_features(adjacency=adjacency)
        assert features["x"]["avg_neighbor_weight"] == 0.5
        assert features["x"]["max_neighbor_weight"] == 0.8

    def test_empty_adjacency_features_empty(self):
        svc = KnowledgeGraphEmbeddingService()
        assert svc.compute_degree_features(adjacency={}) == {}

    def test_target_only_node_included(self):
        svc = KnowledgeGraphEmbeddingService()
        features = svc.compute_degree_features(adjacency={"a": [("b", 1.0)]})
        assert "b" in features
        assert features["b"]["out_degree"] == 0

    def test_leaf_flag_false_for_degree_two(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = {"a": [("b", 1.0)], "b": [("c", 1.0)], "c": []}
        features = svc.compute_degree_features(adjacency=adjacency)
        assert features["b"]["total_degree"] == 2
        assert features["b"]["is_leaf"] is False


class TestStructuralSimilarity:
    def test_identical_vectors_similarity_one(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "n1": {"in_degree": 2, "out_degree": 3, "avg_neighbor_weight": 0.4, "max_neighbor_weight": 0.8},
            "n2": {"in_degree": 2, "out_degree": 3, "avg_neighbor_weight": 0.4, "max_neighbor_weight": 0.8},
        }
        assert svc.structural_similarity(node_a="n1", node_b="n2", features=features) == 1.0

    def test_orthogonal_vectors_similarity_zero(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "n1": {"in_degree": 1, "out_degree": 0, "avg_neighbor_weight": 0.0, "max_neighbor_weight": 0.0},
            "n2": {"in_degree": 0, "out_degree": 1, "avg_neighbor_weight": 0.0, "max_neighbor_weight": 0.0},
        }
        assert svc.structural_similarity(node_a="n1", node_b="n2", features=features) == 0.0

    def test_missing_node_similarity_zero(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {"n1": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.1, "max_neighbor_weight": 0.2}}
        assert svc.structural_similarity(node_a="n1", node_b="missing", features=features) == 0.0

    def test_zero_vectors_similarity_zero(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "n1": {"in_degree": 0, "out_degree": 0, "avg_neighbor_weight": 0.0, "max_neighbor_weight": 0.0},
            "n2": {"in_degree": 0, "out_degree": 0, "avg_neighbor_weight": 0.0, "max_neighbor_weight": 0.0},
        }
        assert svc.structural_similarity(node_a="n1", node_b="n2", features=features) == 0.0

    def test_similarity_bounded_between_zero_and_one(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "n1": {"in_degree": 3, "out_degree": 1, "avg_neighbor_weight": 0.5, "max_neighbor_weight": 0.9},
            "n2": {"in_degree": 2, "out_degree": 2, "avg_neighbor_weight": 0.2, "max_neighbor_weight": 0.3},
        }
        s = svc.structural_similarity(node_a="n1", node_b="n2", features=features)
        assert 0.0 <= s <= 1.0


class TestFindSimilarNodes:
    def test_returns_at_most_top_k(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "q": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.5, "max_neighbor_weight": 1.0},
            "a": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.5, "max_neighbor_weight": 1.0},
            "b": {"in_degree": 1, "out_degree": 0, "avg_neighbor_weight": 0.4, "max_neighbor_weight": 0.4},
            "c": {"in_degree": 0, "out_degree": 1, "avg_neighbor_weight": 0.6, "max_neighbor_weight": 0.6},
        }
        result = svc.find_similar_nodes(query_node="q", features=features, top_k=2)
        assert len(result) == 2

    def test_excludes_query_node(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "q": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.5, "max_neighbor_weight": 1.0},
            "a": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.5, "max_neighbor_weight": 1.0},
        }
        result = svc.find_similar_nodes(query_node="q", features=features)
        assert all(item["node_id"] != "q" for item in result)

    def test_unknown_query_returns_empty(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {"a": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.1, "max_neighbor_weight": 0.2}}
        assert svc.find_similar_nodes(query_node="q", features=features) == []

    def test_negative_top_k_returns_empty(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "q": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.5, "max_neighbor_weight": 1.0},
            "a": {"in_degree": 1, "out_degree": 1, "avg_neighbor_weight": 0.5, "max_neighbor_weight": 1.0},
        }
        assert svc.find_similar_nodes(query_node="q", features=features, top_k=-1) == []

    def test_sorted_descending_by_similarity(self):
        svc = KnowledgeGraphEmbeddingService()
        features = {
            "q": {"in_degree": 3, "out_degree": 3, "avg_neighbor_weight": 0.8, "max_neighbor_weight": 0.9},
            "high": {"in_degree": 3, "out_degree": 3, "avg_neighbor_weight": 0.8, "max_neighbor_weight": 0.9},
            "mid": {"in_degree": 2, "out_degree": 1, "avg_neighbor_weight": 0.4, "max_neighbor_weight": 0.5},
            "low": {"in_degree": 0, "out_degree": 1, "avg_neighbor_weight": 0.1, "max_neighbor_weight": 0.2},
        }
        result = svc.find_similar_nodes(query_node="q", features=features, top_k=3)
        assert result[0]["node_id"] == "high"
        assert result[0]["similarity"] >= result[1]["similarity"] >= result[2]["similarity"]


class TestIdentifyBridges:
    def test_node_with_in_three_out_three_is_bridge(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = svc.build_adjacency(
            edges=[
                {"source_id": "a", "target_id": "bridge", "weight": 1.0},
                {"source_id": "b", "target_id": "bridge", "weight": 1.0},
                {"source_id": "c", "target_id": "bridge", "weight": 1.0},
                {"source_id": "bridge", "target_id": "x", "weight": 1.0},
                {"source_id": "bridge", "target_id": "y", "weight": 1.0},
                {"source_id": "bridge", "target_id": "z", "weight": 1.0},
            ]
        )
        bridges = svc.identify_bridges(adjacency=adjacency)
        assert "bridge" in bridges

    def test_leaf_node_not_bridge(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = svc.build_adjacency(edges=[{"source_id": "root", "target_id": "leaf", "weight": 1.0}])
        bridges = svc.identify_bridges(adjacency=adjacency)
        assert "leaf" not in bridges

    def test_empty_adjacency_no_bridges(self):
        svc = KnowledgeGraphEmbeddingService()
        assert svc.identify_bridges(adjacency={}) == []

    def test_multiple_bridges_sorted(self):
        svc = KnowledgeGraphEmbeddingService()
        adjacency = {
            "a": [("x", 1.0), ("y", 1.0)],
            "b": [("x", 1.0), ("y", 1.0)],
            "x": [("m", 1.0), ("n", 1.0)],
            "y": [("m", 1.0), ("n", 1.0)],
            "m": [],
            "n": [],
        }
        bridges = svc.identify_bridges(adjacency=adjacency)
        assert bridges == sorted(bridges)
        assert "x" in bridges
        assert "y" in bridges
