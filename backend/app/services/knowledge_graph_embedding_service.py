"""Knowledge graph embedding service (Phase 72)."""

from __future__ import annotations

import math
from statistics import pstdev


class KnowledgeGraphEmbeddingService:
    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def build_adjacency(self, *, edges: list[dict]) -> dict[str, list[tuple[str, float]]]:
        adjacency: dict[str, list[tuple[str, float]]] = {}

        for edge in edges or []:
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            if not source_id or not target_id:
                continue

            source = str(source_id)
            target = str(target_id)
            weight = self._safe_float(edge.get("weight"), 1.0)

            adjacency.setdefault(source, []).append((target, weight))
            adjacency.setdefault(target, [])

        return adjacency

    def compute_degree_features(
        self,
        *,
        adjacency: dict[str, list[tuple[str, float]]],
    ) -> dict[str, dict]:
        if not adjacency:
            return {}

        out_degree: dict[str, int] = {node: len(neighbors) for node, neighbors in adjacency.items()}
        in_degree: dict[str, int] = {node: 0 for node in adjacency}

        for neighbors in adjacency.values():
            for neighbor_id, _ in neighbors:
                if neighbor_id not in in_degree:
                    in_degree[neighbor_id] = 0
                    out_degree[neighbor_id] = 0
                in_degree[neighbor_id] += 1

        nodes = sorted(set(in_degree) | set(out_degree))
        total_degrees = [in_degree.get(node, 0) + out_degree.get(node, 0) for node in nodes]
        mean_degree = sum(total_degrees) / len(total_degrees)
        std_degree = pstdev(total_degrees) if len(total_degrees) > 1 else 0.0

        features: dict[str, dict] = {}
        for node in nodes:
            neighbors = adjacency.get(node, [])
            weights = [float(weight) for _, weight in neighbors]
            avg_neighbor_weight = sum(weights) / len(weights) if weights else 0.0
            max_neighbor_weight = max(weights) if weights else 0.0
            total_degree = in_degree.get(node, 0) + out_degree.get(node, 0)

            features[node] = {
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "total_degree": total_degree,
                "avg_neighbor_weight": avg_neighbor_weight,
                "max_neighbor_weight": max_neighbor_weight,
                "is_hub": total_degree > (mean_degree + std_degree),
                "is_leaf": total_degree == 1,
            }

        return features

    @staticmethod
    def _normalized_vector(node_id: str, features: dict[str, dict]) -> list[float]:
        components = ["in_degree", "out_degree", "avg_neighbor_weight", "max_neighbor_weight"]
        maxima = []
        for component in components:
            max_value = 0.0
            for node in features.values():
                value = float(node.get(component, 0.0))
                if value > max_value:
                    max_value = value
            maxima.append(max_value)

        node_features = features.get(node_id, {})
        vector: list[float] = []
        for idx, component in enumerate(components):
            raw = float(node_features.get(component, 0.0))
            max_value = maxima[idx]
            vector.append(raw / max_value if max_value > 0 else 0.0)

        return vector

    def structural_similarity(
        self,
        *,
        node_a: str,
        node_b: str,
        features: dict[str, dict],
    ) -> float:
        if node_a not in features or node_b not in features:
            return 0.0

        vec_a = self._normalized_vector(node_a, features)
        vec_b = self._normalized_vector(node_b, features)

        dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, round(similarity, 6)))

    def find_similar_nodes(
        self,
        *,
        query_node: str,
        features: dict[str, dict],
        top_k: int = 5,
    ) -> list[dict]:
        if query_node not in features:
            return []

        limit = max(0, int(top_k if top_k is not None else 5))
        candidates: list[dict] = []

        for node_id in features:
            if node_id == query_node:
                continue
            candidates.append(
                {
                    "node_id": node_id,
                    "similarity": self.structural_similarity(
                        node_a=query_node,
                        node_b=node_id,
                        features=features,
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (float(item.get("similarity", 0.0)), str(item.get("node_id", ""))),
            reverse=True,
        )
        return candidates[:limit]

    def identify_bridges(self, *, adjacency: dict[str, list[tuple[str, float]]]) -> list[str]:
        if not adjacency:
            return []

        features = self.compute_degree_features(adjacency=adjacency)
        bridges = [
            node_id
            for node_id, info in features.items()
            if int(info.get("in_degree", 0)) >= 2 and int(info.get("out_degree", 0)) >= 2
        ]
        bridges.sort()
        return bridges
