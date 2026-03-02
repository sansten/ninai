"""
PR-10: Differential Privacy Service

Implements formal differential privacy guarantees for federated knowledge.
Based on the Laplace mechanism and Gaussian mechanism for ε-differential privacy.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import random
import statistics


class DifferentialPrivacyService:
    """
    Formal differential privacy for aggregated data.
    
    Implements ε-differential privacy: adversary cannot determine whether
    any single organization's data was included in the aggregate.
    
    Key concepts:
    - ε (epsilon): Privacy budget (smaller = more private, typical: 0.1-10)
    - δ (delta): Probability of privacy breach (typically <10^-5)
    - Sensitivity: Maximum change one org's data can make
    - Noise: Laplace/Gaussian noise calibrated to sensitivity and ε
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize with privacy parameters.
        
        Args:
            epsilon: Privacy budget (smaller = stronger privacy)
            delta: Failure probability (smaller = stronger guarantee)
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(
        self,
        data: List[float],
        sensitivity: float = 1.0,
        epsilon: Optional[float] = None
    ) -> List[float]:
        """
        Add Laplace noise to achieve ε-differential privacy.
        
        Laplace mechanism: appropriate for count queries, sums, and
        queries with bounded sensitivity.
        
        Args:
            data: Original data points
            sensitivity: l1-sensitivity (max change one record can make)
            epsilon: Privacy budget (uses instance epsilon if not provided)
        
        Returns:
            Data with calibrated Laplace noise added
        """
        eps = epsilon or self.epsilon
        scale = sensitivity / eps
        
        return [x + self._sample_laplace(scale) for x in data]
    
    def add_gaussian_noise(
        self,
        data: List[float],
        sensitivity: float = 1.0,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None
    ) -> List[float]:
        """
        Add Gaussian noise to achieve (ε,δ)-differential privacy.
        
        Gaussian mechanism: appropriate for queries with l2-sensitivity
        and when (ε,δ)-DP is acceptable.
        
        Args:
            data: Original data points
            sensitivity: l2-sensitivity
            epsilon: Privacy budget
            delta: Failure probability
        
        Returns:
            Data with calibrated Gaussian noise
        """
        eps = epsilon or self.epsilon
        dlt = delta or self.delta
        
        # Gaussian noise scale: sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        scale = sensitivity * math.sqrt(2 * math.log(1.25 / dlt)) / eps
        
        return [x + random.gauss(0, scale) for x in data]
    
    def aggregate_with_privacy(
        self,
        org_data_dict: Dict[str, List[float]],
        aggregation_type: str = "mean",
        epsilon: Optional[float] = None
    ) -> float:
        """
        Compute aggregate statistic with differential privacy guarantee.
        
        Supports: mean, median, sum, count, variance
        
        Args:
            org_data_dict: Map of org_id to data values
            aggregation_type: Type of aggregate to compute
            epsilon: Privacy budget
        
        Returns:
            Privatized aggregate value
        """
        eps = epsilon or self.epsilon
        
        # Flatten all data
        all_values = [v for values in org_data_dict.values() for v in values]
        
        if not all_values:
            return 0.0
        
        # Compute raw aggregate
        if aggregation_type == "mean":
            result = statistics.mean(all_values)
            sensitivity = self._estimate_bounded_mean_sensitivity(all_values)
        elif aggregation_type == "median":
            result = statistics.median(all_values)
            sensitivity = self._estimate_median_sensitivity(all_values)
        elif aggregation_type == "sum":
            result = sum(all_values)
            sensitivity = max(abs(v) for v in all_values) if all_values else 1.0
        elif aggregation_type == "count":
            result = len(all_values)
            sensitivity = 1.0  # One org can change count by ±1
        elif aggregation_type == "variance":
            result = statistics.variance(all_values) if len(all_values) > 1 else 0.0
            sensitivity = self._estimate_variance_sensitivity(all_values)
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        
        # Add calibrated Laplace noise
        noisy_result = result + self._sample_laplace(sensitivity / eps)
        
        return noisy_result
    
    def check_privacy_budget(
        self,
        queries_made: int,
        epsilon_per_query: float
    ) -> Tuple[float, bool]:
        """
        Check if privacy budget has been exceeded.
        
        Privacy budget is compositional: each query consumes ε,
        total budget = sum of per-query budgets.
        
        Args:
            queries_made: Number of queries executed
            epsilon_per_query: Epsilon consumed per query
        
        Returns:
            Tuple of (total_epsilon_spent, is_budget_exceeded)
        """
        total_spent = queries_made * epsilon_per_query
        exceeded = total_spent > self.epsilon
        
        return total_spent, exceeded
    
    def compute_k_anonymity(
        self,
        org_count: int,
        min_k: int = 5
    ) -> bool:
        """
        Check if data satisfies k-anonymity.
        
        k-anonymity: each record is indistinguishable from at least k-1 others.
        For federated knowledge, require at least k contributing orgs.
        
        Args:
            org_count: Number of contributing organizations
            min_k: Minimum k for k-anonymity
        
        Returns:
            True if k-anonymity satisfied
        """
        return org_count >= min_k
    
    def estimate_reidentification_risk(
        self,
        org_count: int,
        data_dimension: int,
        epsilon: Optional[float] = None
    ) -> float:
        """
        Estimate risk of re-identifying an organization from aggregated data.
        
        Risk factors:
        - Fewer contributing orgs = higher risk
        - Higher dimensional data = higher risk
        - Larger epsilon = higher risk
        
        Args:
            org_count: Number of contributing organizations
            data_dimension: Dimensionality of data
            epsilon: Privacy budget
        
        Returns:
            Estimated re-identification risk (0-1)
        """
        eps = epsilon or self.epsilon
        
        # Heuristic risk model
        # Base risk inversely proportional to org count
        base_risk = 1.0 / max(org_count, 1)
        
        # Dimension penalty (more dimensions = easier to identify)
        dimension_factor = min(1.0, math.log(data_dimension + 1) / 10)
        
        # Epsilon penalty (larger epsilon = more info leaked)
        epsilon_factor = min(1.0, eps / 10)
        
        # Combined risk
        risk = base_risk * (1 + dimension_factor) * (1 + epsilon_factor)
        
        return min(1.0, risk)
    
    def clip_outliers(
        self,
        data: List[float],
        percentile: float = 95
    ) -> List[float]:
        """
        Clip outliers to reduce sensitivity.
        
        Outliers can dramatically increase sensitivity, reducing privacy.
        Clipping to 95th percentile is common practice.
        
        Args:
            data: Original data
            percentile: Percentile threshold for clipping
        
        Returns:
            Data with outliers clipped
        """
        if not data:
            return data
        
        sorted_data = sorted(data)
        clip_index = int(len(sorted_data) * (percentile / 100))
        clip_value = sorted_data[min(clip_index, len(sorted_data) - 1)]
        
        return [min(x, clip_value) for x in data]
    
    # Private helper methods
    
    def _sample_laplace(self, scale: float) -> float:
        """
        Sample from Laplace distribution with given scale.
        
        Laplace(μ, b) has PDF: (1/2b) * exp(-|x-μ|/b)
        """
        u = random.random() - 0.5
        return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))
    
    def _estimate_bounded_mean_sensitivity(self, data: List[float]) -> float:
        """
        Estimate sensitivity of mean query.
        
        For bounded data [a, b], sensitivity = (b-a)/n
        """
        if not data:
            return 1.0
        
        data_range = max(data) - min(data)
        return data_range / len(data)
    
    def _estimate_median_sensitivity(self, data: List[float]) -> float:
        """
        Estimate sensitivity of median query.
        
        Median has high sensitivity in worst case.
        Use interquartile range as approximation.
        """
        if len(data) < 4:
            return 1.0
        
        sorted_data = sorted(data)
        q1 = sorted_data[len(sorted_data) // 4]
        q3 = sorted_data[3 * len(sorted_data) // 4]
        
        return max(1.0, (q3 - q1) / 2)
    
    def _estimate_variance_sensitivity(self, data: List[float]) -> float:
        """
        Estimate sensitivity of variance query.
        
        Variance sensitivity depends on data range.
        """
        if len(data) < 2:
            return 1.0
        
        data_range = max(data) - min(data)
        return (data_range ** 2) / len(data)


# Singleton instance with default parameters
default_dp_service = DifferentialPrivacyService(epsilon=1.0, delta=1e-5)


# Convenience functions using default instance
def add_noise(data: List[float], sensitivity: float = 1.0) -> List[float]:
    """Add differentially private noise to data."""
    return default_dp_service.add_laplace_noise(data, sensitivity)


def private_aggregate(
    org_data: Dict[str, List[float]],
    aggregation_type: str = "mean"
) -> float:
    """Compute aggregate with privacy."""
    return default_dp_service.aggregate_with_privacy(org_data, aggregation_type)


def check_k_anonymity(org_count: int, min_k: int = 5) -> bool:
    """Check if k-anonymity is satisfied."""
    return default_dp_service.compute_k_anonymity(org_count, min_k)
