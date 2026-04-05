from __future__ import annotations

from app.services.cognitive_fingerprint_service import (
    CognitiveFingerprintService,
    _MIN_SAMPLES_FOR_DETECTION,
)


class TestCognitiveFingerprintService:
    def test_no_alerts_before_baseline(self):
        svc = CognitiveFingerprintService()
        output = {"anomaly_score": 10.0, "confidence": 0.99}
        # Record 4 samples (below _MIN_SAMPLES_FOR_DETECTION of 5)
        for _ in range(_MIN_SAMPLES_FOR_DETECTION - 1):
            svc.update_fingerprint("agent_a", output)
        result = svc.detect_anomaly("agent_a", output)
        assert not result.anomalous
        assert result.alerts == []

    def test_normal_output_no_alerts(self):
        svc = CognitiveFingerprintService()
        # Varied baseline around 0.1 / 0.6 so variance is non-zero
        samples = [
            {"anomaly_score": v, "confidence": c}
            for v, c in [
                (0.08, 0.58), (0.09, 0.59), (0.10, 0.60),
                (0.11, 0.61), (0.10, 0.60), (0.09, 0.59),
                (0.10, 0.60), (0.11, 0.61), (0.09, 0.59), (0.10, 0.60),
            ]
        ]
        for s in samples:
            svc.update_fingerprint("agent_a", s)
        # Value well within one stddev of the established baseline
        result = svc.detect_anomaly("agent_a", {"anomaly_score": 0.10, "confidence": 0.60})
        assert not result.anomalous
        assert result.sample_count == 10

    def test_anomalous_output_flagged(self):
        svc = CognitiveFingerprintService()
        # Establish varied-but-tight baseline around 0.1
        for v in [0.08, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11, 0.09, 0.10]:
            svc.update_fingerprint("agent_a", {"anomaly_score": v})
        # Inject a huge outlier (z >> 2.5)
        result = svc.detect_anomaly("agent_a", {"anomaly_score": 99.0})
        assert result.anomalous
        assert len(result.alerts) == 1
        alert = result.alerts[0]
        assert alert.field == "anomaly_score"
        assert alert.agent_name == "agent_a"
        assert abs(alert.z_score) >= 2.5

    def test_get_fingerprint_returns_stats(self):
        svc = CognitiveFingerprintService()
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in vals:
            svc.update_fingerprint("agent_b", {"score": v})
        fp = svc.get_fingerprint("agent_b")
        assert "score" in fp
        assert fp["score"]["n"] == 5
        assert abs(fp["score"]["mean"] - 0.3) < 0.01

    def test_multiple_agents_independent(self):
        svc = CognitiveFingerprintService()
        # Varied baselines: alpha tight around 0.9, beta tight around 0.1
        alpha_vals = [0.88, 0.89, 0.90, 0.91, 0.90, 0.89, 0.90, 0.91, 0.89, 0.90]
        beta_vals  = [0.08, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11, 0.09, 0.10]
        for av, bv in zip(alpha_vals, beta_vals):
            svc.update_fingerprint("alpha", {"confidence": av})
            svc.update_fingerprint("beta", {"confidence": bv})
        # 0.90 is normal for alpha — should not alert
        r_alpha = svc.detect_anomaly("alpha", {"confidence": 0.90})
        # 0.9 is extreme (>2.5 stddev) for beta whose mean is ~0.1
        r_beta = svc.detect_anomaly("beta", {"confidence": 0.9})
        assert not r_alpha.anomalous
        assert r_beta.anomalous

    def test_update_fingerprint_accumulates(self):
        svc = CognitiveFingerprintService()
        for i in range(8):
            svc.update_fingerprint("agent_c", {"score": float(i)})
        fp = svc.get_fingerprint("agent_c")
        assert fp["score"]["n"] == 8
        # Mean of 0..7 = 3.5
        assert abs(fp["score"]["mean"] - 3.5) < 0.01
