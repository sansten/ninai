from app.core.readiness_score import score_checklist


def test_score_checklist_computes_go_when_thresholds_met():
    text = """
## Gate A: Autonomous Cognitive Loop (P0)
- Status: [x] Pass [ ] Partial [ ] Fail
- Status: [x] Pass [ ] Partial [ ] Fail

## Gate E: Cognitive Quality and Evaluation (P1)
- Status: [x] Pass [ ] Partial [ ] Fail
- Status: [x] Pass [ ] Partial [ ] Fail
- Status: [x] Pass [ ] Partial [ ] Fail
- Status: [ ] Pass [x] Partial [ ] Fail
- Status: [x] Pass [ ] Partial [ ] Fail
"""
    score = score_checklist(text)

    assert score.p0_total == 2
    assert score.p0_passed == 2
    assert score.p0_all_pass is True

    assert score.p1_total == 5
    assert score.p1_passed == 4
    assert score.p1_pass_rate == 0.8
    assert score.go is True


def test_score_checklist_fails_when_p0_not_all_pass():
    text = """
## Gate A: Autonomous Cognitive Loop (P0)
- Status: [x] Pass [ ] Partial [ ] Fail
- Status: [ ] Pass [x] Partial [ ] Fail

## Gate E: Cognitive Quality and Evaluation (P1)
- Status: [x] Pass [ ] Partial [ ] Fail
- Status: [x] Pass [ ] Partial [ ] Fail
"""
    score = score_checklist(text)

    assert score.p0_all_pass is False
    assert score.p1_pass_rate == 1.0
    assert score.go is False
