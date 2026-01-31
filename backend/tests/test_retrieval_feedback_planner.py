from __future__ import annotations

from app.utils.retrieval_feedback import plan_relevance_feedback


def test_plan_relevance_feedback_empty_expected_is_noop():
    assert plan_relevance_feedback(expected_ids=[], retrieved_ids=["a"], k=10) == []


def test_plan_relevance_feedback_top1_relevant_emits_positive_only():
    actions = plan_relevance_feedback(expected_ids=["a"], retrieved_ids=["a", "b"], k=10)
    assert [(a.memory_id, a.relevant) for a in actions] == [("a", True)]


def test_plan_relevance_feedback_top1_wrong_but_relevant_in_topk_emits_negative_and_positive():
    actions = plan_relevance_feedback(expected_ids=["b"], retrieved_ids=["a", "b", "c"], k=10)
    assert [(a.memory_id, a.relevant) for a in actions] == [("a", False), ("b", True)]


def test_plan_relevance_feedback_no_relevant_in_topk_emits_negative_only():
    actions = plan_relevance_feedback(expected_ids=["z"], retrieved_ids=["a", "b"], k=10)
    assert [(a.memory_id, a.relevant) for a in actions] == [("a", False)]
