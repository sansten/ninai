"""Regression guard: submit_feedback (memory_enrichment.py) and the human
review queue's resolve endpoint (review_queue.py) used to hardcode
target_agent="FeedbackIntegrationAgent" — an enterprise-only agent with no
execution path in an unlicensed deployment. Nothing filters MemoryFeedback
by target_agent before applying it (FeedbackLearningAgent's write-time run
picks up all is_applied=False rows for the memory regardless), so feedback
was never silently dropped, but every row was mislabeled with a consumer
that never actually processes it — any future per-agent feedback analytics
would misattribute 100% of this feedback. Both now target
FeedbackLearningAgent, the actual local consumer.
"""
from __future__ import annotations

import inspect

from app.api.v1.endpoints import memory_enrichment, review_queue


def test_memory_enrichment_submit_feedback_targets_local_agent():
    source = inspect.getsource(memory_enrichment.submit_feedback)
    assert 'target_agent="FeedbackLearningAgent"' in source
    assert '"FeedbackIntegrationAgent"' not in source


def test_review_queue_resolve_targets_local_agent():
    source = inspect.getsource(review_queue.resolve_review_item)
    assert 'target_agent="FeedbackLearningAgent"' in source
    assert '"FeedbackIntegrationAgent"' not in source
