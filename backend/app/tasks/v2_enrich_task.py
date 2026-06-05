"""Async entity-extraction enrichment task (production write-throughput).

When NINAI_ASYNC_EXTRACT=1, DNCMemoryRouter.write() returns after the cheap
utterance node + episodic-vector upserts and enqueues this task on q.agent_enrich.
The task runs the deferred LLM entity extraction plus the entity / person-profile /
high-signal graph & Qdrant upserts, links them to the utterance, and back-fills the
episodic point — keeping the write hot path fast while the entity graph populates
within seconds in the background.

OFF by default: with NINAI_ASYNC_EXTRACT unset, write() extracts inline and never
enqueues this task, so behaviour is unchanged.
"""

from __future__ import annotations

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.tasks.async_runtime import run_async

logger = get_task_logger(__name__)


@celery_app.task(
    name="v2_enrich_extract_task",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
)
def v2_enrich_extract_task(
    self,
    *,
    tenant_id: str,
    session_id: str,
    utt_id: str,
    point_id: str | None,
    utterance_text: str,
):
    """Run the deferred entity extraction + graph/Qdrant back-fill for one utterance."""

    async def _run():
        from app.v2.pipeline.factory import get_v2_loop
        router = get_v2_loop()._router
        await router.enrich_entities_async(
            tenant_id=tenant_id,
            session_id=session_id,
            utt_id=utt_id,
            point_id=point_id,
            utterance_text=utterance_text,
        )

    try:
        run_async(_run())
    except Exception as exc:
        logger.warning(
            "v2_enrich_extract_task failed for utt_id=%s: %s — retrying", utt_id, exc
        )
        raise self.retry(exc=exc)


def enqueue_v2_enrich(
    *,
    tenant_id: str,
    session_id: str,
    utt_id: str,
    point_id: str | None,
    utterance_text: str,
):
    """Fire-and-forget enqueue. No-ops gracefully when the Celery broker is unavailable
    (e.g. local in-memory broker), so the write path never fails on enqueue."""
    try:
        broker = celery_app.conf.broker_url
        if not broker or str(broker).startswith("memory://"):
            return
        v2_enrich_extract_task.apply_async(
            kwargs={
                "tenant_id": tenant_id,
                "session_id": session_id,
                "utt_id": utt_id,
                "point_id": point_id,
                "utterance_text": utterance_text,
            },
            queue="q.agent_enrich",
            retry=False,
        )
    except Exception as exc:
        logger.warning("enqueue_v2_enrich failed (non-fatal): %s", exc)
