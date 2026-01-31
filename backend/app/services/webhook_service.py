"""Webhook outbox + delivery service."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone

import httpx
from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.webhook import WebhookDelivery, WebhookOutboxEvent, WebhookSubscription


def _fernet() -> Fernet:
    raw = (settings.SECRET_KEY or "dev").encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


class WebhookService:
    DEFAULT_RETRY_SCHEDULE_SECONDS = (5, 15, 60, 300, 900)

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def generate_secret() -> str:
        return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii").rstrip("=")

    def encrypt_secret(self, secret: str) -> str:
        return _fernet().encrypt(secret.encode("utf-8")).decode("utf-8")

    def decrypt_secret(self, secret_encrypted: str) -> str:
        return _fernet().decrypt(secret_encrypted.encode("utf-8")).decode("utf-8")

    async def create_subscription(
        self,
        *,
        organization_id: str,
        url: str,
        event_types: list[str] | None,
        description: str | None,
    ) -> tuple[WebhookSubscription, str]:
        secret = self.generate_secret()
        sub = WebhookSubscription(
            organization_id=organization_id,
            url=url,
            is_active=True,
            event_types=event_types or [],
            description=description,
            secret_encrypted=self.encrypt_secret(secret),
        )
        self.session.add(sub)
        await self.session.flush()
        return sub, secret

    async def emit_event(self, *, organization_id: str, event_type: str, payload: dict) -> None:
        event = WebhookOutboxEvent(
            organization_id=organization_id,
            event_type=event_type,
            payload=payload,
        )
        self.session.add(event)
        await self.session.flush()

        subs_res = await self.session.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.organization_id == organization_id,
                WebhookSubscription.is_active.is_(True),
            )
        )
        subs = list(subs_res.scalars().all())

        due = self._utcnow()
        for sub in subs:
            if sub.event_types and event_type not in sub.event_types:
                continue
            self.session.add(
                WebhookDelivery(
                    organization_id=organization_id,
                    subscription_id=sub.id,
                    outbox_event_id=event.id,
                    status="pending",
                    attempts=0,
                    next_attempt_at=due,
                )
            )

        await self.session.flush()

    async def dispatch_due_deliveries(self, *, limit: int = 50) -> int:
        now = self._utcnow()
        res = await self.session.execute(
            select(WebhookDelivery)
            .where(WebhookDelivery.status == "pending", WebhookDelivery.next_attempt_at <= now)
            .order_by(WebhookDelivery.next_attempt_at.asc())
            .limit(limit)
        )
        deliveries = list(res.scalars().all())
        if not deliveries:
            return 0

        sub_ids = {d.subscription_id for d in deliveries}
        event_ids = {d.outbox_event_id for d in deliveries}

        subs_res = await self.session.execute(select(WebhookSubscription).where(WebhookSubscription.id.in_(sub_ids)))
        subs_by_id = {s.id: s for s in subs_res.scalars().all()}

        events_res = await self.session.execute(select(WebhookOutboxEvent).where(WebhookOutboxEvent.id.in_(event_ids)))
        events_by_id = {e.id: e for e in events_res.scalars().all()}

        sent = 0
        async with httpx.AsyncClient(timeout=10.0) as client:
            for delivery in deliveries:
                sub = subs_by_id.get(delivery.subscription_id)
                ev = events_by_id.get(delivery.outbox_event_id)
                if not sub or not sub.is_active or not ev:
                    delivery.status = "failed"
                    delivery.last_error = "Subscription or event missing/inactive"
                    continue

                body = json.dumps(
                    {
                        "id": ev.id,
                        "type": ev.event_type,
                        "organization_id": ev.organization_id,
                        "created_at": ev.created_at.isoformat() if ev.created_at else None,
                        "payload": ev.payload,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")

                secret = self.decrypt_secret(sub.secret_encrypted)
                sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

                headers = {
                    "Content-Type": "application/json",
                    "X-Ninai-Event-Id": str(ev.id),
                    "X-Ninai-Event-Type": ev.event_type,
                    "X-Ninai-Signature": f"sha256={sig}",
                }

                try:
                    resp = await client.post(sub.url, content=body, headers=headers)
                    delivery.last_http_status = resp.status_code
                    if 200 <= resp.status_code < 300:
                        delivery.status = "delivered"
                        delivery.delivered_at = self._utcnow()
                        delivery.last_error = None
                        sent += 1
                    else:
                        self._schedule_retry(delivery, f"HTTP {resp.status_code}")
                except Exception as e:
                    self._schedule_retry(delivery, f"{type(e).__name__}: {e}")

        await self.session.flush()
        return sent

    def _schedule_retry(self, delivery: WebhookDelivery, error: str) -> None:
        delivery.attempts += 1
        delivery.last_error = error

        schedule = self.DEFAULT_RETRY_SCHEDULE_SECONDS
        if delivery.attempts >= len(schedule):
            delivery.status = "failed"
            delivery.next_attempt_at = self._utcnow() + timedelta(seconds=schedule[-1])
            return

        delay = schedule[delivery.attempts]
        delivery.next_attempt_at = self._utcnow() + timedelta(seconds=delay)
