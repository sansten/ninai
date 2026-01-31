"""
Event Publishing Service - System events for webhooks and streaming

Publishes events when memory items are created, updated, reviewed, etc.
"""

import json
import hmac
import hashlib
from datetime import datetime
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.event import Event
from app.models.webhook_subscription import WebhookSubscription
from app.services.audit_service import AuditService


class EventPublishingService:
    """
    Publishes events for webhooks and event streaming.
    
    Handles:
    - Event creation and persistence
    - Webhook subscription matching
    - Event signing and delivery (via queue)
    - Retry logic and metrics
    """

    def __init__(self, db: AsyncSession, organization_id: str):
        self.db = db
        self.organization_id = organization_id
        self.audit_svc = AuditService(db)

    async def publish_event(
        self,
        event_type: str,
        resource_type: str,
        resource_id: str,
        payload: dict,
        actor_user_id: Optional[str] = None,
        actor_agent_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Event:
        """
        Publish an event and queue webhook deliveries.
        
        Args:
            event_type: e.g., "memory.created", "knowledge.reviewed"
            resource_type: e.g., "memory", "knowledge", "agent"
            resource_id: UUID of the affected resource
            payload: Event-specific data (dict)
            actor_user_id: User who triggered the event
            actor_agent_id: Agent who triggered the event
            trace_id: Request trace for debugging
            request_id: Request ID for idempotency
        
        Returns:
            Created Event object
        """
        event = Event(
            event_type=event_type,
            event_version=1,
            organization_id=self.organization_id,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload,
            actor_user_id=actor_user_id,
            actor_agent_id=actor_agent_id,
            trace_id=trace_id,
            request_id=request_id,
        )
        
        self.db.add(event)
        await self.db.flush()
        
        # Queue webhook deliveries asynchronously
        # In production, this would be a Celery task
        await self._queue_webhook_deliveries(event)
        
        # Audit log
        await self.audit_svc.log_event(
            event_type="event.published",
            actor_id=actor_user_id or actor_agent_id,
            organization_id=self.organization_id,
            resource_type="event",
            resource_id=event.id,
            success=True,
            details={
                "event_type": event_type,
                "resource_type": resource_type,
                "resource_id": str(resource_id),
            }
        )
        
        return event

    async def _queue_webhook_deliveries(self, event: Event) -> None:
        """Queue webhook deliveries for matching subscriptions."""
        # Find matching subscriptions
        query = select(WebhookSubscription).where(
            WebhookSubscription.organization_id == self.organization_id,
            WebhookSubscription.active == True,
        )
        result = await self.db.execute(query)
        subscriptions = result.scalars().all()
        
        for subscription in subscriptions:
            # Check event type filter
            if not self._matches_event_filter(event.event_type, subscription.event_types):
                continue
            
            # Check resource type filter
            if subscription.resource_types and not self._matches_resource_filter(
                event.resource_type, subscription.resource_types
            ):
                continue
            
            # Queue delivery (in production, use Celery)
            # For now, just mark it - webhook worker will pick it up
            await self._create_webhook_delivery(subscription, event)

    def _matches_event_filter(self, event_type: str, filter_str: str) -> bool:
        """Check if event type matches subscription filter."""
        if filter_str == "*":
            return True
        
        filters = [f.strip() for f in filter_str.split(",")]
        
        # Support wildcards: "memory.*" matches "memory.created"
        for f in filters:
            if f == event_type:
                return True
            if f.endswith(".*"):
                prefix = f[:-2]  # Remove .*
                if event_type.startswith(prefix + "."):
                    return True
        
        return False

    def _matches_resource_filter(self, resource_type: str, filter_str: str) -> bool:
        """Check if resource type matches subscription filter."""
        filters = [f.strip() for f in filter_str.split(",")]
        return resource_type in filters

    async def _create_webhook_delivery(
        self, subscription: WebhookSubscription, event: Event
    ) -> None:
        """Create a webhook delivery record."""
        # Sign the payload
        signature = self._sign_payload(
            event_type=event.event_type,
            resource_id=str(event.resource_id),
            timestamp=event.created_at.isoformat(),
            payload=event.payload,
            secret=subscription.secret,
            algorithm=subscription.signing_algorithm,
        )
        
        # In production, queue this to Celery:
        # from app.tasks.webhooks import deliver_webhook_task
        # await deliver_webhook_task.apply_async(
        #     args=[subscription.id, event.id, signature],
        #     queue='webhooks'
        # )
        
        # For now, store in delivery table
        # (implementation of webhook delivery model/service would follow)
        pass

    @staticmethod
    def _sign_payload(
        event_type: str,
        resource_id: str,
        timestamp: str,
        payload: dict,
        secret: str,
        algorithm: str = "sha256",
    ) -> str:
        """
        Generate HMAC signature for webhook payload.
        
        Signature is computed as: HMAC-SHA256(secret, message)
        where message = f"{event_type}.{resource_id}.{timestamp}.{json_payload}"
        """
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        message = f"{event_type}.{resource_id}.{timestamp}.{payload_json}"
        
        if algorithm == "sha256":
            signature = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            raise ValueError(f"Unsupported signing algorithm: {algorithm}")
        
        return signature

    async def get_events(
        self,
        event_type: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Event], int]:
        """
        Get events with optional filtering.
        
        Returns:
            (events, total_count)
        """
        query = select(Event).where(
            Event.organization_id == self.organization_id
        )
        
        if event_type:
            query = query.where(Event.event_type == event_type)
        
        if resource_type:
            query = query.where(Event.resource_type == resource_type)
        
        if resource_id:
            query = query.where(Event.resource_id == resource_id)
        
        # Get total count
        count_result = await self.db.execute(
            select(len(query.distinct(Event.id)))
        )
        total = count_result.scalar() or 0
        
        # Get paginated results
        query = query.order_by(Event.created_at.desc()).limit(limit).offset(offset)
        result = await self.db.execute(query)
        events = result.scalars().all()
        
        return events, total

    async def list_subscriptions(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[list[WebhookSubscription], int]:
        """List webhook subscriptions for the organization."""
        query = select(WebhookSubscription).where(
            WebhookSubscription.organization_id == self.organization_id
        )
        
        # Get total count
        count_result = await self.db.execute(
            select(len(query.distinct(WebhookSubscription.id)))
        )
        total = count_result.scalar() or 0
        
        # Get paginated results
        query = query.order_by(WebhookSubscription.created_at.desc()).limit(limit).offset(offset)
        result = await self.db.execute(query)
        subscriptions = result.scalars().all()
        
        return subscriptions, total

    async def create_subscription(
        self,
        url: str,
        event_types: str = "*",
        resource_types: Optional[str] = None,
        secret: str = "",
        max_retries: int = 5,
        rate_limit_per_minute: Optional[int] = None,
        description: Optional[str] = None,
        custom_headers: Optional[dict] = None,
        created_by_user_id: Optional[str] = None,
    ) -> WebhookSubscription:
        """Create a new webhook subscription."""
        # Validate URL
        if not url.startswith("https://"):
            raise ValueError("Webhook URL must use HTTPS in production")
        
        # Generate secret if not provided
        if not secret:
            import secrets
            secret = secrets.token_urlsafe(32)
        
        subscription = WebhookSubscription(
            organization_id=self.organization_id,
            url=url,
            event_types=event_types,
            resource_types=resource_types,
            secret=secret,
            max_retries=max_retries,
            rate_limit_per_minute=rate_limit_per_minute,
            description=description,
            custom_headers=custom_headers,
            created_by_user_id=created_by_user_id,
        )
        
        self.db.add(subscription)
        await self.db.flush()
        
        await self.audit_svc.log_event(
            event_type="webhook.subscription.created",
            actor_id=created_by_user_id,
            organization_id=self.organization_id,
            resource_type="webhook_subscription",
            resource_id=subscription.id,
            success=True,
            details={"url": url, "event_types": event_types}
        )
        
        return subscription

    async def update_subscription(
        self,
        subscription_id: str,
        **kwargs
    ) -> WebhookSubscription:
        """Update a webhook subscription."""
        result = await self.db.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.id == subscription_id,
                WebhookSubscription.organization_id == self.organization_id,
            )
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'url', 'event_types', 'resource_types', 'active', 'paused_at',
            'paused_reason', 'max_retries', 'retry_delay_seconds',
            'rate_limit_per_minute', 'description', 'custom_headers'
        }
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                setattr(subscription, key, value)
        
        await self.db.flush()
        
        await self.audit_svc.log_event(
            event_type="webhook.subscription.updated",
            actor_id=kwargs.get('updated_by_user_id'),
            organization_id=self.organization_id,
            resource_type="webhook_subscription",
            resource_id=subscription.id,
            success=True,
            details={"changes": list(kwargs.keys())}
        )
        
        return subscription

    async def delete_subscription(self, subscription_id: str) -> None:
        """Delete a webhook subscription."""
        result = await self.db.execute(
            select(WebhookSubscription).where(
                WebhookSubscription.id == subscription_id,
                WebhookSubscription.organization_id == self.organization_id,
            )
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")
        
        await self.db.delete(subscription)
        await self.db.flush()
        
        await self.audit_svc.log_event(
            event_type="webhook.subscription.deleted",
            actor_id=None,
            organization_id=self.organization_id,
            resource_type="webhook_subscription",
            resource_id=subscription_id,
            success=True,
        )
