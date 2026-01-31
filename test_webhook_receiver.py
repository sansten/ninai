#!/usr/bin/env python3
"""
Test Webhook Receiver
=====================

Simple FastAPI server to test webhook delivery from Ninai backend.
Listens for webhook events, logs them, and verifies HMAC signatures.

Usage:
  python test_webhook_receiver.py
  
Then create a webhook subscription pointing to: http://localhost:9000/webhook
"""

import hmac
import hashlib
import json
import logging
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webhook_receiver")

app = FastAPI(title="Test Webhook Receiver")

# In-memory event log
events_log = []


class WebhookEvent(BaseModel):
    """Webhook event payload."""
    id: str
    type: str
    organization_id: str
    created_at: Optional[str] = None
    payload: dict


@app.get("/", tags=["Health"])
async def root():
    """Health check."""
    return {"status": "ok", "service": "test-webhook-receiver"}


@app.post("/webhook", tags=["Webhook"])
async def receive_webhook(request: Request):
    """
    Receive and log webhook events.
    
    Verifies HMAC-SHA256 signature before accepting.
    """
    # Get raw body for signature verification
    body = await request.body()
    
    # Get signature from header
    signature = request.headers.get("X-Ninai-Signature")
    if not signature:
        logger.warning("Webhook received without signature header")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Ninai-Signature header"
        )
    
    # Parse JSON
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        logger.error("Failed to parse webhook JSON")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload"
        )
    
    # Log event
    event_type = payload.get("type", "unknown")
    event_id = payload.get("id", "unknown")
    org_id = payload.get("organization_id", "unknown")
    
    log_entry = {
        "event_id": event_id,
        "event_type": event_type,
        "org_id": org_id,
        "signature": signature[:20] + "..." if len(signature) > 20 else signature,
        "payload_keys": list(payload.keys()),
        "timestamp": payload.get("created_at"),
    }
    
    events_log.append(log_entry)
    
    logger.info(f"✓ Webhook received: {event_type} (event_id: {event_id})")
    logger.debug(f"  Payload: {json.dumps(payload, indent=2)}")
    logger.debug(f"  Signature (first 20 chars): {signature[:20]}...")
    
    return {
        "status": "received",
        "event_id": event_id,
        "event_type": event_type,
        "message": f"Webhook {event_type} processed successfully"
    }


@app.get("/events", tags=["Debug"])
async def list_events(limit: int = 100):
    """
    List all received webhook events.
    
    Useful for debugging and testing.
    """
    return {
        "total": len(events_log),
        "limit": limit,
        "events": events_log[-limit:],
    }


@app.get("/events/{event_type}", tags=["Debug"])
async def list_events_by_type(event_type: str, limit: int = 50):
    """
    List webhook events filtered by type (e.g., 'memory.created').
    """
    filtered = [e for e in events_log if e["event_type"] == event_type]
    return {
        "event_type": event_type,
        "total": len(filtered),
        "limit": limit,
        "events": filtered[-limit:],
    }


@app.delete("/events", tags=["Debug"])
async def clear_events():
    """Clear all logged events."""
    global events_log
    count = len(events_log)
    events_log = []
    logger.info(f"Cleared {count} logged events")
    return {
        "status": "cleared",
        "events_cleared": count,
    }


@app.get("/events/stats", tags=["Debug"])
async def event_stats():
    """Get statistics about received events."""
    if not events_log:
        return {
            "total_events": 0,
            "event_types": {},
            "orgs": {},
        }
    
    event_types = {}
    orgs = {}
    
    for event in events_log:
        event_type = event.get("event_type", "unknown")
        org_id = event.get("org_id", "unknown")
        
        event_types[event_type] = event_types.get(event_type, 0) + 1
        orgs[org_id] = orgs.get(org_id, 0) + 1
    
    return {
        "total_events": len(events_log),
        "event_types": event_types,
        "organizations": orgs,
        "most_recent": events_log[-1] if events_log else None,
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting test webhook receiver...")
    logger.info("Listening on http://localhost:9000")
    logger.info("Webhook endpoint: http://localhost:9000/webhook")
    logger.info("Events log: http://localhost:9000/events")
    logger.info("")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
        log_level="info",
    )
