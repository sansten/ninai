"""Ninai × Google Agent Development Kit (ADK) adapter.

Provides Ninai-backed components for ADK agents:

- ``ninai_search_memory`` / ``ninai_store_memory`` — plain async functions
  suitable for wrapping with ``google.adk.tools.FunctionTool``.
- ``NinaiADKMemoryService`` — persists ADK session events in Ninai so
  future agents can recall prior sessions via semantic search.
- ``get_ninai_adk_tools`` — convenience factory returning ready-made
  ``FunctionTool`` instances for both operations.

Install extras::

    pip install "ninai[adk]"

Example::

    from ninai import NinaiClient
    from ninai.adapters.adk import get_ninai_adk_tools
    from google.adk import Agent

    client = NinaiClient(api_key="nai_...")
    tools = get_ninai_adk_tools(client)

    agent = Agent(
        name="enterprise_assistant",
        model="gemini-2.0-flash",
        tools=tools,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

try:
    from google.adk.tools import FunctionTool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Google ADK extras are required for this adapter.\n"
        "Install with: pip install \"ninai[adk]\""
    ) from exc

if TYPE_CHECKING:
    from ninai.client import NinaiClient


# ---------------------------------------------------------------------------
# Tool functions — plain callables that ADK wraps as FunctionTool
# ---------------------------------------------------------------------------

def _make_search_fn(ninai_client: "NinaiClient"):
    def ninai_search_memory(query: str, top_k: int = 5) -> dict:
        """Search Ninai organisational memory and return relevant context.

        Args:
            query: Natural-language search query.
            top_k: Maximum number of results (default 5).

        Returns:
            dict with ``results`` (list of content strings) and ``count``.
        """
        result = ninai_client.memories.search(query, limit=top_k)
        return {
            "results": [r.content for r in result.items],
            "count": len(result.items),
        }

    return ninai_search_memory


def _make_store_fn(ninai_client: "NinaiClient", scope: str):
    def ninai_store_memory(content: str, tags: str = "") -> dict:
        """Store an important fact or decision in Ninai organisational memory.

        Args:
            content: The text to remember.
            tags: Comma-separated tag list (e.g. ``"finance,Q3"``).

        Returns:
            dict with ``memory_id`` and ``status``.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        mem = ninai_client.memories.create(content=content, tags=tag_list, scope=scope)
        return {"memory_id": mem.id, "status": "stored"}

    return ninai_store_memory


# ---------------------------------------------------------------------------
# NinaiADKMemoryService
# ---------------------------------------------------------------------------

class NinaiADKMemoryService:
    """Persists ADK session events in Ninai memory.

    Attach this to your ADK runner to make prior conversations retrievable
    across sessions via ``ninai_search_memory``.

    Args:
        ninai_client: An authenticated :class:`~ninai.NinaiClient` instance.
        scope: Ninai memory scope for stored events (default ``"personal"``).
        tag_prefix: Prefix added to auto-generated tags (default ``"adk"``).

    Example::

        svc = NinaiADKMemoryService(client)
        svc.save_session_event("sess-99", {"role": "user", "content": "..."})
        hits = svc.search("Q3 planning")
    """

    def __init__(
        self,
        ninai_client: "NinaiClient",
        *,
        scope: str = "personal",
        tag_prefix: str = "adk",
    ) -> None:
        self._client = ninai_client
        self._scope = scope
        self._tag_prefix = tag_prefix

    def save_session_event(self, session_id: str, event: dict) -> None:
        """Write one ADK session event to Ninai memory.

        Args:
            session_id: The ADK session identifier.
            event: Dict with at least ``role`` and ``content`` keys.
        """
        content = event.get("content") or str(event)
        role = event.get("role", "unknown")
        self._client.memories.create(
            content=content,
            title=f"ADK [{role}] session:{session_id}",
            scope=self._scope,
            tags=[
                self._tag_prefix,
                f"session:{session_id}",
                f"role:{role}",
            ],
        )

    def search(self, query: str, limit: int = 5) -> List[str]:
        """Search prior ADK session events stored in Ninai.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results.

        Returns:
            List of matching content strings.
        """
        result = self._client.memories.search(query, limit=limit)
        return [r.content for r in result.items]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_ninai_adk_tools(
    ninai_client: "NinaiClient",
    *,
    scope: str = "personal",
) -> List[FunctionTool]:
    """Return Ninai search + store as ADK ``FunctionTool`` instances.

    Args:
        ninai_client: An authenticated :class:`~ninai.NinaiClient` instance.
        scope: Memory scope for store operations (default ``"personal"``).

    Returns:
        ``[FunctionTool(ninai_search_memory), FunctionTool(ninai_store_memory)]``

    Example::

        from ninai.adapters.adk import get_ninai_adk_tools
        from google.adk import Agent

        agent = Agent(
            name="assistant",
            model="gemini-2.0-flash",
            tools=get_ninai_adk_tools(client),
        )
    """
    return [
        FunctionTool(_make_search_fn(ninai_client)),
        FunctionTool(_make_store_fn(ninai_client, scope)),
    ]
