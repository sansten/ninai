"""Ninai × LangChain adapter.

Provides drop-in LangChain components backed by the Ninai API:

- ``NinaiChatMessageHistory`` — ``BaseChatMessageHistory`` implementation that
  persists every conversation turn as a Ninai memory.
- ``NinaiSearchTool`` — ``BaseTool`` for semantic memory search.
- ``NinaiMemoryTool`` — ``BaseTool`` for writing new memories.
- ``get_ninai_toolkit`` — convenience factory returning all tools.

Install extras::

    pip install "ninai[langchain]"

Example::

    from ninai import NinaiClient
    from ninai.adapters.langchain import NinaiChatMessageHistory, get_ninai_toolkit
    from langchain_core.runnables.history import RunnableWithMessageHistory

    client = NinaiClient(api_key="nai_...")
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: NinaiChatMessageHistory(session_id, client),
        input_messages_key="input",
        history_messages_key="history",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Type

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain extras are required for this adapter.\n"
        "Install with: pip install \"ninai[langchain]\""
    ) from exc

if TYPE_CHECKING:
    from ninai.client import NinaiClient


class NinaiChatMessageHistory(BaseChatMessageHistory):
    """Persists LangChain conversation turns in Ninai organisational memory.

    Each ``add_messages`` call writes one Ninai memory per message, tagged with
    the session ID and role so they are searchable across sessions.

    Args:
        session_id: Unique conversation identifier.
        ninai_client: An authenticated :class:`~ninai.NinaiClient` instance.
        scope: Ninai memory scope (``"personal"``, ``"team"``, ``"organization"``).
        extra_tags: Additional tags to attach to every stored message.
    """

    def __init__(
        self,
        session_id: str,
        ninai_client: "NinaiClient",
        *,
        scope: str = "personal",
        extra_tags: Optional[List[str]] = None,
    ) -> None:
        self.session_id = session_id
        self._client = ninai_client
        self._scope = scope
        self._extra_tags = extra_tags or []
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            self._client.memories.create(
                content=msg.content,
                title=f"[{role}] session:{self.session_id}",
                scope=self._scope,
                tags=["conversation", f"session:{self.session_id}", role] + self._extra_tags,
            )
            self._messages.append(msg)

    def clear(self) -> None:
        self._messages.clear()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class _SearchInput(BaseModel):
    query: str = Field(description="Natural-language search query")
    top_k: int = Field(default=5, description="Maximum number of results to return")


class NinaiSearchTool(BaseTool):
    """Search Ninai organisational memory for relevant context.

    Returns a newline-separated list of matching memory contents ranked by
    semantic similarity score.

    Args:
        ninai_client: An authenticated :class:`~ninai.NinaiClient` instance.
    """

    name: str = "ninai_search"
    description: str = (
        "Search organisational memory for relevant facts, past decisions, or "
        "prior conversations. Input: a natural-language query string."
    )
    args_schema: Type[BaseModel] = _SearchInput
    ninai_client: object = None

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self.ninai_client.memories.search(query, limit=top_k)
        if not result.items:
            return "No relevant memories found."
        return "\n".join(
            f"- [{getattr(r, 'score', 0):.2f}] {r.content}" for r in result.items
        )


class _StoreInput(BaseModel):
    content: str = Field(description="The fact or decision to remember")
    tags: str = Field(default="", description="Comma-separated tags (e.g. 'finance,Q3')")


class NinaiMemoryTool(BaseTool):
    """Store an important fact or decision in Ninai organisational memory.

    Args:
        ninai_client: An authenticated :class:`~ninai.NinaiClient` instance.
        scope: Ninai memory scope (default ``"personal"``).
    """

    name: str = "ninai_remember"
    description: str = (
        "Store an important fact, decision, or observation in Ninai "
        "organisational memory so it can be recalled later."
    )
    args_schema: Type[BaseModel] = _StoreInput
    ninai_client: object = None
    scope: str = "personal"

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        content: str,
        tags: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        mem = self.ninai_client.memories.create(
            content=content, tags=tag_list, scope=self.scope
        )
        return f"Stored memory {mem.id}: {content[:100]}"


def get_ninai_toolkit(ninai_client: "NinaiClient", *, scope: str = "personal") -> List[BaseTool]:
    """Return all Ninai LangChain tools ready for an agent executor.

    Args:
        ninai_client: An authenticated :class:`~ninai.NinaiClient` instance.
        scope: Memory scope for ``NinaiMemoryTool`` writes.

    Returns:
        List of :class:`~langchain_core.tools.BaseTool` instances.

    Example::

        from ninai.adapters.langchain import get_ninai_toolkit
        from langchain.agents import create_tool_calling_agent

        tools = get_ninai_toolkit(client)
        agent = create_tool_calling_agent(llm, tools, prompt)
    """
    return [
        NinaiSearchTool(ninai_client=ninai_client),
        NinaiMemoryTool(ninai_client=ninai_client, scope=scope),
    ]
