"""
Framework Integrations

Adapters for LangChain, LlamaIndex, CrewAI, and LangGraph to use Ninai Memory OS.
Also includes external system sync adapters for Obsidian, Notion, and Roam Research.
"""

from .langchain_adapter import NinaiLangChainMemory
from .llamaindex_adapter import NinaiLlamaIndexVectorStore
from .langgraph_adapter import NinaiLangGraphCheckpointSaver, create_checkpoint_saver
from .external_sync import (
    ExternalSyncAdapter,
    ObsidianVaultAdapter,
    NotionDatabaseAdapter,
    RoamResearchAdapter,
    SyncManager,
    get_sync_manager,
)

# Optional framework integrations.
# Some third-party stacks (e.g., CrewAI) may have strict dependency constraints
# that can break import-time evaluation under certain Pydantic versions.
try:
    from .crewai_adapter import NinaiCrewAIMemory, create_crew_memory
except Exception as _exc:  # pragma: no cover
    NinaiCrewAIMemory = None  # type: ignore

    def create_crew_memory(*args, **kwargs):  # type: ignore
        raise ImportError(
            "CrewAI integration is unavailable (optional dependency failed to import)."
        ) from _exc

__all__ = [
    "NinaiLangChainMemory",
    "NinaiLlamaIndexVectorStore",
    "NinaiCrewAIMemory",
    "create_crew_memory",
    "NinaiLangGraphCheckpointSaver",
    "create_checkpoint_saver",
    "ExternalSyncAdapter",
    "ObsidianVaultAdapter",
    "NotionDatabaseAdapter",
    "RoamResearchAdapter",
    "SyncManager",
    "get_sync_manager",
]
