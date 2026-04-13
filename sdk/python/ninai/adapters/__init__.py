"""Ninai framework adapters.

Each adapter is an optional extra — the relevant framework must be installed.

LangChain::

    pip install "ninai[langchain]"

    from ninai.adapters.langchain import (
        NinaiChatMessageHistory,
        NinaiSearchTool,
        NinaiMemoryTool,
        get_ninai_toolkit,
    )

Google ADK::

    pip install "ninai[adk]"

    from ninai.adapters.adk import (
        NinaiADKMemoryService,
        get_ninai_adk_tools,
    )

All adapters::

    pip install "ninai[all]"
"""

# Adapters are imported on demand — importing this package does not pull in
# LangChain or ADK. Users import directly from the sub-modules.

__all__ = [
    "langchain",
    "adk",
]
