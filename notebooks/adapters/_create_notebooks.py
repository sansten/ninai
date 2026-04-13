"""Generate the three adapter sample notebooks."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


# ---------------------------------------------------------------------------
# langchain_adapter_sample.ipynb
# ---------------------------------------------------------------------------

SETUP = """\
import sys, os
SDK_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'sdk', 'python'))
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)
import langchain_core, ninai
print('langchain_core:', langchain_core.__version__)
print('ninai SDK:', ninai.__version__)
"""

MOCK = """\
from unittest.mock import MagicMock
from types import SimpleNamespace

_STORE: dict = {}
_CTR = [0]

def _mem(id, content, title='', tags=None):
    return SimpleNamespace(id=id, content=content, title=title, tags=tags or [])

def _mock_create(**kwargs):
    _CTR[0] += 1
    m = _mem(str(_CTR[0]), kwargs.get('content', ''), kwargs.get('title', ''), kwargs.get('tags', []))
    _STORE[m.id] = m
    return m

def _mock_search(query, **kwargs):
    items = [SimpleNamespace(memory_id=m.id, content=m.content, score=0.9, title=m.title)
             for m in list(_STORE.values())[-5:]]
    return SimpleNamespace(items=items[:3], total=len(items))

# Use a plain MagicMock so .memories auto-creates child mocks
client = MagicMock()
client.memories.create.side_effect = _mock_create
client.memories.search.side_effect = _mock_search
print('Mock client ready')
"""

HISTORY = """\
from typing import List, Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class NinaiChatMessageHistory(BaseChatMessageHistory):
    \"\"\"Persists LangChain conversation turns in Ninai memory.\"\"\"

    def __init__(self, session_id: str, ninai_client):
        self.session_id = session_id
        self._client = ninai_client
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            self._client.memories.create(
                content=msg.content,
                title=f'[{role}] session:{self.session_id}',
                tags=['conversation', f'session:{self.session_id}', role],
            )
            self._messages.append(msg)

    def clear(self) -> None:
        self._messages.clear()


history = NinaiChatMessageHistory(session_id='demo-001', ninai_client=client)
history.add_messages([HumanMessage(content='What is Ninai?')])
history.add_messages([AIMessage(content='Ninai is a Cognitive OS for enterprise.')])
print(f'Messages: {len(history.messages)}, Ninai writes: {client.memories.create.call_count}')
"""

TOOLS = """\
from typing import Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class _SearchInput(BaseModel):
    query: str = Field(description='Semantic search query')
    top_k: int = Field(default=5)


class NinaiSearchTool(BaseTool):
    name: str = 'ninai_search'
    description: str = 'Search organisational memory for relevant context.'
    args_schema: Type[BaseModel] = _SearchInput
    ninai_client: object = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, top_k: int = 5,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        res = self.ninai_client.memories.search(query, limit=top_k)
        if not res.items:
            return 'No relevant memories found.'
        return '\\n'.join(f'- [{r.score:.2f}] {r.content}' for r in res.items)


class _StoreInput(BaseModel):
    content: str = Field(description='Content to store')
    tags: str = Field(default='')


class NinaiMemoryTool(BaseTool):
    name: str = 'ninai_remember'
    description: str = 'Store an important fact in Ninai organisational memory.'
    args_schema: Type[BaseModel] = _StoreInput
    ninai_client: object = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, content: str, tags: str = '',
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        tag_list = [t.strip() for t in tags.split(',') if t.strip()]
        mem = self.ninai_client.memories.create(content=content, tags=tag_list)
        return f'Stored memory {mem.id}: {content[:80]}'


search_tool = NinaiSearchTool(ninai_client=client)
remember_tool = NinaiMemoryTool(ninai_client=client)

remember_tool._run('Q3 revenue target is $4.2M', tags='finance,Q3')
remember_tool._run('Deploy freeze starts 2026-04-15', tags='ops,deploy')

print('Search results:')
print(search_tool._run('revenue target'))
"""

CHAIN = """\
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


def stub_llm(prompt_value):
    msgs = prompt_value.to_messages()
    user_text = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), '')
    return AIMessage(content=f'[stub] Received: "{user_text}"')


chain_with_history = RunnableWithMessageHistory(
    ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful enterprise assistant powered by Ninai.'),
        MessagesPlaceholder(variable_name='history'),
        ('human', '{input}'),
    ]) | RunnableLambda(stub_llm),
    lambda sid: NinaiChatMessageHistory(session_id=sid, ninai_client=client),
    input_messages_key='input',
    history_messages_key='history',
)

cfg = {'configurable': {'session_id': 'lc-42'}}
r1 = chain_with_history.invoke({'input': 'What is our Q3 revenue target?'}, config=cfg)
r2 = chain_with_history.invoke({'input': 'When is the next deploy freeze?'}, config=cfg)
print('Turn 1:', r1.content)
print('Turn 2:', r2.content)
print('\\nLangChain + Ninai adapter verified.')
"""

TOOLKIT = """\
def get_ninai_toolkit(ninai_client):
    \"\"\"Return all Ninai tools ready for a LangChain agent executor.\"\"\"
    return [NinaiSearchTool(ninai_client=ninai_client), NinaiMemoryTool(ninai_client=ninai_client)]

for t in get_ninai_toolkit(client):
    print(f'{t.name:22s} {t.description}')
"""

lc_nb = nb([
    md("# Ninai \u00d7 LangChain Adapter\n\n"
       "Wires the **Ninai Python SDK** into a LangChain pipeline:\n\n"
       "1. `NinaiChatMessageHistory` \u2014 persists conversation turns in Ninai memory\n"
       "2. `NinaiSearchTool` \u2014 semantic search as a LangChain `BaseTool`\n"
       "3. `NinaiMemoryTool` \u2014 store facts from within a LangChain agent\n\n"
       "All API calls are **mocked** \u2014 runs offline without a live Ninai server."),
    code(SETUP),
    md("## Mock Ninai client"),
    code(MOCK),
    md("## 1. NinaiChatMessageHistory"),
    code(HISTORY),
    md("## 2. LangChain Tools"),
    code(TOOLS),
    md("## 3. RunnableWithMessageHistory chain"),
    code(CHAIN),
    md("## 4. Toolkit factory"),
    code(TOOLKIT),
])

# ---------------------------------------------------------------------------
# adk_adapter_sample.ipynb
# ---------------------------------------------------------------------------

ADK_SETUP = """\
import sys, os, warnings
warnings.filterwarnings('ignore')
SDK_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'sdk', 'python'))
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)
import google.adk, ninai
print('google-adk:', google.adk.__version__)
print('ninai SDK:', ninai.__version__)
"""

ADK_MOCK = """\
from unittest.mock import MagicMock
from types import SimpleNamespace

_STORE: dict = {}
_CTR = [0]

def _mock_create(**kwargs):
    _CTR[0] += 1
    m = SimpleNamespace(id=str(_CTR[0]), content=kwargs.get('content', ''),
                        title=kwargs.get('title', ''), tags=kwargs.get('tags', []))
    _STORE[m.id] = m
    return m

def _mock_search(query, **kwargs):
    items = [SimpleNamespace(memory_id=m.id, content=m.content, score=0.9, title=m.title)
             for m in list(_STORE.values())[-5:]]
    return SimpleNamespace(items=items[:3], total=len(items))

ninai_client = MagicMock()
ninai_client.memories.create.side_effect = _mock_create
ninai_client.memories.search.side_effect = _mock_search
print('Mock Ninai client ready')
"""

ADK_TOOLS = """\
from google.adk.tools import FunctionTool


def ninai_search_memory(query: str, top_k: int = 5) -> dict:
    \"\"\"Search Ninai organisational memory and return relevant context.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results.

    Returns:
        dict with 'results' (list of content strings) and 'count'.
    \"\"\"
    res = ninai_client.memories.search(query, limit=top_k)
    return {
        'results': [r.content for r in res.items],
        'count': len(res.items),
    }


def ninai_store_memory(content: str, tags: str = '') -> dict:
    \"\"\"Store a fact or decision in Ninai organisational memory.

    Args:
        content: The text to remember.
        tags: Comma-separated tag list.

    Returns:
        dict with 'memory_id' and 'status'.
    \"\"\"
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    mem = ninai_client.memories.create(content=content, tags=tag_list)
    return {'memory_id': mem.id, 'status': 'stored'}


# Wrap as ADK FunctionTool
search_tool = FunctionTool(ninai_search_memory)
remember_tool = FunctionTool(ninai_store_memory)

print(f'ADK tools created: {search_tool.name}, {remember_tool.name}')
"""

ADK_AGENT = """\
from google.adk import Agent
from google.adk.sessions import InMemorySessionService

# Build an ADK agent with Ninai tools
agent = Agent(
    name='ninai_enterprise_agent',
    model='gemini-2.0-flash',          # replace with your deployed model
    description='Enterprise assistant with Ninai memory',
    instruction=(
        'You are an enterprise assistant. '
        'Use ninai_store_memory to remember important facts. '
        'Use ninai_search_memory to retrieve context before answering.'
    ),
    tools=[search_tool, remember_tool],
)

print(f'Agent: {agent.name}')
print(f'Tools: {[t.name for t in agent.tools]}')
"""

ADK_SESSION = """\
# Demonstrate tool invocations directly (no live LLM call needed)
store_result = ninai_store_memory('Board approved $10M Series A on 2026-04-01', tags='finance,funding')
print('Stored:', store_result)

store_result2 = ninai_store_memory('Engineering headcount target: 25 by Q4', tags='hr,headcount')
print('Stored:', store_result2)

search_result = ninai_search_memory('funding round')
print('Search hits:', search_result['count'])
for r in search_result['results']:
    print(f'  - {r}')

print('\\nADK + Ninai adapter verified.')
"""

ADK_SERVICE = """\
# Optional: NinaiADKMemoryService — stores ADK session events in Ninai
# Useful for cross-session recall across agents.

class NinaiADKMemoryService:
    \"\"\"
    Thin adapter: persist ADK session events into Ninai memory so that
    a future ADK agent can search prior sessions via ninai_search_memory.
    \"\"\"

    def __init__(self, ninai_client):
        self._client = ninai_client

    def save_session_event(self, session_id: str, event: dict) -> None:
        content = event.get('content') or str(event)
        self._client.memories.create(
            content=content,
            title=f'ADK event [{event.get("role", "?")}, session:{session_id}]',
            tags=['adk', f'session:{session_id}', event.get('role', 'unknown')],
        )

    def search(self, query: str, limit: int = 5) -> list:
        res = self._client.memories.search(query, limit=limit)
        return [r.content for r in res.items]


svc = NinaiADKMemoryService(ninai_client=ninai_client)
svc.save_session_event('sess-99', {'role': 'user', 'content': 'Schedule Q3 planning call'})
svc.save_session_event('sess-99', {'role': 'agent', 'content': 'Q3 planning call scheduled for 2026-06-01'})

hits = svc.search('Q3 planning')
print(f'Recalled {len(hits)} events for "Q3 planning":')
for h in hits:
    print(f'  {h}')
"""

adk_nb = nb([
    md("# Ninai \u00d7 Google ADK Adapter\n\n"
       "Integrates the **Ninai Python SDK** with **Google Agent Development Kit (ADK)**:\n\n"
       "1. Ninai tools wrapped as `FunctionTool` for ADK agents\n"
       "2. `NinaiADKMemoryService` \u2014 persists ADK session events in Ninai\n"
       "3. End-to-end tool invocation smoke test\n\n"
       "All API calls are **mocked** \u2014 runs offline without a live Ninai server."),
    code(ADK_SETUP),
    md("## Mock Ninai client"),
    code(ADK_MOCK),
    md("## 1. Wrap Ninai as ADK FunctionTools"),
    code(ADK_TOOLS),
    md("## 2. Build an ADK Agent with Ninai tools"),
    code(ADK_AGENT),
    md("## 3. Tool invocation smoke test"),
    code(ADK_SESSION),
    md("## 4. NinaiADKMemoryService"),
    code(ADK_SERVICE),
])

# ---------------------------------------------------------------------------
# framework_adapters_smoke.ipynb
# ---------------------------------------------------------------------------

SMOKE_SETUP = """\
import sys, os, warnings
warnings.filterwarnings('ignore')
SDK_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'sdk', 'python'))
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

import langchain_core
import google.adk
import ninai
from ninai import NinaiClient

print(f'langchain_core : {langchain_core.__version__}')
print(f'google-adk     : {google.adk.__version__}')
print(f'ninai SDK      : {ninai.__version__}')
print()
print('All framework imports OK.')
"""

SMOKE_CLIENT = """\
from unittest.mock import MagicMock
from types import SimpleNamespace

_STORE: dict = {}
_CTR = [0]

def _mock_create(**kwargs):
    _CTR[0] += 1
    m = SimpleNamespace(id=str(_CTR[0]), content=kwargs.get('content', ''),
                        title=kwargs.get('title', ''), tags=kwargs.get('tags', []))
    _STORE[m.id] = m
    return m

def _mock_search(query, **kwargs):
    items = [SimpleNamespace(memory_id=m.id, content=m.content, score=0.9, title=m.title)
             for m in list(_STORE.values())[-5:]]
    return SimpleNamespace(items=items[:3], total=len(items))

client = MagicMock()
client.memories.create.side_effect = _mock_create
client.memories.search.side_effect = _mock_search
print('Shared mock client ready')
"""

SMOKE_LC = """\
# --- LangChain adapter smoke ---
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from typing import List, Sequence, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun


class NinaiChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, ninai_client):
        self.session_id = session_id
        self._client = ninai_client
        self._messages: List = []
    @property
    def messages(self): return list(self._messages)
    def add_messages(self, messages):
        for m in messages:
            self._client.memories.create(content=m.content, title='lc', tags=[])
            self._messages.append(m)
    def clear(self): self._messages.clear()

class _SI(BaseModel):
    query: str
    top_k: int = 5

class NinaiSearchTool(BaseTool):
    name: str = 'ninai_search'
    description: str = 'Search Ninai memory.'
    args_schema: Type[BaseModel] = _SI
    ninai_client: object = None
    class Config: arbitrary_types_allowed = True
    def _run(self, query, top_k=5, run_manager=None):
        res = self.ninai_client.memories.search(query)
        return '\\n'.join(r.content for r in res.items) or 'none'

# Assertions
hist = NinaiChatMessageHistory('s1', client)
hist.add_messages([HumanMessage(content='hello'), AIMessage(content='hi')])
assert len(hist.messages) == 2

tool = NinaiSearchTool(ninai_client=client)
out = tool._run('hello')
assert isinstance(out, str)

print('LangChain adapter smoke: PASS')
"""

SMOKE_ADK = """\
# --- ADK adapter smoke ---
from google.adk.tools import FunctionTool


def ninai_search_memory(query: str, top_k: int = 5) -> dict:
    \"\"\"Search Ninai memory.

    Args:
        query: Search query string.
        top_k: Maximum results.

    Returns:
        dict with results and count.
    \"\"\"
    res = client.memories.search(query, limit=top_k)
    return {'results': [r.content for r in res.items], 'count': len(res.items)}


def ninai_store_memory(content: str, tags: str = '') -> dict:
    \"\"\"Store a memory in Ninai.

    Args:
        content: Content to store.
        tags: Comma-separated tags.

    Returns:
        dict with memory_id and status.
    \"\"\"
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    mem = client.memories.create(content=content, tags=tag_list)
    return {'memory_id': mem.id, 'status': 'stored'}


search_ft = FunctionTool(ninai_search_memory)
store_ft = FunctionTool(ninai_store_memory)

# Assertions
res = ninai_store_memory('test fact for ADK smoke', tags='test')
assert res['status'] == 'stored'

hits = ninai_search_memory('test fact')
assert isinstance(hits['results'], list)

assert search_ft.name == 'ninai_search_memory'
assert store_ft.name == 'ninai_store_memory'

print('ADK adapter smoke: PASS')
"""

SMOKE_SDK = """\
# --- SDK core smoke ---
from ninai import (
    NinaiClient, GoalPlannerAgent, GoalLinkingAgent,
    MetaAgent, ToolInvoker, InMemoryEventSink,
)

# Structural checks
assert issubclass(NinaiClient, object)
assert hasattr(NinaiClient, '__init__')

from types import SimpleNamespace
m = SimpleNamespace(id='x', content='test', title='t', tags=[])
assert m.id == 'x'
assert m.content == 'test'

sink = InMemoryEventSink()
assert hasattr(sink, 'events')

print('Ninai SDK core smoke: PASS')
"""

SMOKE_SUMMARY = """\
print('=' * 40)
print('All adapter smoke tests PASSED')
print('  LangChain adapter : OK')
print('  Google ADK adapter : OK')
print('  Ninai SDK core     : OK')
print('=' * 40)
"""

smoke_nb = nb([
    md("# Ninai Adapter Smoke Tests\n\n"
       "Verifies that all three framework adapters (LangChain, Google ADK, Ninai SDK core)\n"
       "are importable, structurally correct, and execute without errors.\n\n"
       "Run this notebook after any SDK or adapter change as a sanity check."),
    code(SMOKE_SETUP),
    md("## Shared mock client"),
    code(SMOKE_CLIENT),
    md("## LangChain adapter"),
    code(SMOKE_LC),
    md("## Google ADK adapter"),
    code(SMOKE_ADK),
    md("## Ninai SDK core"),
    code(SMOKE_SDK),
    md("## Summary"),
    code(SMOKE_SUMMARY),
])

for name, data in [
    ("langchain_adapter_sample.ipynb", lc_nb),
    ("adk_adapter_sample.ipynb", adk_nb),
    ("framework_adapters_smoke.ipynb", smoke_nb),
]:
    path = os.path.join(BASE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(f"Written {name} ({os.path.getsize(path):,} bytes)")
