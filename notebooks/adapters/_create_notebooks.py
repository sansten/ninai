"""Generate the three adapter sample notebooks.

Output goes to sdk/python/notebooks/ — the canonical home for SDK examples.
The adapters/ folder keeps a copy for the monorepo notebooks tree.
"""
import json
import os

# Canonical output: sdk/python/notebooks/
_HERE = os.path.dirname(os.path.abspath(__file__))
SDK_NB_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "sdk", "python", "notebooks"))
os.makedirs(SDK_NB_DIR, exist_ok=True)
BASE = SDK_NB_DIR  # notebooks written here


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
# Shared mock helper (used by all three notebooks)
# ---------------------------------------------------------------------------

MOCK = """\
from unittest.mock import MagicMock
from types import SimpleNamespace

_STORE: dict = {}
_CTR = [0]

def _mock_create(**kwargs):
    _CTR[0] += 1
    m = SimpleNamespace(
        id=str(_CTR[0]), content=kwargs.get('content', ''),
        title=kwargs.get('title', ''), tags=kwargs.get('tags', []),
    )
    _STORE[m.id] = m
    return m

def _mock_search(query, **kwargs):
    items = [
        SimpleNamespace(memory_id=m.id, content=m.content, score=0.9, title=m.title)
        for m in list(_STORE.values())[-5:]
    ]
    return SimpleNamespace(items=items[:3], total=len(items))

client = MagicMock()
client.memories.create.side_effect = _mock_create
client.memories.search.side_effect = _mock_search
print('Mock client ready')
"""

# ---------------------------------------------------------------------------
# langchain_adapter_sample.ipynb
# ---------------------------------------------------------------------------

LC_SETUP = """\
import sys, os
SDK_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

import langchain_core
import ninai
from ninai.adapters.langchain import (
    NinaiChatMessageHistory,
    NinaiSearchTool,
    NinaiMemoryTool,
    get_ninai_toolkit,
)

print('langchain_core :', langchain_core.__version__)
print('ninai SDK      :', ninai.__version__)
print('Adapter imports OK.')
"""

LC_HISTORY = """\
from langchain_core.messages import HumanMessage, AIMessage

history = NinaiChatMessageHistory(session_id='demo-001', ninai_client=client)
history.add_messages([HumanMessage(content='What is Ninai?')])
history.add_messages([AIMessage(content='Ninai is a Cognitive OS for enterprise.')])
print(f'Messages in history : {len(history.messages)}')
print(f'Ninai writes        : {client.memories.create.call_count}')
"""

LC_TOOLS = """\
search_tool = NinaiSearchTool(ninai_client=client)
remember_tool = NinaiMemoryTool(ninai_client=client)

remember_tool._run('Q3 revenue target is $4.2M', tags='finance,Q3')
remember_tool._run('Deploy freeze starts 2026-04-15', tags='ops,deploy')

print('Search results:')
print(search_tool._run('revenue target'))
"""

LC_CHAIN = """\
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


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

LC_TOOLKIT = """\
print('Full toolkit:')
for t in get_ninai_toolkit(client):
    print(f'  {t.name:22s} {t.description}')
"""

LC_ENTERPRISE = """\
# Enterprise feature gating example.
# If your org does not have an Enterprise license, the server returns 403
# and the SDK raises EnterpriseFeatureRequired with a clear upgrade message.
from ninai.exceptions import EnterpriseFeatureRequired

try:
    # Simulated: calling an enterprise-only endpoint
    raise EnterpriseFeatureRequired(
        'This feature requires an Enterprise license.',
        feature='enterprise.advanced_observability',
    )
except EnterpriseFeatureRequired as e:
    print(f'Caught: {e}')
    print(f'Feature flag: {e.feature}')
"""

lc_nb = nb([
    md("# Ninai \u00d7 LangChain Adapter\n\n"
       "Demonstrates `ninai.adapters.langchain` — the official LangChain integration\n"
       "shipped **inside the Ninai SDK** (`pip install \"ninai[langchain]\"`):\n\n"
       "- `NinaiChatMessageHistory` — persists conversation turns in Ninai memory\n"
       "- `NinaiSearchTool` / `NinaiMemoryTool` — LangChain `BaseTool` wrappers\n"
       "- `get_ninai_toolkit` — convenience factory for agent executors\n\n"
       "All API calls are **mocked** \u2014 runs offline without a live Ninai server."),
    code(LC_SETUP),
    md("## Mock Ninai client\n\nIn production replace `client` with "
       "`NinaiClient(api_key=os.environ['NINAI_API_KEY'])`."),
    code(MOCK),
    md("## 1. NinaiChatMessageHistory"),
    code(LC_HISTORY),
    md("## 2. Search & Memory tools"),
    code(LC_TOOLS),
    md("## 3. RunnableWithMessageHistory chain"),
    code(LC_CHAIN),
    md("## 4. Full toolkit"),
    code(LC_TOOLKIT),
    md("## 5. Enterprise feature gating\n\n"
       "Community users get a clear `EnterpriseFeatureRequired` exception — "
       "not a raw 403 — when they hit an enterprise-only endpoint."),
    code(LC_ENTERPRISE),
])

# ---------------------------------------------------------------------------
# adk_adapter_sample.ipynb
# ---------------------------------------------------------------------------

ADK_SETUP = """\
import sys, os, warnings
warnings.filterwarnings('ignore')
SDK_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

import google.adk
import ninai
from ninai.adapters.adk import (
    get_ninai_adk_tools,
    NinaiADKMemoryService,
)

print('google-adk :', google.adk.__version__)
print('ninai SDK  :', ninai.__version__)
print('Adapter imports OK.')
"""

ADK_TOOLS = """\
tools = get_ninai_adk_tools(client)
print(f'ADK tools created: {[t.name for t in tools]}')
"""

ADK_AGENT = """\
from google.adk import Agent

agent = Agent(
    name='ninai_enterprise_agent',
    model='gemini-2.0-flash',
    description='Enterprise assistant with Ninai memory',
    instruction=(
        'You are an enterprise assistant. '
        'Use ninai_store_memory to remember important facts. '
        'Use ninai_search_memory to retrieve context before answering.'
    ),
    tools=tools,
)

print(f'Agent : {agent.name}')
print(f'Tools : {[t.name for t in agent.tools]}')
"""

ADK_DIRECT = """\
# Direct tool invocations — no live LLM call needed
search_fn, store_fn = tools[0].func, tools[1].func

store_fn('Board approved $10M Series A on 2026-04-01', tags='finance,funding')
store_fn('Engineering headcount target: 25 by Q4', tags='hr,headcount')

hits = search_fn('funding round')
print(f'Search hits: {hits["count"]}')
for r in hits['results']:
    print(f'  - {r}')

print('\\nADK + Ninai adapter verified.')
"""

ADK_SERVICE = """\
svc = NinaiADKMemoryService(ninai_client=client)
svc.save_session_event('sess-99', {'role': 'user',  'content': 'Schedule Q3 planning call'})
svc.save_session_event('sess-99', {'role': 'agent', 'content': 'Q3 planning call scheduled for 2026-06-01'})

hits = svc.search('Q3 planning')
print(f'Recalled {len(hits)} events:')
for h in hits:
    print(f'  {h}')
"""

ADK_ENTERPRISE = """\
from ninai.exceptions import EnterpriseFeatureRequired

try:
    raise EnterpriseFeatureRequired(
        'Advanced federated memory requires an Enterprise license.',
        feature='enterprise.federated_memory',
    )
except EnterpriseFeatureRequired as e:
    print(f'Caught: {e}')
    print(f'Feature: {e.feature}')
"""

adk_nb = nb([
    md("# Ninai \u00d7 Google ADK Adapter\n\n"
       "Demonstrates `ninai.adapters.adk` — the official Google ADK integration\n"
       "shipped **inside the Ninai SDK** (`pip install \"ninai[adk]\"`):\n\n"
       "- `get_ninai_adk_tools` — returns search + store as ADK `FunctionTool` instances\n"
       "- `NinaiADKMemoryService` — persists ADK session events in Ninai\n\n"
       "All API calls are **mocked** \u2014 runs offline without a live Ninai server."),
    code(ADK_SETUP),
    md("## Mock Ninai client"),
    code(MOCK),
    md("## 1. Create ADK tools from Ninai SDK"),
    code(ADK_TOOLS),
    md("## 2. Build an ADK Agent"),
    code(ADK_AGENT),
    md("## 3. Direct tool invocation smoke test"),
    code(ADK_DIRECT),
    md("## 4. NinaiADKMemoryService — cross-session recall"),
    code(ADK_SERVICE),
    md("## 5. Enterprise feature gating"),
    code(ADK_ENTERPRISE),
])

# ---------------------------------------------------------------------------
# framework_adapters_smoke.ipynb
# ---------------------------------------------------------------------------

SMOKE_SETUP = """\
import sys, os, warnings
warnings.filterwarnings('ignore')
SDK_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

import langchain_core
import google.adk
import ninai
from ninai import NinaiClient, EnterpriseFeatureRequired
from ninai.adapters.langchain import NinaiChatMessageHistory, NinaiSearchTool, NinaiMemoryTool, get_ninai_toolkit
from ninai.adapters.adk import get_ninai_adk_tools, NinaiADKMemoryService

print(f'langchain_core : {langchain_core.__version__}')
print(f'google-adk     : {google.adk.__version__}')
print(f'ninai SDK      : {ninai.__version__}')
print()
print('All imports OK.')
"""

SMOKE_LC = """\
from langchain_core.messages import HumanMessage, AIMessage

hist = NinaiChatMessageHistory('s1', client)
hist.add_messages([HumanMessage(content='hello'), AIMessage(content='hi')])
assert len(hist.messages) == 2, 'history length'

tool = NinaiSearchTool(ninai_client=client)
out = tool._run('hello')
assert isinstance(out, str), 'search returns str'

mem_tool = NinaiMemoryTool(ninai_client=client)
result = mem_tool._run('important decision', tags='eng,arch')
assert 'Stored' in result, 'store returns confirmation'

toolkit = get_ninai_toolkit(client)
assert len(toolkit) == 2, 'toolkit has 2 tools'

print('LangChain adapter smoke: PASS')
"""

SMOKE_ADK = """\
tools = get_ninai_adk_tools(client)
assert len(tools) == 2, 'adk tools count'
assert tools[0].name == 'ninai_search_memory'
assert tools[1].name == 'ninai_store_memory'

store_fn = tools[1].func
search_fn = tools[0].func
res = store_fn('adk smoke test fact', tags='test')
assert res['status'] == 'stored'

hits = search_fn('smoke test')
assert isinstance(hits['results'], list)

svc = NinaiADKMemoryService(ninai_client=client)
svc.save_session_event('s2', {'role': 'user', 'content': 'test event'})
recalled = svc.search('test event')
assert isinstance(recalled, list)

print('ADK adapter smoke: PASS')
"""

SMOKE_EXCEPTIONS = """\
try:
    raise EnterpriseFeatureRequired('needs license', feature='enterprise.scim')
except EnterpriseFeatureRequired as e:
    assert e.feature == 'enterprise.scim'
    assert 'sansten.com' in str(e)

print('EnterpriseFeatureRequired smoke: PASS')
"""

SMOKE_SDK = """\
from ninai import (
    NinaiClient, GoalPlannerAgent, GoalLinkingAgent,
    MetaAgent, ToolInvoker, InMemoryEventSink,
)

assert issubclass(NinaiClient, object)
sink = InMemoryEventSink()
assert hasattr(sink, 'events')

print('Ninai SDK core smoke: PASS')
"""

SMOKE_SUMMARY = """\
print('=' * 42)
print('All adapter smoke tests PASSED')
print('  ninai.adapters.langchain : OK')
print('  ninai.adapters.adk       : OK')
print('  EnterpriseFeatureRequired: OK')
print('  Ninai SDK core           : OK')
print('=' * 42)
"""

smoke_nb = nb([
    md("# Ninai Adapter Smoke Tests\n\n"
       "Validates all components of `ninai.adapters` in one shot.\n"
       "Run this after any SDK change as a quick sanity check.\n\n"
       "Install: `pip install \"ninai[all]\"`"),
    code(SMOKE_SETUP),
    md("## Shared mock client"),
    code(MOCK),
    md("## LangChain adapter"),
    code(SMOKE_LC),
    md("## Google ADK adapter"),
    code(SMOKE_ADK),
    md("## EnterpriseFeatureRequired exception"),
    code(SMOKE_EXCEPTIONS),
    md("## SDK core"),
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
