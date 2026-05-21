"""
V2 Graph-RAG Prompt Builder

Assembles the final LLM prompt for Phase 2 (inference) by:
  1. Serialising the retrieved graph subgraph into a readable text block
  2. Including recent session utterances as conversational context
  3. Wrapping everything with the system instruction that requests structured JSON output

Output schema the model must return:
{
  "response": "<natural language answer>",
  "cited_node_ids": ["<id1>", "<id2>", ...],
  "extracted_entities": [{"id": "...", "name": "...", "type": "..."}]
}
"""

from __future__ import annotations

from typing import Any

_MAX_GRAPH_NODES = 20
_MAX_QDRANT_CHUNKS = 5
_MAX_NODE_CONTENT_CHARS = 300


_SYSTEM_INSTRUCTION = """\
You are NINAI, a cognitive assistant with access to a chronological knowledge graph.
You will be given:
  - GRAPH CONTEXT: relevant nodes from the knowledge graph (each has an id and content)
  - EPISODE CONTEXT: recent vector-retrieved memory chunks
  - SESSION HISTORY: the most recent utterances in this conversation
  - USER INPUT: the current user message

Your task:
1. Reason using the provided context nodes.
2. Produce a helpful, accurate response.
3. Cite the exact node IDs from GRAPH CONTEXT that informed your answer.
4. Extract any NEW named entities from the user input (not already in the graph context).

Return ONLY valid JSON in this exact schema — no prose outside the JSON:
{
  "response": "<your natural language answer>",
  "cited_node_ids": ["<node_id_1>", "<node_id_2>"],
  "extracted_entities": [
    {"id": "<snake_case_id>", "name": "<entity name>", "type": "<concept|user|task|object>"}
  ]
}"""


def build_inference_prompt(
    user_input: str,
    graph_nodes: list[dict[str, Any]],
    qdrant_chunks: list[dict[str, Any]],
    session_utterances: list[dict[str, Any]],
) -> str:
    parts: list[str] = [_SYSTEM_INSTRUCTION, ""]

    # --- Graph context ---
    parts.append("=== GRAPH CONTEXT ===")
    for node in graph_nodes[:_MAX_GRAPH_NODES]:
        nid = node.get("id", "?")
        label = node.get("label", "Node")
        content = str(node.get("content") or node.get("text") or node.get("name") or "")
        content = content[:_MAX_NODE_CONTENT_CHARS]
        weight = node.get("weight", 0)
        parts.append(f"[{label} id={nid} weight={weight:.2f}] {content}")
    if not graph_nodes:
        parts.append("(no graph context retrieved)")

    # --- Episode context from Qdrant ---
    parts.append("")
    parts.append("=== EPISODE CONTEXT ===")
    for chunk in qdrant_chunks[:_MAX_QDRANT_CHUNKS]:
        payload = chunk.get("payload", {})
        text = str(payload.get("text") or payload.get("content") or "")[:_MAX_NODE_CONTENT_CHARS]
        score = chunk.get("score", 0)
        cid = chunk.get("id", "?")
        parts.append(f"[chunk id={cid} score={score:.3f}] {text}")
    if not qdrant_chunks:
        parts.append("(no episodic context retrieved)")

    # --- Session history ---
    parts.append("")
    parts.append("=== SESSION HISTORY ===")
    # Show oldest → newest
    for utt in reversed(session_utterances[-6:]):
        role = utt.get("role", "?")
        text = str(utt.get("text") or utt.get("content") or "")[:200]
        parts.append(f"{role.upper()}: {text}")
    if not session_utterances:
        parts.append("(new session — no prior turns)")

    # --- User input ---
    parts.append("")
    parts.append("=== USER INPUT ===")
    parts.append(user_input)
    parts.append("")
    parts.append("JSON response:")

    return "\n".join(parts)
