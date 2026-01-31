import os
import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
SUMMARY_PROMPT = os.getenv(
    "SUMMARY_PROMPT",
    "Summarize the following short-term memories into a concise, factual long-term memory entry. Focus on key actions, decisions, and important context."
)

async def summarize_short_term_memories(memories: list[str], prompt: str = None) -> str:
    """
    Summarize a list of short-term memory strings using Ollama LLM.
    Args:
        memories: List of memory strings.
        prompt: Optional custom prompt (otherwise uses env/default).
    Returns:
        str: Summarized text.
    """
    if not memories:
        return ""
    prompt = prompt or SUMMARY_PROMPT
    context = "\n".join(memories)
    full_prompt = f"{prompt}\n\n{context}"
    payload = {"model": "llama3", "prompt": full_prompt}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response") or data.get("text") or ""
