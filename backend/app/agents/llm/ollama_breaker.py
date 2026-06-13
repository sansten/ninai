"""Compatibility shim — ollama_breaker was renamed to llm_breaker."""
from app.agents.llm.llm_breaker import create_llm_client as create_ollama_client

__all__ = ["create_ollama_client"]
