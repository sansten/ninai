"""LLM integration with circuit breaker protection.

Wraps all LLM API calls with circuit breaker to prevent cascade failures.
"""

from typing import Optional, Any, Dict
import logging

from app.core.circuit_breaker import (
    get_circuit_breaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
)

logger = logging.getLogger(__name__)


class LLMCircuitBreakerConfig:
    """Standard circuit breaker configs for different LLM providers."""
    
    # Ollama - local LLM, aggressive since service outage is critical
    OLLAMA = CircuitBreakerConfig(
        failure_threshold=3,  # Open after 3 failures
        recovery_timeout_seconds=15,  # Try recovery every 15s
        success_threshold=1,  # One success to close
        error_rate_threshold=0.5,
    )
    
    # Fail after 5 consecutive errors or 50%+ error rate
    OPENAI = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30,
        success_threshold=2,
        error_rate_threshold=0.5,
    )
    
    ANTHROPIC = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=30,
        success_threshold=2,
        error_rate_threshold=0.5,
    )
    
    LOCAL = CircuitBreakerConfig(
        failure_threshold=3,  # More aggressive for local models
        recovery_timeout_seconds=10,
        success_threshold=2,
        error_rate_threshold=0.5,
    )


async def call_llm_with_breaker(
    provider: str,
    func,
    *args,
    **kwargs,
) -> Any:
    """
    Call LLM API with circuit breaker protection.
    
    Args:
        provider: LLM provider name (openai, anthropic, local)
        func: Async function to call
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        CircuitBreakerOpen: If circuit breaker is open
        Exception: Original exception from LLM
    """
    # Get or create breaker for this provider
    breaker_name = f"llm_{provider}"
    
    # Select config based on provider
    provider_lower = provider.lower()
    if provider_lower == "openai":
        config = LLMCircuitBreakerConfig.OPENAI
    elif provider_lower == "anthropic":
        config = LLMCircuitBreakerConfig.ANTHROPIC
    elif provider_lower == "ollama":
        config = LLMCircuitBreakerConfig.OLLAMA
    else:
        config = LLMCircuitBreakerConfig.LOCAL
    
    breaker = await get_circuit_breaker(breaker_name, config)
    
    try:
        return await breaker.call(func, *args, **kwargs)
    except CircuitBreakerOpen as e:
        logger.error(f"LLM call blocked by circuit breaker: {e}")
        raise


async def get_llm_status() -> Dict[str, Any]:
    """Get circuit breaker status for all LLM providers."""
    from app.core.circuit_breaker import circuit_breaker_registry
    
    status = circuit_breaker_registry.get_all_status()
    
    # Filter to LLM breakers
    llm_status = {
        name: data
        for name, data in status.items()
        if name.startswith("llm_")
    }
    
    return llm_status
