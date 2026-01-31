"""Circuit breaker pattern implementation for fault tolerance.

Prevents cascade failures by stopping calls to failing services.
States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing) -> CLOSED
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Callable, Any, Optional, TypeVar
import asyncio
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Service failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        success_threshold: int = 2,
        error_rate_threshold: float = 0.5,  # 50% error rate triggers open
    ):
        """
        Args:
            failure_threshold: Number of failures to trigger OPEN state
            recovery_timeout_seconds: Time to wait before trying HALF_OPEN
            success_threshold: Successful calls needed in HALF_OPEN to close
            error_rate_threshold: Error rate (0.0-1.0) to trigger open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.success_threshold = success_threshold
        self.error_rate_threshold = error_rate_threshold


class CircuitBreaker:
    """Circuit breaker for async operations."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Args:
            name: Breaker identifier (e.g., "openai_api", "anthropic_api")
            config: CircuitBreakerConfig instance
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.total_calls = 0
        self.total_failures = 0
        self.state_changed_at = datetime.now(timezone.utc)
        
        self._lock = asyncio.Lock()

    async def call(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            # Check if should open based on timeout
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable (will retry in {self._time_until_recovery()}s)"
                    )
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.failure_count = 0
            self.total_calls += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info(f"Circuit breaker '{self.name}' closed (service recovered)")
            elif self.state == CircuitState.CLOSED:
                self.total_calls += 1

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = datetime.now(timezone.utc)
            self.total_calls += 1
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN goes back to OPEN
                self.state = CircuitState.OPEN
                self.state_changed_at = datetime.now(timezone.utc)
                logger.warning(
                    f"Circuit breaker '{self.name}' re-opened "
                    f"(recovery failed)"
                )
            elif self.state == CircuitState.CLOSED:
                # Check both failure count and error rate
                error_rate = self.total_failures / max(self.total_calls, 1)
                
                if (self.failure_count >= self.config.failure_threshold or
                    error_rate >= self.config.error_rate_threshold):
                    self.state = CircuitState.OPEN
                    self.state_changed_at = datetime.now(timezone.utc)
                    logger.error(
                        f"Circuit breaker '{self.name}' opened "
                        f"(failures: {self.failure_count}, error_rate: {error_rate:.2%})"
                    )

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout_seconds

    def _time_until_recovery(self) -> int:
        """Seconds until recovery attempt."""
        if not self.last_failure_time:
            return 0
        
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        remaining = max(0, self.config.recovery_timeout_seconds - elapsed)
        return int(remaining)

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        error_rate = self.total_failures / max(self.total_calls, 1)
        
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "error_rate": error_rate,
            "time_until_recovery_seconds": self._time_until_recovery() if self.state == CircuitState.OPEN else None,
            "state_changed_at": self.state_changed_at.isoformat(),
        }

    def reset(self):
        """Reset circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.total_failures = 0
        self.state_changed_at = datetime.now(timezone.utc)
        logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")


class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
                logger.info(f"Created circuit breaker '{name}'")
            return self._breakers[name]

    async def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker."""
        return self._breakers.get(name)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    async def reset_all(self):
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


async def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create circuit breaker from global registry."""
    return await circuit_breaker_registry.get_or_create(name, config)
