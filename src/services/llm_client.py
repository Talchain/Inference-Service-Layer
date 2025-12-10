"""
LLM client with cost tracking and controls.

Wraps OpenAI/Anthropic with:
- Cost estimation and tracking
- Rate limiting
- Caching
- Fallback handling
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.config.llm_config import LLMConfig, get_llm_config
from src.infrastructure.memory_cache import get_memory_cache
from src.infrastructure.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class CostTracker:
    """Track LLM costs per session."""

    # Pricing (as of Nov 2024, verify current pricing)
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }

    def __init__(self, redis_client=None) -> None:
        """Initialize cost tracker."""
        self.redis = redis_client or get_redis_client()

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for LLM call."""
        if model not in self.PRICING:
            logger.warning(f"Unknown model pricing: {model}")
            return 0.0

        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    def track_cost(self, session_id: str, cost: float) -> float:
        """
        Add cost to session total.

        Returns: Total cost for session
        """
        if not self.redis:
            logger.warning("Redis not available, cost tracking disabled")
            return cost

        key = f"isl:llm_cost:{session_id}"

        try:
            current = float(self.redis.get(key) or 0.0)
            new_total = current + cost
            self.redis.setex(key, 86400, str(new_total))  # 24h TTL

            logger.info(
                "LLM cost tracked",
                extra={
                    "session_id": session_id,
                    "cost_added": cost,
                    "total_cost": new_total,
                },
            )

            return new_total
        except Exception as e:
            logger.error(f"Failed to track cost: {e}")
            return cost

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for session."""
        if not self.redis:
            return 0.0

        key = f"isl:llm_cost:{session_id}"
        try:
            return float(self.redis.get(key) or 0.0)
        except Exception:
            return 0.0

    def check_budget(self, session_id: str, max_cost: float) -> tuple[bool, float]:
        """
        Check if session within budget.

        Returns: (within_budget, current_cost)
        """
        current_cost = self.get_session_cost(session_id)
        within_budget = current_cost < max_cost

        return within_budget, current_cost


class LLMClient:
    """
    LLM client with cost controls and caching.

    Wraps OpenAI/Anthropic with production safeguards.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cost_tracker: Optional[CostTracker] = None,
        redis_client=None,
        memory_cache=None,
    ) -> None:
        """Initialize LLM client."""
        self.config = config or get_llm_config()
        self.cost_tracker = cost_tracker or CostTracker(redis_client)
        self.redis = redis_client or get_redis_client()
        self.memory_cache = memory_cache or get_memory_cache()

        # Initialize provider clients
        if self.config.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            self.openai = openai.Client(api_key=self.config.openai_api_key)
            self.anthropic_client = None

        elif self.config.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            if not self.config.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.config.anthropic_api_key
            )
            self.openai = None

        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        request_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Generate LLM completion with cost tracking.

        Args:
            messages: Chat messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max output tokens
            session_id: Session for cost tracking
            cache_key: Cache key for identical prompts
            request_id: Request ID for tracing

        Returns:
            {
                "content": "LLM response text",
                "usage": {"input_tokens": ..., "output_tokens": ...},
                "cost": 0.05,
                "cached": False
            }
        """
        # Check cache first
        if cache_key and self.config.enable_prompt_caching:
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.info(
                    "LLM cache hit",
                    extra={"cache_key": cache_key, "request_id": request_id},
                )
                return {**cached, "cached": True}

        # Check budget if session provided
        if session_id:
            within_budget, current_cost = self.cost_tracker.check_budget(
                session_id, self.config.max_cost_per_session
            )

            if not within_budget:
                raise ValueError(
                    f"Session {session_id} exceeded budget: "
                    f"${current_cost:.2f} >= ${self.config.max_cost_per_session}"
                )

            # Warn if approaching limit
            if current_cost > (
                self.config.max_cost_per_session * self.config.budget_warning_threshold
            ):
                logger.warning(
                    "Session approaching cost limit",
                    extra={
                        "session_id": session_id,
                        "current_cost": current_cost,
                        "max_cost": self.config.max_cost_per_session,
                    },
                )

        # Apply token limit
        max_tokens = max_tokens or self.config.max_tokens_per_request

        # Call LLM
        try:
            if self.config.provider == "openai":
                response = self._call_openai(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif self.config.provider == "anthropic":
                response = self._call_anthropic(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise ValueError(f"Unknown provider: {self.config.provider}")

            # Track cost
            cost = self.cost_tracker.estimate_cost(
                model=model,
                input_tokens=response["usage"]["input_tokens"],
                output_tokens=response["usage"]["output_tokens"],
            )

            if session_id:
                self.cost_tracker.track_cost(session_id, cost)

            response["cost"] = cost
            response["cached"] = False

            # Cache response
            if cache_key and self.config.enable_prompt_caching:
                self._cache_response(cache_key, response)

            logger.info(
                "LLM completion",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "input_tokens": response["usage"]["input_tokens"],
                    "output_tokens": response["usage"]["output_tokens"],
                    "cost": cost,
                },
            )

            return response

        except Exception as e:
            logger.error(
                f"LLM completion failed: {e}",
                extra={"request_id": request_id, "model": model},
            )
            raise

    def _call_openai(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        response = self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        }

    def _call_anthropic(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        # Convert OpenAI format to Anthropic format
        system_msg = None
        user_msgs = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_msgs.append(msg)

        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=user_msgs,
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """
        Get cached LLM response.

        Uses in-memory cache as primary, falls back to Redis if available.
        """
        # Try in-memory cache first (fast)
        if self.memory_cache:
            cached = self.memory_cache.get(cache_key)
            if cached:
                logger.debug(f"Memory cache hit for {cache_key}")
                return cached

        # Fall back to Redis (slower, but shared across instances)
        if self.redis:
            key = f"isl:llm_cache:{cache_key}"
            try:
                data = self.redis.get(key)
                if data:
                    response = json.loads(data)

                    # Populate memory cache for next time
                    if self.memory_cache:
                        self.memory_cache.set(
                            cache_key, response, ttl=self.config.cache_ttl_seconds
                        )

                    logger.debug(f"Redis cache hit for {cache_key}")
                    return response
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        return None

    def _cache_response(self, cache_key: str, response: Dict) -> None:
        """
        Cache LLM response.

        Caches in both memory (fast) and Redis (shared) if available.
        """
        # Remove non-serializable fields
        cacheable = {
            "content": response["content"],
            "usage": response["usage"],
            "cost": response["cost"],
        }

        # Cache in memory (always, fast)
        if self.memory_cache:
            try:
                self.memory_cache.set(
                    cache_key, cacheable, ttl=self.config.cache_ttl_seconds
                )
                logger.debug(f"Cached response in memory for {cache_key}")
            except Exception as e:
                logger.warning(f"Memory cache write failed: {e}")

        # Also cache in Redis (shared across instances, optional)
        if self.redis:
            key = f"isl:llm_cache:{cache_key}"
            try:
                self.redis.setex(
                    key, self.config.cache_ttl_seconds, json.dumps(cacheable)
                )
                logger.debug(f"Cached response in Redis for {cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
