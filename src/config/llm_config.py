"""
LLM configuration and cost control for ISL.

Manages OpenAI/Anthropic API integration with safeguards:
- Rate limiting per user/session
- Cost tracking and budget limits
- Model selection (cheap vs expensive)
- Caching for repeated prompts
"""

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="LLM provider",
    )

    # Model selection
    extraction_model: str = Field(
        default="gpt-3.5-turbo",
        description="Cheaper model for value extraction",
    )
    consensus_model: str = Field(
        default="gpt-4",
        description="Better model for consensus generation",
    )

    # API keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key",
    )

    # Cost controls
    max_tokens_per_request: int = Field(
        default=2000,
        description="Maximum tokens per LLM call",
    )
    max_cost_per_session: float = Field(
        default=1.0,
        description="Maximum cost per deliberation session ($)",
    )
    budget_warning_threshold: float = Field(
        default=0.8,
        description="Warn when this fraction of budget used",
    )

    # Rate limiting
    max_requests_per_minute: int = Field(
        default=10,
        description="Max LLM requests per minute per user",
    )
    max_requests_per_session: int = Field(
        default=50,
        description="Max LLM requests total per session",
    )

    # Caching
    enable_prompt_caching: bool = Field(
        default=True,
        description="Cache LLM responses for identical prompts",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL (1 hour default)",
    )

    # Fallback
    fallback_to_rules: bool = Field(
        default=True,
        description="Fall back to rule-based if LLM fails/costly",
    )

    model_config = {"env_prefix": "LLM_"}


@lru_cache()
def get_llm_config() -> LLMConfig:
    """Get LLM configuration (cached)."""
    return LLMConfig(
        openai_api_key=os.getenv("LLM_OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("LLM_ANTHROPIC_API_KEY"),
    )
