"""
Unit tests for LLM client and cost tracking.

Tests cost controls, caching, and fallback behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.config.llm_config import LLMConfig
from src.services.llm_client import LLMClient, CostTracker


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_estimate_cost_gpt35(self):
        """Test cost estimation for GPT-3.5."""
        tracker = CostTracker(redis_client=None)

        cost = tracker.estimate_cost(
            model="gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500,
        )

        # (1000/1000) * 0.0015 + (500/1000) * 0.002 = 0.0015 + 0.001 = 0.0025
        assert cost == pytest.approx(0.0025, abs=0.0001)

    def test_estimate_cost_gpt4(self):
        """Test cost estimation for GPT-4."""
        tracker = CostTracker(redis_client=None)

        cost = tracker.estimate_cost(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
        )

        # (1000/1000) * 0.03 + (500/1000) * 0.06 = 0.03 + 0.03 = 0.06
        assert cost == pytest.approx(0.06, abs=0.001)

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        tracker = CostTracker(redis_client=None)

        cost = tracker.estimate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should return 0 for unknown models
        assert cost == 0.0

    def test_track_cost_with_redis(self):
        """Test tracking cost with Redis."""
        mock_redis = Mock()
        mock_redis.get.return_value = "0.05"  # Existing cost
        mock_redis.setex = Mock()

        tracker = CostTracker(redis_client=mock_redis)

        total_cost = tracker.track_cost(
            session_id="test_session",
            cost=0.03,
        )

        # Should add to existing cost
        assert total_cost == 0.08

        # Should have called Redis
        mock_redis.get.assert_called_once()
        mock_redis.setex.assert_called_once()

    def test_check_budget_within_limit(self):
        """Test budget check when within limit."""
        mock_redis = Mock()
        mock_redis.get.return_value = "0.50"

        tracker = CostTracker(redis_client=mock_redis)

        within_budget, current_cost = tracker.check_budget(
            session_id="test_session",
            max_cost=1.0,
        )

        assert within_budget is True
        assert current_cost == 0.50

    def test_check_budget_exceeded(self):
        """Test budget check when exceeded."""
        mock_redis = Mock()
        mock_redis.get.return_value = "1.50"

        tracker = CostTracker(redis_client=mock_redis)

        within_budget, current_cost = tracker.check_budget(
            session_id="test_session",
            max_cost=1.0,
        )

        assert within_budget is False
        assert current_cost == 1.50


class TestLLMClient:
    """Test LLM client functionality."""

    def test_budget_check_before_request(self):
        """Test that budget is checked before making request."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            max_cost_per_session=1.0,
        )

        mock_redis = Mock()
        mock_redis.get.return_value = "1.50"  # Already over budget

        mock_cost_tracker = Mock()
        mock_cost_tracker.check_budget.return_value = (False, 1.50)

        client = LLMClient(
            config=config,
            cost_tracker=mock_cost_tracker,
            redis_client=mock_redis,
        )

        # Should raise ValueError due to budget exceeded
        with pytest.raises(ValueError, match="exceeded budget"):
            client.complete(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-3.5-turbo",
                session_id="test_session",
            )

    def test_cache_hit_skips_api_call(self):
        """Test that cache hit prevents API call."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=True,
        )

        mock_redis = Mock()
        mock_redis.get.return_value = '{"content": "cached response", "usage": {"input_tokens": 10, "output_tokens": 20}, "cost": 0.001}'

        client = LLMClient(
            config=config,
            cost_tracker=CostTracker(redis_client=mock_redis),
            redis_client=mock_redis,
        )

        # Patch OpenAI client to ensure it's not called
        with patch.object(client, "openai", create=True) as mock_openai:
            response = client.complete(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-3.5-turbo",
                cache_key="test_cache_key",
            )

            # Should not have called OpenAI
            assert not mock_openai.chat.completions.create.called

            # Should have cache hit
            assert response["cached"] is True
            assert response["content"] == "cached response"

    @patch("src.services.llm_client.OPENAI_AVAILABLE", True)
    def test_openai_call_success(self):
        """Test successful OpenAI API call."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=False,  # Disable cache for this test
        )

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_openai = Mock()
        mock_openai.chat.completions.create.return_value = mock_response

        client = LLMClient(config=config, redis_client=None)
        client.openai = mock_openai

        response = client.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
        )

        # Should have called OpenAI
        mock_openai.chat.completions.create.assert_called_once()

        # Should have correct response
        assert response["content"] == "AI response"
        assert response["usage"]["input_tokens"] == 100
        assert response["usage"]["output_tokens"] == 50
        assert response["cost"] > 0
        assert response["cached"] is False


# ARCHIVED: TestLLMIntegration class removed as it tested Habermas-specific services
# (ValueExtractorLLM, ConsensusGeneratorLLM) which have been deferred to TAE PoC v02
