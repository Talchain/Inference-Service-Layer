"""
Additional tests for LLM client edge cases and error handling.

Covers:
- Redis failures
- Budget warnings
- Provider initialization errors
- Anthropic API calls
- Cache error handling
- Memory cache integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.config.llm_config import LLMConfig
from src.services.llm_client import LLMClient, CostTracker


class TestCostTrackerErrorHandling:
    """Test cost tracker error handling."""

    def test_track_cost_without_redis(self):
        """Test cost tracking when Redis unavailable."""
        tracker = CostTracker(redis_client=None)

        # Should not raise, just return the cost
        total_cost = tracker.track_cost("session_123", 0.05)

        assert total_cost == 0.05

    def test_track_cost_redis_error(self):
        """Test cost tracking when Redis throws error."""
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis connection failed")

        tracker = CostTracker(redis_client=mock_redis)

        # Should handle error gracefully
        total_cost = tracker.track_cost("session_123", 0.05)

        assert total_cost == 0.05

    def test_get_session_cost_redis_error(self):
        """Test getting session cost when Redis fails."""
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis error")

        tracker = CostTracker(redis_client=mock_redis)

        # Should return 0 on error
        cost = tracker.get_session_cost("session_123")

        assert cost == 0.0

    def test_estimate_cost_claude_models(self):
        """Test cost estimation for Claude models."""
        tracker = CostTracker(redis_client=None)

        opus_cost = tracker.estimate_cost("claude-3-opus", 1000, 500)
        sonnet_cost = tracker.estimate_cost("claude-3-sonnet", 1000, 500)

        # Opus is more expensive than Sonnet
        assert opus_cost > sonnet_cost
        assert opus_cost == pytest.approx(0.0525, abs=0.001)  # (1*0.015 + 0.5*0.075)
        assert sonnet_cost == pytest.approx(0.0105, abs=0.001)  # (1*0.003 + 0.5*0.015)


class TestLLMClientInitialization:
    """Test LLM client initialization errors."""

    def test_openai_init_without_api_key(self):
        """Test OpenAI initialization fails without API key."""
        config = LLMConfig(
            provider="openai",
            openai_api_key=None,  # No API key
        )

        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            LLMClient(config=config)

    @patch("src.services.llm_client.OPENAI_AVAILABLE", False)
    def test_openai_init_package_not_installed(self):
        """Test OpenAI initialization fails when package not installed."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
        )

        with pytest.raises(ImportError, match="openai package not installed"):
            LLMClient(config=config)

    def test_anthropic_init_without_api_key(self):
        """Test Anthropic initialization fails without API key."""
        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key=None,  # No API key
        )

        with pytest.raises(ValueError, match="Anthropic API key not configured"):
            LLMClient(config=config)

    @patch("src.services.llm_client.ANTHROPIC_AVAILABLE", False)
    def test_anthropic_init_package_not_installed(self):
        """Test Anthropic initialization fails when package not installed."""
        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key="test-key",
        )

        with pytest.raises(ImportError, match="anthropic package not installed"):
            LLMClient(config=config)

    # Note: Unknown provider validation happens at Pydantic level (LLMConfig),
    # not in LLMClient init, so we don't test it here


class TestLLMClientBudgetWarnings:
    """Test budget warning thresholds."""

    def test_budget_warning_triggered(self):
        """Test warning is logged when approaching budget limit."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            max_cost_per_session=1.0,
            budget_warning_threshold=0.8,  # Warn at 80%
        )

        mock_redis = Mock()
        mock_cost_tracker = Mock()
        mock_cost_tracker.check_budget.return_value = (True, 0.85)  # 85% of budget

        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_openai.chat.completions.create.return_value = mock_response

        client = LLMClient(
            config=config,
            cost_tracker=mock_cost_tracker,
            redis_client=mock_redis,
        )
        client.openai = mock_openai

        # Should log warning (captured in logs, not tested here)
        # But should still complete successfully
        with patch("src.services.llm_client.logger") as mock_logger:
            response = client.complete(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-3.5-turbo",
                session_id="test_session",
            )

            # Should have logged warning
            mock_logger.warning.assert_called()
            assert "approaching cost limit" in str(mock_logger.warning.call_args)


class TestLLMClientAnthropicCalls:
    """Test Anthropic-specific API calls."""

    @patch("src.services.llm_client.ANTHROPIC_AVAILABLE", True)
    def test_anthropic_call_success(self):
        """Test successful Anthropic API call."""
        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key="test-key",
            enable_prompt_caching=False,
        )

        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude response"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_anthropic = Mock()
        mock_anthropic.messages.create.return_value = mock_response

        with patch("src.services.llm_client.anthropic.Anthropic") as mock_anthropic_class:
            mock_anthropic_class.return_value = mock_anthropic

            client = LLMClient(config=config, redis_client=None)

            response = client.complete(
                messages=[
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "test"}
                ],
                model="claude-3-sonnet",
            )

            # Should have called Anthropic
            mock_anthropic.messages.create.assert_called_once()

            # Should have correct response
            assert response["content"] == "Claude response"
            assert response["usage"]["input_tokens"] == 100
            assert response["usage"]["output_tokens"] == 50

    @patch("src.services.llm_client.ANTHROPIC_AVAILABLE", True)
    def test_anthropic_system_message_conversion(self):
        """Test that system messages are converted correctly for Anthropic."""
        config = LLMConfig(
            provider="anthropic",
            anthropic_api_key="test-key",
            enable_prompt_caching=False,
        )

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_anthropic = Mock()
        mock_anthropic.messages.create.return_value = mock_response

        with patch("src.services.llm_client.anthropic.Anthropic") as mock_anthropic_class:
            mock_anthropic_class.return_value = mock_anthropic

            client = LLMClient(config=config, redis_client=None)

            client.complete(
                messages=[
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "User message"}
                ],
                model="claude-3-sonnet",
            )

            # Check that system message was passed separately
            call_kwargs = mock_anthropic.messages.create.call_args[1]
            assert call_kwargs["system"] == "System prompt"
            assert len(call_kwargs["messages"]) == 1  # Only user message


class TestLLMClientCacheErrorHandling:
    """Test cache error handling."""

    def test_cache_read_error_falls_back_to_api(self):
        """Test that cache read errors don't prevent API calls."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=True,
        )

        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis read failed")

        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_openai.chat.completions.create.return_value = mock_response

        client = LLMClient(config=config, redis_client=mock_redis)
        client.openai = mock_openai

        # Should handle cache error and call API
        response = client.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            cache_key="test_key",
        )

        # Should have called API despite cache error
        assert response["content"] == "Response"
        mock_openai.chat.completions.create.assert_called_once()

    def test_cache_write_error_doesnt_fail_request(self):
        """Test that cache write errors don't fail the request."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=True,
        )

        mock_redis = Mock()
        mock_redis.get.return_value = None  # Cache miss
        mock_redis.setex.side_effect = Exception("Redis write failed")

        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_openai.chat.completions.create.return_value = mock_response

        client = LLMClient(config=config, redis_client=mock_redis)
        client.openai = mock_openai

        # Should complete successfully despite cache write error
        response = client.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            cache_key="test_key",
        )

        assert response["content"] == "Response"

    def test_api_call_failure_raises_exception(self):
        """Test that API failures are properly raised."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=False,
        )

        mock_openai = Mock()
        mock_openai.chat.completions.create.side_effect = Exception("API Error")

        client = LLMClient(config=config, redis_client=None)
        client.openai = mock_openai

        # Should raise the API error
        with pytest.raises(Exception, match="API Error"):
            client.complete(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-3.5-turbo",
            )


class TestLLMClientMemoryCacheIntegration:
    """Test memory cache integration with LLM client."""

    def test_memory_cache_hit_before_redis(self):
        """Test that memory cache is checked before Redis."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=True,
        )

        mock_memory_cache = Mock()
        mock_memory_cache.get.return_value = {
            "content": "Memory cached",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "cost": 0.001
        }

        mock_redis = Mock()

        client = LLMClient(
            config=config,
            redis_client=mock_redis,
            memory_cache=mock_memory_cache,
        )

        response = client.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            cache_key="test_key",
        )

        # Should have used memory cache
        assert response["content"] == "Memory cached"
        assert response["cached"] is True

        # Should NOT have checked Redis
        mock_redis.get.assert_not_called()

    def test_redis_cache_populates_memory_cache(self):
        """Test that Redis cache hits populate memory cache."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=True,
            cache_ttl_seconds=3600,
        )

        mock_memory_cache = Mock()
        mock_memory_cache.get.return_value = None  # Memory cache miss

        mock_redis = Mock()
        mock_redis.get.return_value = '{"content": "Redis cached", "usage": {"input_tokens": 10, "output_tokens": 5}, "cost": 0.001}'

        client = LLMClient(
            config=config,
            redis_client=mock_redis,
            memory_cache=mock_memory_cache,
        )

        response = client.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            cache_key="test_key",
        )

        # Should have used Redis cache
        assert response["content"] == "Redis cached"

        # Should have populated memory cache for next time
        mock_memory_cache.set.assert_called_once()
        assert mock_memory_cache.set.call_args[0][0] == "test_key"
        assert mock_memory_cache.set.call_args[1]["ttl"] == 3600

    def test_memory_cache_write_on_api_response(self):
        """Test that API responses are written to memory cache."""
        config = LLMConfig(
            provider="openai",
            openai_api_key="test-key",
            enable_prompt_caching=True,
            cache_ttl_seconds=7200,
        )

        mock_memory_cache = Mock()
        mock_memory_cache.get.return_value = None  # Cache miss

        mock_redis = Mock()
        mock_redis.get.return_value = None  # Redis miss too

        mock_openai = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "API Response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_openai.chat.completions.create.return_value = mock_response

        client = LLMClient(
            config=config,
            redis_client=mock_redis,
            memory_cache=mock_memory_cache,
        )
        client.openai = mock_openai

        response = client.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-3.5-turbo",
            cache_key="test_key",
        )

        # Should have written to memory cache
        mock_memory_cache.set.assert_called_once()
        call_args = mock_memory_cache.set.call_args
        assert call_args[0][0] == "test_key"
        assert call_args[1]["ttl"] == 7200
        assert call_args[0][1]["content"] == "API Response"
