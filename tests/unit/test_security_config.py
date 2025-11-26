"""
Unit tests for security configuration validation.

Tests configuration validation, production hardening, and fail-fast behavior.
"""

import pytest
from unittest.mock import patch

from src.config import Settings, get_settings


class TestSettingsValidation:
    """Test cases for Settings validation."""

    def test_default_environment_is_development(self):
        """Test that default environment is development."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings()
            assert settings.ENVIRONMENT == "development"
            assert not settings.is_production()

    def test_production_environment_detection(self):
        """Test production environment detection."""
        with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.ENVIRONMENT == "production"
            assert settings.is_production()

    def test_cors_origins_list_parsing(self):
        """Test CORS origins are parsed correctly."""
        with patch.dict("os.environ", {"CORS_ORIGINS": "http://a.com,http://b.com"}):
            settings = Settings()
            origins = settings.get_cors_origins_list()
            assert origins == ["http://a.com", "http://b.com"]

    def test_cors_origins_list_with_whitespace(self):
        """Test CORS origins parsing handles whitespace."""
        with patch.dict("os.environ", {"CORS_ORIGINS": " http://a.com , http://b.com "}):
            settings = Settings()
            origins = settings.get_cors_origins_list()
            assert origins == ["http://a.com", "http://b.com"]

    def test_cors_origins_empty(self):
        """Test empty CORS origins returns empty list."""
        with patch.dict("os.environ", {"CORS_ORIGINS": ""}):
            settings = Settings()
            origins = settings.get_cors_origins_list()
            assert origins == []

    def test_trusted_proxies_list_parsing(self):
        """Test trusted proxies are parsed correctly."""
        with patch.dict("os.environ", {"TRUSTED_PROXIES": "10.0.0.0/8,172.16.0.0/12"}):
            settings = Settings()
            proxies = settings.get_trusted_proxies_list()
            assert proxies == ["10.0.0.0/8", "172.16.0.0/12"]

    def test_trusted_proxies_empty(self):
        """Test empty trusted proxies returns empty list."""
        with patch.dict("os.environ", {"TRUSTED_PROXIES": ""}):
            settings = Settings()
            proxies = settings.get_trusted_proxies_list()
            assert proxies == []


class TestProductionConfigValidation:
    """Test production configuration validation."""

    def test_development_validation_passes(self):
        """Test that development environment passes validation."""
        with patch.dict("os.environ", {"ENVIRONMENT": "development"}):
            settings = Settings()
            errors = settings.validate_production_config()
            assert errors == []

    def test_production_requires_api_keys(self):
        """Test that production requires ISL_API_KEYS."""
        with patch.dict("os.environ", {
            "ENVIRONMENT": "production",
            "ISL_API_KEYS": "",
            "CORS_ORIGINS": "https://app.example.com",
        }):
            settings = Settings()
            errors = settings.validate_production_config()
            assert any("ISL_API_KEYS" in e for e in errors)

    def test_production_rejects_wildcard_cors(self):
        """Test that production rejects wildcard CORS origins at validation level."""
        from pydantic import ValidationError

        with patch.dict("os.environ", {
            "ENVIRONMENT": "production",
            "ISL_API_KEYS": "valid_key",
            "CORS_ORIGINS": "*",
        }):
            # Pydantic validator should raise ValidationError
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            # Check error message mentions wildcard
            assert "wildcard" in str(exc_info.value).lower()

    def test_production_warns_default_cors(self):
        """Test that production warns about default CORS origins."""
        with patch.dict("os.environ", {
            "ENVIRONMENT": "production",
            "ISL_API_KEYS": "valid_key",
            "CORS_ORIGINS": "http://localhost:3000,http://localhost:8080",
        }):
            settings = Settings()
            errors = settings.validate_production_config()
            assert any("CORS_ORIGINS" in e for e in errors)

    def test_production_valid_config(self):
        """Test that valid production config passes."""
        with patch.dict("os.environ", {
            "ENVIRONMENT": "production",
            "ISL_API_KEYS": "secret_key_1,secret_key_2",
            "CORS_ORIGINS": "https://app.example.com,https://admin.example.com",
        }):
            settings = Settings()
            errors = settings.validate_production_config()
            assert errors == []


class TestSecuritySettings:
    """Test security-related settings."""

    def test_default_rate_limit(self):
        """Test default rate limit setting."""
        settings = Settings()
        assert settings.RATE_LIMIT_REQUESTS_PER_MINUTE == 100

    def test_custom_rate_limit(self):
        """Test custom rate limit setting."""
        with patch.dict("os.environ", {"RATE_LIMIT_REQUESTS_PER_MINUTE": "200"}):
            settings = Settings()
            assert settings.RATE_LIMIT_REQUESTS_PER_MINUTE == 200

    def test_default_request_size_limit(self):
        """Test default request size limit."""
        settings = Settings()
        assert settings.MAX_REQUEST_SIZE_MB == 10

    def test_custom_request_size_limit(self):
        """Test custom request size limit."""
        with patch.dict("os.environ", {"MAX_REQUEST_SIZE_MB": "50"}):
            settings = Settings()
            assert settings.MAX_REQUEST_SIZE_MB == 50

    def test_default_request_timeout(self):
        """Test default request timeout."""
        settings = Settings()
        assert settings.REQUEST_TIMEOUT_SECONDS == 60

    def test_cors_credentials_default_false(self):
        """Test CORS allow credentials defaults to false."""
        settings = Settings()
        assert settings.CORS_ALLOW_CREDENTIALS is False


class TestRedisSettings:
    """Test Redis configuration settings."""

    def test_default_redis_host(self):
        """Test default Redis host."""
        settings = Settings()
        assert settings.REDIS_HOST == "localhost"

    def test_default_redis_port(self):
        """Test default Redis port."""
        settings = Settings()
        assert settings.REDIS_PORT == 6379

    def test_redis_tls_disabled_by_default(self):
        """Test Redis TLS is disabled by default."""
        settings = Settings()
        assert settings.REDIS_TLS_ENABLED is False

    def test_redis_tls_enabled(self):
        """Test Redis TLS can be enabled."""
        with patch.dict("os.environ", {"REDIS_TLS_ENABLED": "true"}):
            settings = Settings()
            assert settings.REDIS_TLS_ENABLED is True

    def test_redis_password_none_by_default(self):
        """Test Redis password is None by default."""
        settings = Settings()
        assert settings.REDIS_PASSWORD is None

    def test_redis_max_connections(self):
        """Test Redis max connections setting."""
        settings = Settings()
        assert settings.REDIS_MAX_CONNECTIONS == 50
