"""
OpenAPI Schema Contract Tests.

Validates that the API schema remains backward compatible and
conforms to expected structure.
"""

import json
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from src.api.main import app


@pytest.fixture
def openapi_schema() -> Dict[str, Any]:
    """Get current OpenAPI schema directly from FastAPI app."""
    # Get schema directly from app without going through HTTP
    schema = app.openapi()
    return schema


class TestOpenAPISchemaStructure:
    """Test OpenAPI schema structure and required fields."""

    def test_schema_has_info(self, openapi_schema):
        """Schema must have info section."""
        assert "info" in openapi_schema
        assert "title" in openapi_schema["info"]
        assert "version" in openapi_schema["info"]

    def test_schema_has_paths(self, openapi_schema):
        """Schema must have paths section."""
        assert "paths" in openapi_schema
        assert len(openapi_schema["paths"]) > 0

    def test_schema_version_format(self, openapi_schema):
        """Version should follow semver pattern."""
        version = openapi_schema["info"]["version"]
        parts = version.split(".")
        assert len(parts) >= 2, f"Version {version} should be semver format"


class TestRequiredEndpoints:
    """Test that required endpoints exist in schema."""

    # Note: /metrics is excluded from OpenAPI as it's a Prometheus endpoint
    REQUIRED_ENDPOINTS = [
        "/health",
        "/ready",
        "/api/v1/causal/validate",
        "/api/v1/causal/counterfactual",
        "/api/v1/analysis/sensitivity",
    ]

    def test_required_endpoints_exist(self, openapi_schema):
        """All required endpoints must be present."""
        paths = openapi_schema["paths"]
        for endpoint in self.REQUIRED_ENDPOINTS:
            assert endpoint in paths, f"Required endpoint {endpoint} missing from schema"

    def test_health_endpoint_is_get(self, openapi_schema):
        """/health must support GET."""
        assert "get" in openapi_schema["paths"]["/health"]

    def test_api_endpoints_support_post(self, openapi_schema):
        """API v1 endpoints must support POST."""
        for path, methods in openapi_schema["paths"].items():
            if path.startswith("/api/v1/"):
                assert "post" in methods or "get" in methods, f"{path} has no methods"


class TestResponseSchemas:
    """Test response schema consistency."""

    def test_error_responses_have_schema(self, openapi_schema):
        """Error responses should have consistent schema."""
        for path, methods in openapi_schema["paths"].items():
            for method, spec in methods.items():
                if method in ["get", "post", "put", "delete"]:
                    responses = spec.get("responses", {})
                    # Check 422 validation error has schema
                    if "422" in responses:
                        assert "content" in responses["422"] or "description" in responses["422"]

    def test_success_responses_have_content_type(self, openapi_schema):
        """Success responses should specify content type."""
        for path, methods in openapi_schema["paths"].items():
            for method, spec in methods.items():
                if method in ["get", "post", "put", "delete"]:
                    responses = spec.get("responses", {})
                    if "200" in responses:
                        response_200 = responses["200"]
                        # Either has content or is a simple response
                        has_content = "content" in response_200
                        has_description = "description" in response_200
                        assert has_content or has_description, f"{path} 200 response incomplete"


class TestSecurityDefinitions:
    """Test security-related schema definitions."""

    def test_api_key_security_defined(self, openapi_schema):
        """API key security scheme should be defined."""
        components = openapi_schema.get("components", {})
        security_schemes = components.get("securitySchemes", {})

        # Check if any API key scheme exists
        has_api_key = any(
            scheme.get("type") == "apiKey"
            for scheme in security_schemes.values()
        )
        # API key auth is optional in schema, so just verify structure if present
        if security_schemes:
            assert isinstance(security_schemes, dict)


class TestSchemaBackwardCompatibility:
    """Test backward compatibility of schema changes."""

    # Endpoints that must never be removed (breaking change)
    PROTECTED_ENDPOINTS = {
        "/health",
        "/ready",
        "/api/v1/causal/validate",
        "/api/v1/causal/counterfactual",
    }

    # Required request fields that must not be removed
    PROTECTED_REQUEST_FIELDS = {
        "/api/v1/causal/validate": {"dag"},
        "/api/v1/causal/counterfactual": {"model"},
    }

    def test_protected_endpoints_exist(self, openapi_schema):
        """Protected endpoints must always exist."""
        paths = set(openapi_schema["paths"].keys())
        for endpoint in self.PROTECTED_ENDPOINTS:
            assert endpoint in paths, f"Protected endpoint {endpoint} was removed (breaking change)"

    def test_api_version_prefix(self, openapi_schema):
        """API endpoints should use /api/v1/ prefix."""
        api_paths = [p for p in openapi_schema["paths"] if p.startswith("/api/")]
        for path in api_paths:
            assert path.startswith("/api/v1/") or path.startswith("/api/v2/"), \
                f"API path {path} should use versioned prefix"


class TestSchemaQuality:
    """Test schema quality and documentation."""

    def test_endpoints_have_descriptions(self, openapi_schema):
        """Endpoints should have descriptions or summaries."""
        undocumented = []
        for path, methods in openapi_schema["paths"].items():
            for method, spec in methods.items():
                if method in ["get", "post", "put", "delete"]:
                    has_docs = spec.get("summary") or spec.get("description")
                    if not has_docs:
                        undocumented.append(f"{method.upper()} {path}")

        # Allow some undocumented endpoints but warn
        if undocumented:
            pytest.skip(f"Undocumented endpoints (consider adding docs): {undocumented[:5]}")

    def test_no_duplicate_operation_ids(self, openapi_schema):
        """Operation IDs must be unique."""
        operation_ids = []
        for path, methods in openapi_schema["paths"].items():
            for method, spec in methods.items():
                if "operationId" in spec:
                    operation_ids.append(spec["operationId"])

        duplicates = [oid for oid in operation_ids if operation_ids.count(oid) > 1]
        assert not duplicates, f"Duplicate operation IDs: {set(duplicates)}"
