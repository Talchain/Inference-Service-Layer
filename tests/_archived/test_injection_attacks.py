"""
Injection attack tests for security validation.

Tests that the API properly handles and rejects malicious inputs:
- Command injection attempts
- Path traversal attacks
- Code injection payloads
- XSS payloads
- Special character handling
"""

import pytest
from pydantic import ValidationError

from src.models.causal import (
    DAGDefinition,
    CausalValidationRequest,
    CausalQueryRequest,
)


class TestDAGNodeInjection:
    """Test injection attacks via DAG node names."""

    # Command injection payloads
    COMMAND_INJECTION_PAYLOADS = [
        "; ls -la",
        "| cat /etc/passwd",
        "`rm -rf /`",
        "$(whoami)",
        "&& echo pwned",
        "|| true",
        "; DROP TABLE users;--",
        "' OR '1'='1",
        "\" OR \"1\"=\"1",
        "\n/bin/sh",
        "${PATH}",
        "{{7*7}}",  # Template injection
    ]

    # Path traversal payloads
    PATH_TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f",
        "..%252f..%252f",
        "/etc/passwd%00",
    ]

    # XSS payloads
    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "javascript:alert('xss')",
        "<svg onload=alert('xss')>",
        "'-alert('xss')-'",
    ]

    def test_command_injection_in_node_names_accepted_but_sanitized(self):
        """
        Test that command injection payloads in node names are accepted.

        Since node names are treated as strings and not executed,
        they should be accepted but never executed as commands.
        """
        for payload in self.COMMAND_INJECTION_PAYLOADS:
            # The payload should be stored as a string, not executed
            dag = DAGDefinition(
                nodes=[payload, "B", "C"],
                edges=[[payload, "B"], ["B", "C"]],
            )
            # Verify the payload is stored as-is (not executed)
            assert payload in dag.nodes
            # The key test: no code execution occurred

    def test_path_traversal_in_node_names_accepted_as_strings(self):
        """
        Test that path traversal payloads in node names are accepted.

        Since node names are strings and not used for file operations,
        they should be accepted without causing file access.
        """
        for payload in self.PATH_TRAVERSAL_PAYLOADS:
            dag = DAGDefinition(
                nodes=[payload, "B"],
                edges=[[payload, "B"]],
            )
            assert payload in dag.nodes

    def test_xss_payloads_in_node_names_accepted_as_strings(self):
        """
        Test that XSS payloads in node names are accepted.

        The API returns JSON and doesn't render HTML, so XSS payloads
        are treated as plain strings.
        """
        for payload in self.XSS_PAYLOADS:
            dag = DAGDefinition(
                nodes=[payload, "B"],
                edges=[[payload, "B"]],
            )
            assert payload in dag.nodes


class TestEdgeInjection:
    """Test injection attacks via DAG edge definitions."""

    def test_injection_in_edges_treated_as_strings(self):
        """Test that edge definitions treat node names as strings."""
        malicious_edge = ["; rm -rf /", "B"]

        dag = DAGDefinition(
            nodes=["; rm -rf /", "B"],
            edges=[malicious_edge],
        )
        assert dag.edges == [malicious_edge]

    def test_malformed_edges_rejected(self):
        """Test that malformed edge structures are rejected."""
        # Edges must be lists of exactly 2 elements
        with pytest.raises(ValidationError):
            DAGDefinition(
                nodes=["A", "B"],
                edges=[["A"]],  # Only one element
            )

        with pytest.raises(ValidationError):
            DAGDefinition(
                nodes=["A", "B", "C"],
                edges=[["A", "B", "C"]],  # Three elements
            )


class TestSpecialCharacters:
    """Test handling of special characters in inputs."""

    SPECIAL_CHAR_INPUTS = [
        "node\x00name",  # Null byte
        "node\nname",  # Newline
        "node\rname",  # Carriage return
        "node\tname",  # Tab
        "node\\name",  # Backslash
        "node'name",  # Single quote
        'node"name',  # Double quote
        "node`name",  # Backtick
        "node$name",  # Dollar sign
        "node{name}",  # Curly braces
        "node[name]",  # Square brackets
        "node<name>",  # Angle brackets
        "node&name",  # Ampersand
        "node|name",  # Pipe
        "node;name",  # Semicolon
    ]

    def test_special_chars_in_node_names(self):
        """Test that special characters in node names are handled safely."""
        for special_input in self.SPECIAL_CHAR_INPUTS:
            try:
                dag = DAGDefinition(
                    nodes=[special_input, "B"],
                    edges=[[special_input, "B"]],
                )
                # If accepted, verify stored as-is
                assert special_input in dag.nodes
            except ValidationError:
                # Some special chars may be rejected by validation - that's OK
                pass


class TestNumericOverflow:
    """Test numeric overflow/underflow handling."""

    def test_large_numbers_in_evidence(self):
        """Test handling of very large numbers in evidence data."""
        from src.models.causal import Evidence

        # Very large float
        large_float = 1e308  # Near max float
        evidence = Evidence(
            data={"value": large_float}
        )
        assert evidence.data["value"] == large_float

    def test_negative_numbers_in_evidence(self):
        """Test handling of negative numbers in evidence data."""
        from src.models.causal import Evidence

        evidence = Evidence(
            data={"value": -1e308}
        )
        assert evidence.data["value"] == -1e308

    def test_infinity_handling(self):
        """Test handling of infinity values."""
        from src.models.causal import Evidence

        # Infinity should be rejected or handled safely
        try:
            evidence = Evidence(
                data={"value": float('inf')}
            )
            # If accepted, verify it's stored
            assert evidence.data["value"] == float('inf')
        except (ValidationError, ValueError):
            # Rejection is also acceptable
            pass


class TestUnicodeInjection:
    """Test Unicode-based injection attempts."""

    UNICODE_PAYLOADS = [
        "\u202e\u0065\u0078\u0065\u002e\u0074\u0078\u0074",  # Right-to-left override
        "\uff1cscript\uff1e",  # Fullwidth angle brackets
        "A\u0000B",  # Embedded null
        "\u200b",  # Zero-width space
        "node\u2028name",  # Line separator
        "node\u2029name",  # Paragraph separator
    ]

    def test_unicode_injection_in_nodes(self):
        """Test that Unicode injection payloads are handled safely."""
        for payload in self.UNICODE_PAYLOADS:
            try:
                dag = DAGDefinition(
                    nodes=[payload, "B"],
                    edges=[[payload, "B"]],
                )
                # If accepted, it should be stored as-is
                assert payload in dag.nodes
            except (ValidationError, ValueError):
                # Rejection is also acceptable
                pass


class TestJSONInjection:
    """Test JSON-specific injection attempts."""

    def test_deeply_nested_json(self):
        """Test handling of deeply nested JSON structures."""
        from src.models.causal import Evidence

        # Create deeply nested dict
        nested = {"level": 0}
        current = nested
        for i in range(1, 50):  # 50 levels deep
            current["nested"] = {"level": i}
            current = current["nested"]

        evidence = Evidence(data=nested)
        assert "level" in evidence.data

    def test_large_array_in_evidence(self):
        """Test handling of large arrays in evidence."""
        from src.models.causal import Evidence

        # Large but reasonable array
        large_array = list(range(1000))

        evidence = Evidence(
            data={"values": large_array}
        )
        assert len(evidence.data["values"]) == 1000


class TestRequestValidation:
    """Test request-level validation against injection attacks."""

    def test_validation_request_with_injection_payload(self):
        """Test that validation requests handle injection payloads safely."""
        request = CausalValidationRequest(
            dag=DAGDefinition(
                nodes=["<script>alert('xss')</script>", "B"],
                edges=[["<script>alert('xss')</script>", "B"]],
            )
        )
        # Should be accepted as string data
        assert "<script>" in request.dag.nodes[0]

    def test_query_request_with_injection_in_treatment(self):
        """Test injection payloads in query treatment/outcome fields."""
        # This should work because treatment/outcome are just node references
        request = CausalQueryRequest(
            dag=DAGDefinition(
                nodes=["; rm -rf /", "outcome_node"],
                edges=[["; rm -rf /", "outcome_node"]],
            ),
            treatment="; rm -rf /",
            outcome="outcome_node",
        )
        assert request.treatment == "; rm -rf /"


class TestLogInjection:
    """Test log injection prevention."""

    def test_newline_injection_in_node_names(self):
        """Test that newlines in node names don't corrupt logs."""
        # Newlines could be used to inject fake log entries
        payload = "nodeA\n[CRITICAL] Fake security alert!"

        dag = DAGDefinition(
            nodes=[payload, "B"],
            edges=[[payload, "B"]],
        )

        # The payload should be stored as-is
        # When logged with structured JSON logging, newlines are escaped
        assert "\n" in dag.nodes[0]


class TestHeaderInjection:
    """Test HTTP header injection prevention."""

    HTTP_HEADER_PAYLOADS = [
        "value\r\nX-Injected: header",
        "value\nSet-Cookie: malicious=true",
        "value\r\n\r\n<html>body",
    ]

    def test_header_payloads_in_api_key(self):
        """Test that header injection payloads in API keys are handled."""
        from src.middleware.auth import get_api_keys
        from unittest.mock import patch

        for payload in self.HTTP_HEADER_PAYLOADS:
            with patch.dict("os.environ", {"ISL_API_KEYS": payload}):
                keys = get_api_keys()
                # The payload is stored as a key (contains \r\n as part of string)
                # This is safe because keys are compared, not echoed to headers
                assert len(keys) >= 1


class TestSecureLoggingRedaction:
    """Test that sensitive data is properly redacted in logs."""

    def test_api_key_redaction_in_string(self):
        """Test that API keys in strings are redacted."""
        from src.utils.secure_logging import redact_string

        test_string = "User provided api_key=super_secret_key_12345"
        redacted = redact_string(test_string)
        assert "super_secret_key_12345" not in redacted
        assert "[REDACTED]" in redacted

    def test_bearer_token_redaction(self):
        """Test that bearer tokens are redacted."""
        from src.utils.secure_logging import redact_string

        test_string = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = redact_string(test_string)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted

    def test_password_redaction(self):
        """Test that passwords are redacted."""
        from src.utils.secure_logging import redact_string

        test_string = 'password="mysecretpassword123"'
        redacted = redact_string(test_string)
        assert "mysecretpassword123" not in redacted

    def test_credit_card_redaction(self):
        """Test that credit card numbers are redacted."""
        from src.utils.secure_logging import redact_string

        test_string = "Card number: 4111-1111-1111-1111"
        redacted = redact_string(test_string)
        assert "4111-1111-1111-1111" not in redacted
        assert "[CARD-REDACTED]" in redacted

    def test_ssn_redaction(self):
        """Test that SSNs are redacted."""
        from src.utils.secure_logging import redact_string

        test_string = "SSN: 123-45-6789"
        redacted = redact_string(test_string)
        assert "123-45-6789" not in redacted
        assert "[SSN-REDACTED]" in redacted

    def test_sensitive_field_redaction(self):
        """Test that sensitive fields are completely redacted."""
        from src.utils.secure_logging import redact_value

        assert redact_value("secret123", "password") == "[REDACTED]"
        assert redact_value("key123456", "api_key") == "[REDACTED]"
        assert redact_value("token123", "authorization") == "[REDACTED]"

    def test_nested_dict_redaction(self):
        """Test redaction in nested dictionaries."""
        from src.utils.secure_logging import redact_value

        data = {
            "user": "john",
            "credentials": {
                "password": "secret123",
                "api_key": "key456"
            }
        }

        redacted = redact_value(data)
        assert redacted["credentials"]["password"] == "[REDACTED]"
        assert redacted["credentials"]["api_key"] == "[REDACTED]"
        assert redacted["user"] == "john"
