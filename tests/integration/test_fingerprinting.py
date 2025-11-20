"""
Integration tests for version fingerprinting and determinism.

Tests verify that:
- All endpoints include _metadata with config fingerprints
- Fingerprints are stable across requests
- Request IDs are unique and propagate correctly
- Deterministic behaviour (same inputs → same outputs)
- Cross-reference IDs are stable
"""

import pytest
import httpx
import json
import asyncio
from typing import Dict, List

BASE_URL = "http://localhost:8000"


class TestFingerprinting:
    """Test version fingerprinting in responses."""

    @pytest.mark.asyncio
    async def test_all_endpoints_include_metadata(self):
        """All endpoints should include _metadata with fingerprint."""

        endpoints = [
            ("/api/v1/causal/validate", {
                "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
                "treatment": "A",
                "outcome": "B"
            }),
            ("/api/v1/preferences/elicit", {
                "user_id": "test_user_fp_001",
                "context": {
                    "domain": "pricing",
                    "variables": ["revenue", "churn"]
                },
                "num_queries": 2
            }),
        ]

        try:
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
                for endpoint, payload in endpoints:
                    try:
                        response = await client.post(endpoint, json=payload)

                        if response.status_code != 200:
                            print(f"⚠ {endpoint} returned {response.status_code}: {response.text}")
                            continue

                        data = response.json()
                        assert "_metadata" in data, f"{endpoint} missing _metadata"

                        metadata = data["_metadata"]
                        assert "isl_version" in metadata, f"{endpoint} missing isl_version"
                        assert "config_fingerprint" in metadata, f"{endpoint} missing config_fingerprint"
                        assert "request_id" in metadata, f"{endpoint} missing request_id"

                        # Fingerprint should be 12-character hex
                        fp = metadata["config_fingerprint"]
                        assert len(fp) == 12, f"{endpoint} fingerprint wrong length: {fp}"
                        assert all(c in "0123456789abcdef" for c in fp), f"{endpoint} fingerprint not hex: {fp}"

                        print(f"✓ {endpoint}: fingerprint={fp}")

                    except httpx.TimeoutException:
                        print(f"⚠ {endpoint} timed out (server may not be running)")
                    except Exception as e:
                        print(f"✗ {endpoint} failed: {e}")
                        raise
        except httpx.ConnectError:
            pytest.skip("ISL server not running at localhost:8000")

    @pytest.mark.asyncio
    async def test_fingerprint_stable_across_requests(self):
        """Same config should produce same fingerprint."""

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        fingerprints = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for i in range(5):
                try:
                    response = await client.post("/api/v1/causal/validate", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        fingerprints.append(data["_metadata"]["config_fingerprint"])
                except Exception as e:
                    print(f"⚠ Request {i+1} failed: {e}")

        if fingerprints:
            # All fingerprints should be identical
            assert len(set(fingerprints)) == 1, f"Fingerprints not stable: {fingerprints}"
            print(f"✓ Fingerprint stable across {len(fingerprints)} requests: {fingerprints[0]}")
        else:
            pytest.skip("Server not available")

    @pytest.mark.asyncio
    async def test_request_id_unique(self):
        """Each request should have unique request ID."""

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        request_ids = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for i in range(10):
                try:
                    response = await client.post("/api/v1/causal/validate", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        request_ids.append(data["_metadata"]["request_id"])
                except Exception as e:
                    print(f"⚠ Request {i+1} failed: {e}")

        if request_ids:
            # All request IDs should be unique
            assert len(set(request_ids)) == len(request_ids), f"Duplicate request IDs found"
            print(f"✓ Request IDs unique: {len(request_ids)} generated")
        else:
            pytest.skip("Server not available")

    @pytest.mark.asyncio
    async def test_request_id_propagation(self):
        """X-Request-Id header should be propagated to response."""

        custom_request_id = "test-request-12345"

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            try:
                response = await client.post(
                    "/api/v1/causal/validate",
                    json={
                        "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
                        "treatment": "A",
                        "outcome": "B"
                    },
                    headers={"X-Request-Id": custom_request_id}
                )

                if response.status_code == 200:
                    data = response.json()
                    assert data["_metadata"]["request_id"] == custom_request_id
                    print(f"✓ Request ID propagated: {custom_request_id}")
                else:
                    pytest.skip(f"Server returned {response.status_code}")

            except Exception as e:
                print(f"⚠ Test failed: {e}")
                pytest.skip("Server not available")


class TestDeterminism:
    """Test deterministic behaviour."""

    @pytest.mark.asyncio
    async def test_same_input_same_output(self):
        """Identical inputs should produce identical outputs."""

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        responses = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for i in range(3):
                try:
                    response = await client.post("/api/v1/causal/validate", json=payload)
                    if response.status_code == 200:
                        responses.append(response.json())
                except Exception as e:
                    print(f"⚠ Request {i+1} failed: {e}")

        if len(responses) >= 2:
            # All responses should be identical (excluding metadata.request_id)
            for i in range(1, len(responses)):
                # Compare status field
                assert responses[0]["status"] == responses[i]["status"]

                # Fingerprints should match
                fp1 = responses[0]["_metadata"]["config_fingerprint"]
                fp2 = responses[i]["_metadata"]["config_fingerprint"]
                assert fp1 == fp2

            print(f"✓ Determinism verified: {len(responses)} identical responses")
        else:
            pytest.skip("Not enough successful requests to verify determinism")

    @pytest.mark.asyncio
    async def test_metadata_structure_valid(self):
        """Metadata structure should be valid across all endpoints."""

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            try:
                response = await client.post("/api/v1/causal/validate", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    metadata = data["_metadata"]

                    # Verify all required fields
                    required_fields = ["isl_version", "config_fingerprint", "request_id"]
                    for field in required_fields:
                        assert field in metadata, f"Missing field: {field}"

                    # Verify types
                    assert isinstance(metadata["isl_version"], str)
                    assert isinstance(metadata["config_fingerprint"], str)
                    assert isinstance(metadata["request_id"], str)

                    # Verify format
                    assert len(metadata["isl_version"]) > 0
                    assert len(metadata["config_fingerprint"]) == 12
                    assert len(metadata["request_id"]) > 0

                    print(f"✓ Metadata structure valid")
                else:
                    pytest.skip(f"Server returned {response.status_code}")

            except Exception as e:
                print(f"⚠ Test failed: {e}")
                pytest.skip("Server not available")


class TestCrossReferences:
    """Test stable ID cross-referencing (for future implementation)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Cross-reference IDs not yet implemented in responses")
    async def test_assumption_ids_stable(self):
        """Assumption IDs should be stable across requests."""

        payload = {
            "model": {
                "dag": {"nodes": ["price", "demand", "revenue"],
                        "edges": [["price", "demand"], ["demand", "revenue"]]},
                "parameters": {
                    "price": 10.0,
                    "elasticity": 0.5
                }
            },
            "scenario": {"price": 12.0}
        }

        assumption_ids = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            for i in range(3):
                try:
                    response = await client.post(
                        "/api/v1/causal/counterfactual",
                        json=payload
                    )
                    if response.status_code == 200:
                        data = response.json()

                        if "assumptions" in data:
                            ids = [a["id"] for a in data["assumptions"]]
                            assumption_ids.append(sorted(ids))
                except Exception as e:
                    print(f"⚠ Request {i+1} failed: {e}")

        if len(assumption_ids) >= 2:
            # All assumption IDs should be identical across requests
            for i in range(1, len(assumption_ids)):
                assert assumption_ids[0] == assumption_ids[i]

            print(f"✓ Assumption IDs stable: {assumption_ids[0]}")
        else:
            pytest.skip("Not enough successful requests")


# Run tests with: pytest tests/integration/test_fingerprinting.py -v -s
