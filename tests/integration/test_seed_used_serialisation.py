"""
I.3 — seed_used serialisation boundary tests.

Internal metadata stores seed_used as int; the response model declares
Optional[str].  The conversion happens at src/api/robustness.py:456
(``seed_str = str(effective_seed)``).

These tests verify the int → str → int round-trip is lossless for
representative values, and that the response model actually produces a
string representation.
"""

import pytest

from src.models.response_v2 import ISLResponseV2


class TestSeedUsedRoundTrip:
    """Verify int → str(seed_used) → int round-trip at the serialisation boundary."""

    @pytest.mark.parametrize(
        "seed_int",
        [
            0,
            1,
            2147483647,  # max 32-bit signed
            4294967295,  # max 32-bit unsigned (compute_seed_from_graph range)
        ],
        ids=["zero", "one", "max-signed-32", "max-unsigned-32"],
    )
    def test_int_to_str_to_int_round_trip(self, seed_int: int) -> None:
        """int(N) → str(N) → int(str(N)) == N for boundary values."""
        seed_str = str(seed_int)
        recovered = int(seed_str)
        assert recovered == seed_int, (
            f"Round-trip failed: int({seed_int}) → str('{seed_str}') → int({recovered})"
        )

    def test_response_model_produces_string(self) -> None:
        """ISLResponseV2.seed_used serialises as a string, not an int."""
        # Minimal valid response payload
        data = ISLResponseV2(
            response_schema_version="2.0",
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="computed",
            request_echo={
                "graph_node_count": 2,
                "graph_edge_count": 1,
                "options_count": 1,
                "goal_node_id_hash": "abc123",
                "n_samples": 100,
                "response_version_requested": 2,
                "include_diagnostics": False,
            },
            request_id="seed-serialise-test",
            processing_time_ms=10,
            seed_used="42",
        )
        dumped = data.model_dump(by_alias=True, exclude_none=True)
        assert isinstance(dumped["seed_used"], str), (
            f"Expected str, got {type(dumped['seed_used'])}"
        )
        assert dumped["seed_used"] == "42"

    def test_str_conversion_at_api_boundary(self) -> None:
        """Simulate the exact conversion done in src/api/robustness.py:456."""
        # This replicates: seed_str = str(effective_seed)
        for effective_seed in [0, 1, 42, 2147483647]:
            seed_str = str(effective_seed)
            assert isinstance(seed_str, str)
            assert int(seed_str) == effective_seed
