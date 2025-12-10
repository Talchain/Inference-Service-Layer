"""
Property-based tests for weight normalization.

Uses Hypothesis to generate random weight vectors and validate mathematical properties.
"""

import pytest
from hypothesis import given, strategies as st, assume
from hypothesis import settings

from src.services.multi_criteria_aggregator import MultiCriteriaAggregator


class TestWeightNormalizationProperties:
    """Property-based tests for weight normalization."""

    @given(
        weights=st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_normalized_weights_sum_to_one(self, weights):
        """
        Property: Normalized weights must always sum to exactly 1.0.

        This is a fundamental mathematical requirement for weights in
        aggregation functions.
        """
        aggregator = MultiCriteriaAggregator()

        # Create weights dict
        weights_dict = {f"criterion_{i}": w for i, w in enumerate(weights)}

        # Normalize
        normalized = aggregator._normalize_weights(weights_dict)

        # Property: Sum should equal 1.0 (within floating point tolerance)
        total = sum(normalized.values())
        assert abs(total - 1.0) < 1e-10, f"Weights sum to {total}, expected 1.0"

    @given(
        weights=st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_normalized_weights_preserve_order(self, weights):
        """
        Property: Normalization preserves relative ordering.

        If weight A > weight B before normalization, then
        normalized_A > normalized_B after normalization.
        """
        assume(len(set(weights)) > 1)  # Need at least 2 distinct weights

        aggregator = MultiCriteriaAggregator()
        weights_dict = {f"criterion_{i}": w for i, w in enumerate(weights)}
        normalized = aggregator._normalize_weights(weights_dict)

        # Check relative ordering is preserved
        for i in range(len(weights)):
            for j in range(len(weights)):
                if weights[i] > weights[j]:
                    assert normalized[f"criterion_{i}"] > normalized[f"criterion_{j}"]
                elif weights[i] < weights[j]:
                    assert normalized[f"criterion_{i}"] < normalized[f"criterion_{j}"]

    @given(
        weights=st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100)
    def test_all_normalized_weights_positive(self, weights):
        """
        Property: All normalized weights must be positive.

        Negative or zero weights would violate aggregation assumptions.
        """
        aggregator = MultiCriteriaAggregator()
        weights_dict = {f"criterion_{i}": w for i, w in enumerate(weights)}
        normalized = aggregator._normalize_weights(weights_dict)

        for weight in normalized.values():
            assert weight > 0, f"Normalized weight {weight} is not positive"
            assert weight <= 1.0, f"Normalized weight {weight} exceeds 1.0"

    @given(
        uniform_value=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        size=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=50)
    def test_uniform_weights_equal_after_normalization(self, uniform_value, size):
        """
        Property: If all weights are equal, normalized weights should be 1/n.

        This tests the edge case of uniform weights.
        """
        aggregator = MultiCriteriaAggregator()
        weights_dict = {f"criterion_{i}": uniform_value for i in range(size)}
        normalized = aggregator._normalize_weights(weights_dict)

        expected = 1.0 / size
        for weight in normalized.values():
            assert abs(weight - expected) < 1e-10, \
                f"Uniform weights not equal: got {weight}, expected {expected}"
