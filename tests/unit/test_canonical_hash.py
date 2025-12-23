"""
Unit tests for canonical JSON hashing utility.

Tests cross-language compatible hashing for response verification.
"""

import pytest

from src.utils.canonical_hash import (
    canonical_json_hash,
    canonical_json_string,
    verify_test_vectors,
    TEST_VECTORS,
)


class TestCanonicalJsonHash:
    """Test cases for canonical_json_hash function."""

    def test_basic_hash(self):
        """Test basic dictionary hashing."""
        result = canonical_json_hash({"a": 1, "b": 2})
        assert len(result) == 64  # SHA-256 hex length
        assert result == "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"

    def test_key_order_independence(self):
        """Test that key order doesn't affect hash."""
        hash1 = canonical_json_hash({"a": 1, "b": 2})
        hash2 = canonical_json_hash({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_nested_dict_sorting(self):
        """Test that nested dictionaries are also sorted."""
        hash1 = canonical_json_hash({"outer": {"z": 1, "a": 2}})
        hash2 = canonical_json_hash({"outer": {"a": 2, "z": 1}})
        assert hash1 == hash2

    def test_deeply_nested_sorting(self):
        """Test deep nesting preserves sorting."""
        obj1 = {"l1": {"l2": {"l3": {"z": 1, "a": 2}}}}
        obj2 = {"l1": {"l2": {"l3": {"a": 2, "z": 1}}}}
        assert canonical_json_hash(obj1) == canonical_json_hash(obj2)

    def test_different_values_different_hash(self):
        """Test that different values produce different hashes."""
        hash1 = canonical_json_hash({"a": 1})
        hash2 = canonical_json_hash({"a": 2})
        assert hash1 != hash2

    def test_string_input(self):
        """Test hashing pre-serialized JSON string."""
        json_str = '{"a":1,"b":2}'
        result = canonical_json_hash(json_str)
        assert len(result) == 64

    def test_bytes_input(self):
        """Test hashing JSON bytes directly."""
        json_bytes = b'{"a":1,"b":2}'
        result = canonical_json_hash(json_bytes)
        assert len(result) == 64

    def test_unicode_handling(self):
        """Test Unicode characters are handled correctly."""
        obj = {"unicode": "caf\u00e9", "emoji": "\U0001F680"}  # cafe with accent, rocket
        result = canonical_json_hash(obj)
        assert len(result) == 64

    def test_null_values(self):
        """Test None/null values are handled."""
        result = canonical_json_hash({"key": None})
        assert len(result) == 64
        # Should contain "null" in canonical form
        assert "null" in canonical_json_string({"key": None})

    def test_empty_dict(self):
        """Test empty dictionary."""
        result = canonical_json_hash({})
        assert len(result) == 64
        assert canonical_json_string({}) == "{}"

    def test_list_values(self):
        """Test lists preserve order (lists are not sorted)."""
        obj1 = {"items": [1, 2, 3]}
        obj2 = {"items": [3, 2, 1]}
        # List order matters - different hashes
        assert canonical_json_hash(obj1) != canonical_json_hash(obj2)

    def test_numeric_types(self):
        """Test integer vs float handling."""
        # Python's json.dumps treats 1 and 1.0 differently
        # 1 becomes "1" and 1.0 becomes "1.0"
        hash_int = canonical_json_hash({"val": 1})
        hash_float = canonical_json_hash({"val": 1.0})
        # Note: Different types produce different hashes
        assert hash_int != hash_float

    def test_boolean_values(self):
        """Test boolean serialization."""
        obj = {"true": True, "false": False}
        canonical = canonical_json_string(obj)
        assert "true" in canonical
        assert "false" in canonical


class TestCanonicalJsonString:
    """Test cases for canonical_json_string function."""

    def test_sorted_keys(self):
        """Test keys are sorted alphabetically."""
        result = canonical_json_string({"z": 1, "a": 2, "m": 3})
        assert result == '{"a":2,"m":3,"z":1}'

    def test_compact_format(self):
        """Test no extra whitespace."""
        result = canonical_json_string({"key": "value"})
        assert " " not in result.replace('"key"', "").replace('"value"', "")

    def test_nested_sorted(self):
        """Test nested objects are also sorted."""
        result = canonical_json_string({"outer": {"z": 1, "a": 2}})
        assert result == '{"outer":{"a":2,"z":1}}'


class TestTestVectors:
    """Test cases for cross-language verification vectors."""

    def test_verify_all_vectors(self):
        """Test all test vectors pass verification."""
        # This should not raise
        assert verify_test_vectors() is True

    @pytest.mark.parametrize("vector_index", range(len(TEST_VECTORS)))
    def test_individual_vector(self, vector_index):
        """Test each vector individually."""
        vector = TEST_VECTORS[vector_index]

        # Verify canonical form
        canonical = canonical_json_string(vector["input"])
        assert canonical == vector["canonical"], (
            f"Vector {vector_index}: canonical mismatch\n"
            f"Expected: {vector['canonical']}\n"
            f"Got: {canonical}"
        )

        # Verify hash
        computed_hash = canonical_json_hash(vector["input"])
        assert computed_hash == vector["hash"], (
            f"Vector {vector_index}: hash mismatch\n"
            f"Expected: {vector['hash']}\n"
            f"Got: {computed_hash}"
        )

    def test_first_two_vectors_same_hash(self):
        """Test that first two vectors (different key order) produce same hash."""
        assert TEST_VECTORS[0]["hash"] == TEST_VECTORS[1]["hash"]


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_special_characters(self):
        """Test special JSON characters are escaped."""
        obj = {"quote": '"', "backslash": "\\", "newline": "\n"}
        result = canonical_json_hash(obj)
        assert len(result) == 64

    def test_very_long_string(self):
        """Test handling of long strings."""
        long_string = "x" * 100000
        result = canonical_json_hash({"long": long_string})
        assert len(result) == 64

    def test_many_keys(self):
        """Test dictionary with many keys."""
        obj = {f"key_{i}": i for i in range(1000)}
        result = canonical_json_hash(obj)
        assert len(result) == 64

    def test_deep_nesting(self):
        """Test deeply nested structure."""
        obj = {"level": 0}
        current = obj
        for i in range(1, 100):
            current["nested"] = {"level": i}
            current = current["nested"]

        result = canonical_json_hash(obj)
        assert len(result) == 64


class TestDeterminism:
    """Tests for hash determinism and reproducibility."""

    def test_repeated_calls_same_result(self):
        """Test same input always produces same hash."""
        obj = {"test": "determinism", "value": 42}
        results = [canonical_json_hash(obj) for _ in range(100)]
        assert len(set(results)) == 1  # All same

    def test_dictionary_reconstruction(self):
        """Test hash is same after dictionary reconstruction."""
        original = {"a": 1, "b": {"c": 2}}
        reconstructed = dict(original)
        assert canonical_json_hash(original) == canonical_json_hash(reconstructed)
