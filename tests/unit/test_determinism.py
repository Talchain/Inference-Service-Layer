"""
Unit tests for determinism utilities.
"""

import pytest

from src.utils.determinism import (
    canonical_hash,
    make_deterministic,
    seed_from_input,
    verify_determinism,
)


def test_canonical_hash_deterministic():
    """Test that canonical_hash produces same output for same input."""
    data1 = {"b": 2, "a": 1, "c": 3}
    data2 = {"a": 1, "c": 3, "b": 2}  # Different order

    hash1 = canonical_hash(data1)
    hash2 = canonical_hash(data2)

    assert hash1 == hash2, "Hashes should be identical regardless of key order"


def test_canonical_hash_different_inputs():
    """Test that different inputs produce different hashes."""
    data1 = {"a": 1, "b": 2}
    data2 = {"a": 1, "b": 3}

    hash1 = canonical_hash(data1)
    hash2 = canonical_hash(data2)

    assert hash1 != hash2, "Different inputs should produce different hashes"


def test_seed_from_input_deterministic():
    """Test that seed generation is deterministic."""
    data = {"treatment": "Price", "outcome": "Revenue"}

    seed1 = seed_from_input(data)
    seed2 = seed_from_input(data)

    assert seed1 == seed2, "Same input should produce same seed"
    assert isinstance(seed1, int), "Seed should be an integer"


def test_make_deterministic():
    """Test that make_deterministic returns a per-request RNG."""
    from src.utils.rng import SeededRNG

    data = {"test": "data"}

    rng1 = make_deterministic(data)
    assert isinstance(rng1, SeededRNG), "make_deterministic should return SeededRNG"

    # Generate some random numbers using the RNG
    rand1 = rng1.random()
    np_rand1 = rng1.normal(0, 1)

    # Create new RNG with same data
    rng2 = make_deterministic(data)

    # Should get same random numbers from new RNG with same seed
    rand2 = rng2.random()
    np_rand2 = rng2.normal(0, 1)

    assert rng1.seed == rng2.seed, "Same input should produce same seed"
    assert rand1 == rand2, "Random numbers should be reproducible"
    assert np_rand1 == np_rand2, "NumPy random numbers should be reproducible"


def test_verify_determinism():
    """Test determinism verification."""

    def deterministic_func(data):
        rng = make_deterministic(data)
        return rng.random()

    assert verify_determinism(deterministic_func, {"test": 1}), \
        "Deterministic function should pass verification"
