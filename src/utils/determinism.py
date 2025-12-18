"""
Determinism utilities for reproducible computations.

Ensures identical inputs always produce identical outputs by providing
deterministic hashing and seeding mechanisms.

IMPORTANT: For concurrent request safety, always use make_deterministic()
which returns a per-request SeededRNG. Do NOT use set_global_seed() in
production code as it affects global state.
"""

import hashlib
import json
import logging
import random
from typing import Any, Dict, Union

import numpy as np

from src.utils.rng import SeededRNG

logger = logging.getLogger(__name__)


def canonical_hash(data: Union[Dict[str, Any], list, str, int, float]) -> str:
    """
    Generate deterministic hash for any input data.

    Ensures identical inputs always produce identical hashes by:
    - Sorting dictionary keys recursively
    - Using consistent JSON serialization
    - SHA256 for cryptographic strength

    Args:
        data: Input data to hash (dict, list, primitive types)

    Returns:
        str: Hexadecimal hash string

    Example:
        >>> canonical_hash({"b": 2, "a": 1})
        'a1b2c3...'
        >>> canonical_hash({"a": 1, "b": 2})  # Same hash despite different order
        'a1b2c3...'
    """
    # Convert to JSON with sorted keys for consistency
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
    # Generate SHA256 hash
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seed_from_input(data: Union[Dict[str, Any], list, str]) -> int:
    """
    Generate deterministic seed from input for random number generation.

    Uses the canonical hash to create a deterministic integer seed
    that can be used with numpy.random.seed() and random.seed().

    Args:
        data: Input data to generate seed from

    Returns:
        int: Deterministic seed value (32-bit integer)

    Example:
        >>> seed = seed_from_input({"treatment": "Price", "outcome": "Revenue"})
        >>> np.random.seed(seed)
        >>> # Now random operations are reproducible
    """
    hash_value = canonical_hash(data)
    # Use first 8 hex digits (32 bits) as seed
    return int(hash_value[:8], 16)


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for all libraries.

    DEPRECATED: This function sets global state and is NOT safe for concurrent
    requests. Use make_deterministic() which returns a per-request SeededRNG.

    Args:
        seed: Integer seed value
    """
    logger.warning(
        "set_global_seed() is deprecated for production use - "
        "use make_deterministic() which returns per-request SeededRNG"
    )
    random.seed(seed)
    np.random.seed(seed)


def make_deterministic(request_data: Union[Dict[str, Any], list, str]) -> SeededRNG:
    """
    Create per-request RNG for deterministic, thread-safe random operations.

    Returns a SeededRNG instance that is isolated to this request, preventing
    cross-request interference in concurrent environments.

    Args:
        request_data: Input request data to use for seeding

    Returns:
        SeededRNG: Per-request random number generator

    Example:
        >>> rng = make_deterministic({"dag": {...}, "treatment": "X"})
        >>> value = rng.normal(0, 1)  # Thread-safe, deterministic
        >>> samples = rng.normal_array(0, 1, 100)  # Array of samples
    """
    seed = seed_from_input(request_data)
    return SeededRNG(seed)


def verify_determinism(
    func: callable,
    input_data: Dict[str, Any],
    num_trials: int = 3,
) -> bool:
    """
    Verify that a function produces deterministic outputs.

    Runs the function multiple times with the same input and checks
    that outputs are identical.

    Args:
        func: Function to test
        input_data: Input data to pass to function
        num_trials: Number of times to run (default 3)

    Returns:
        bool: True if outputs are identical across all trials

    Example:
        >>> def my_analysis(data):
        ...     make_deterministic(data)
        ...     return np.random.random()
        >>> verify_determinism(my_analysis, {"test": 1})
        True
    """
    results = []
    for _ in range(num_trials):
        result = func(input_data)
        result_hash = canonical_hash(result if isinstance(result, dict) else str(result))
        results.append(result_hash)

    # All hashes should be identical
    return len(set(results)) == 1
