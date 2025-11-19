"""
Determinism utilities for reproducible computations.

Ensures identical inputs always produce identical outputs by providing
deterministic hashing and seeding mechanisms.
"""

import hashlib
import json
import random
from typing import Any, Dict, Union

import numpy as np


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

    Ensures reproducibility across:
    - Python's random module
    - NumPy
    - Any other libraries using these RNGs

    Args:
        seed: Integer seed value

    Example:
        >>> set_global_seed(42)
        >>> # All subsequent random operations are now deterministic
    """
    random.seed(seed)
    np.random.seed(seed)


def make_deterministic(request_data: Union[Dict[str, Any], list, str]) -> int:
    """
    Convenience function to make all random operations deterministic.

    Combines seed generation and global seed setting in one call.

    Args:
        request_data: Input request data to use for seeding

    Returns:
        int: The seed value that was set

    Example:
        >>> seed = make_deterministic({"dag": {...}, "treatment": "X"})
        >>> # Now all random operations are reproducible
    """
    seed = seed_from_input(request_data)
    set_global_seed(seed)
    return seed


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
