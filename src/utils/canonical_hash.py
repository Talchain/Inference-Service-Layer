"""
Canonical JSON hashing utility for deterministic response fingerprinting.

Provides cross-language compatible hashing for response bodies using:
- Sorted keys (alphabetical)
- No extra whitespace (compact JSON)
- SHA-256 hash algorithm
- UTF-8 encoding

This enables response verification across different services and languages.
"""

import hashlib
import json
from typing import Any, Dict, Union


def canonical_json_hash(obj: Union[Dict[str, Any], str, bytes]) -> str:
    """
    Compute SHA-256 hash of canonical JSON representation.

    The canonical form ensures:
    - Keys are sorted alphabetically at all nesting levels
    - No extra whitespace (compact: separators=(',', ':'))
    - UTF-8 encoding
    - Consistent output across Python versions

    Args:
        obj: Dictionary to hash, or pre-serialized JSON string/bytes

    Returns:
        Lowercase hex SHA-256 hash (64 characters)

    Examples:
        >>> canonical_json_hash({"a": 1, "b": 2})
        '43258cff783fe7036d8a43033f830adfc60ec037...'

        >>> # Key order doesn't matter - same hash
        >>> canonical_json_hash({"b": 2, "a": 1})
        '43258cff783fe7036d8a43033f830adfc60ec037...'

    Test Vectors (for cross-language verification):
        {"a": 1, "b": 2}  → canonical: '{"a":1,"b":2}'
                          → SHA-256: '43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777'

        {"b": 2, "a": 1}  → canonical: '{"a":1,"b":2}' (sorted)
                          → SHA-256: '43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777' (same)

        {"nested": {"z": 26, "a": 1}, "top": "value"}
                          → canonical: '{"nested":{"a":1,"z":26},"top":"value"}'
                          → SHA-256: '364c86da6b8b2ac24d66ae42e5f710f052d830a519c681f3af90f732c24afeb5'
    """
    if isinstance(obj, bytes):
        # Already bytes, hash directly (assumes canonical JSON bytes)
        canonical_bytes = obj
    elif isinstance(obj, str):
        # Pre-serialized JSON string
        canonical_bytes = obj.encode("utf-8")
    else:
        # Dictionary - serialize to canonical JSON
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        canonical_bytes = canonical.encode("utf-8")

    return hashlib.sha256(canonical_bytes).hexdigest()


def canonical_json_string(obj: Dict[str, Any]) -> str:
    """
    Convert dictionary to canonical JSON string.

    Useful when you need the canonical form for logging or transmission
    before hashing.

    Args:
        obj: Dictionary to serialize

    Returns:
        Canonical JSON string (sorted keys, compact)

    Example:
        >>> canonical_json_string({"b": 2, "a": 1})
        '{"a":1,"b":2}'
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# Aliases for cross-service compatibility (matches CEE API)
def canonical_json(obj: Dict[str, Any]) -> str:
    """Alias for canonical_json_string() - matches CEE API."""
    return canonical_json_string(obj)


def compute_payload_hash(obj: Dict[str, Any]) -> str:
    """
    Compute short payload hash for cross-service verification.

    Returns first 12 hex characters of SHA-256 hash of canonical JSON.
    This matches CEE's implementation for header verification.

    Args:
        obj: Dictionary to hash

    Returns:
        12-character hex hash prefix

    Example:
        >>> compute_payload_hash({"a": 1, "b": 2})
        '43258cff783f'
    """
    return canonical_json_hash(obj)[:12]


# Test vectors for cross-language verification
# These MUST produce identical hashes in all ISL-compatible implementations
TEST_VECTORS = [
    {
        "input": {"a": 1, "b": 2},
        "canonical": '{"a":1,"b":2}',
        "hash": "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777",
    },
    {
        "input": {"b": 2, "a": 1},  # Different key order, same hash
        "canonical": '{"a":1,"b":2}',
        "hash": "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777",
    },
    {
        "input": {"nested": {"z": 26, "a": 1}, "top": "value"},
        "canonical": '{"nested":{"a":1,"z":26},"top":"value"}',
        "hash": "364c86da6b8b2ac24d66ae42e5f710f052d830a519c681f3af90f732c24afeb5",
    },
    {
        # Unicode is escaped by default in Python's json.dumps (ensure_ascii=True)
        "input": {"unicode": "café", "emoji": "🚀"},
        "canonical": '{"emoji":"\\ud83d\\ude80","unicode":"caf\\u00e9"}',
        "hash": "a60b496b50428850cf1d441bfa50cc3f267b1944ddfe128bdbae5e4ce6bf081c",
    },
    {
        "input": {"empty": {}, "null": None, "list": [1, 2, 3]},
        "canonical": '{"empty":{},"list":[1,2,3],"null":null}',
        "hash": "c5faec7dedb631c6ed1d2d959558196001a15e106a75673cca487028a03f5a6d",
    },
]


def verify_test_vectors() -> bool:
    """
    Verify all test vectors produce expected hashes.

    Returns:
        True if all vectors pass, raises AssertionError otherwise
    """
    for i, vector in enumerate(TEST_VECTORS):
        canonical = canonical_json_string(vector["input"])
        computed_hash = canonical_json_hash(vector["input"])

        # Verify canonical form
        if canonical != vector["canonical"]:
            raise AssertionError(
                f"Test vector {i} canonical mismatch:\n"
                f"  Expected: {vector['canonical']}\n"
                f"  Got: {canonical}"
            )

        # Verify hash
        if computed_hash != vector["hash"]:
            raise AssertionError(
                f"Test vector {i} hash mismatch:\n"
                f"  Expected: {vector['hash']}\n"
                f"  Got: {computed_hash}"
            )

    return True
