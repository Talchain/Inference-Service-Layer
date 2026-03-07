"""
I.4 — edge_type in seed hash (versioned).

Bidirected vs directed edges produce identical seeds under version 1
because edge_type is omitted.  Version 2 includes edge_type so
they produce different seeds.

Golden-fixture tests lock backward compatibility and verify the new
behaviour.
"""

import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    NodeV2,
    StrengthDistribution,
)
from src.utils.rng import SEED_HASH_VERSION, compute_seed_from_graph


def _make_graph(edge_type: str = "directed") -> GraphV2:
    """Build a minimal graph with a single edge of the given type."""
    return GraphV2(
        nodes=[
            NodeV2(id="a", kind="factor", label="A"),
            NodeV2(id="b", kind="outcome", label="B"),
        ],
        edges=[
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=0.9,
                strength=StrengthDistribution(mean=0.5, std=0.1),
                edge_type=edge_type,
            ),
        ],
    )


def _make_graph_no_edge_type() -> GraphV2:
    """Build a graph where edge_type is None (absent)."""
    return GraphV2(
        nodes=[
            NodeV2(id="a", kind="factor", label="A"),
            NodeV2(id="b", kind="outcome", label="B"),
        ],
        edges=[
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=0.9,
                strength=StrengthDistribution(mean=0.5, std=0.1),
                # edge_type intentionally omitted
            ),
        ],
    )


class TestSeedHashVersion1BackwardCompat:
    """Version 1: edge_type omitted — existing golden fixtures pass."""

    def test_directed_and_bidirected_produce_same_seed_v1(self) -> None:
        """V1: changing edge_type does NOT change the seed (backward compat)."""
        g_directed = _make_graph("directed")
        g_bidirected = _make_graph("bidirected")
        seed_dir = compute_seed_from_graph(g_directed, version=1)
        seed_bid = compute_seed_from_graph(g_bidirected, version=1)
        assert seed_dir == seed_bid, (
            f"Version 1 should ignore edge_type: directed={seed_dir}, bidirected={seed_bid}"
        )

    def test_v1_deterministic(self) -> None:
        """V1: same graph always produces the same seed."""
        g = _make_graph("directed")
        assert compute_seed_from_graph(g, version=1) == compute_seed_from_graph(g, version=1)


class TestSeedHashVersion2:
    """Version 2: edge_type included — bidirected ≠ directed."""

    def test_directed_and_bidirected_produce_different_seed_v2(self) -> None:
        """V2: changing edge_type changes the seed."""
        g_directed = _make_graph("directed")
        g_bidirected = _make_graph("bidirected")
        seed_dir = compute_seed_from_graph(g_directed, version=2)
        seed_bid = compute_seed_from_graph(g_bidirected, version=2)
        assert seed_dir != seed_bid, (
            f"Version 2 should differentiate edge_type: directed={seed_dir}, bidirected={seed_bid}"
        )

    def test_v2_differs_from_v1_for_directed_graphs(self) -> None:
        """V2 always differs from V1 — edge_type is always part of the V2 hash.

        Design decision: V2 includes edge_type in the canonical JSON for ALL
        edges (defaulting to "directed" when absent).  This means V1 and V2
        produce different seeds even for all-directed graphs.  This is
        intentional — version tagging provides clean separation between hash
        generations without conditional logic.
        """
        g = _make_graph("directed")
        seed_v1 = compute_seed_from_graph(g, version=1)
        seed_v2 = compute_seed_from_graph(g, version=2)
        assert seed_v1 != seed_v2, (
            "V2 canonical JSON includes edge_type; seeds must differ from V1"
        )

    def test_v2_absent_edge_type_defaults_to_directed(self) -> None:
        """V2: absent edge_type defaults to 'directed' and matches explicit."""
        g_explicit = _make_graph("directed")
        g_none = _make_graph_no_edge_type()
        seed_explicit = compute_seed_from_graph(g_explicit, version=2)
        seed_none = compute_seed_from_graph(g_none, version=2)
        assert seed_explicit == seed_none, (
            "V2: absent edge_type should default to 'directed' and match explicit"
        )

    def test_v2_deterministic(self) -> None:
        """V2: same graph always produces the same seed."""
        g = _make_graph("bidirected")
        assert compute_seed_from_graph(g, version=2) == compute_seed_from_graph(g, version=2)


class TestDefaultVersion:
    """Default version is 2."""

    def test_default_version_is_2(self) -> None:
        """Module-level SEED_HASH_VERSION defaults to 2."""
        assert SEED_HASH_VERSION == 2

    def test_compute_seed_uses_default_version(self) -> None:
        """compute_seed_from_graph() without version= uses SEED_HASH_VERSION."""
        g = _make_graph("bidirected")
        seed_default = compute_seed_from_graph(g)
        seed_v2 = compute_seed_from_graph(g, version=2)
        assert seed_default == seed_v2


class TestResponseMetadataIncludesVersion:
    """Response metadata includes seed_hash_version."""

    def test_metadata_has_seed_hash_version(self) -> None:
        """ResponseMetadataV2 includes seed_hash_version field."""
        from src.models.robustness_v2 import ResponseMetadataV2

        meta = ResponseMetadataV2(
            isl_version="0.2.0-dev",
            n_samples_used=100,
            seed_used=42,
            execution_time_ms=50,
            config_fingerprint="abc123",
            seed_hash_version=2,
        )
        assert meta.seed_hash_version == 2

    def test_metadata_serialises_seed_hash_version(self) -> None:
        """seed_hash_version appears in JSON serialisation."""
        from src.models.robustness_v2 import ResponseMetadataV2

        meta = ResponseMetadataV2(
            isl_version="0.2.0-dev",
            n_samples_used=100,
            seed_used=42,
            execution_time_ms=50,
            config_fingerprint="abc123",
            seed_hash_version=2,
        )
        data = meta.model_dump()
        assert "seed_hash_version" in data
        assert data["seed_hash_version"] == 2
