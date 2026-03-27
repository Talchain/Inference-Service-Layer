"""
Seedable PRNG utilities for reproducible inference.

Provides deterministic random number generation that is critical for
reproducible Monte Carlo sampling in robustness analysis.

IMPORTANT: Do NOT use random.random() or np.random.random() directly
in inference code. Always use SeededRNG for reproducibility.
"""

import hashlib
import json
import os
from typing import Any, List, Optional, TYPE_CHECKING, Union, overload

import numpy as np

if TYPE_CHECKING:
    from src.models.robustness_v2 import GraphV2

# Seed hash version controls which fields are included in the graph seed hash.
# Version 1: original — omits edge_type (backward compat).
# Version 2: includes edge_type so directed vs bidirected edges produce different seeds.
# Override via env var SEED_HASH_VERSION for backward compat testing; default is 2.
SEED_HASH_VERSION: int = int(os.environ.get("SEED_HASH_VERSION", "2"))


class SeededRNG:
    """
    Deterministic PRNG for reproducible inference.

    Uses NumPy's PCG64 generator which provides:
    - High-quality randomness
    - Reproducibility from seed
    - Good statistical properties for Monte Carlo

    Example:
        >>> rng = SeededRNG(42)
        >>> rng.random()  # Always returns same value for seed 42
        0.7739560485559633
        >>> rng.normal(0, 1)  # Deterministic normal sample
        -0.4380743093895612
    """

    def __init__(self, seed: int):
        """
        Initialize PRNG with seed.

        Args:
            seed: Integer seed for reproducibility
        """
        self._seed = seed
        self._rng = np.random.Generator(np.random.PCG64(seed))

    @property
    def seed(self) -> int:
        """Return the seed used to initialize this RNG."""
        return self._seed

    def random(self) -> float:
        """
        Generate uniform random float in [0, 1).

        Returns:
            Random float uniformly distributed in [0, 1)
        """
        return float(self._rng.random())

    def uniform(self, low: float, high: float) -> float:
        """
        Generate uniform random float in [low, high).

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)

        Returns:
            Random float uniformly distributed in [low, high)
        """
        return float(self._rng.uniform(low, high))

    def normal(self, mean: float, std: float) -> float:
        """
        Generate normal distribution sample.

        Args:
            mean: Mean of distribution
            std: Standard deviation of distribution

        Returns:
            Random float from Normal(mean, std)
        """
        return float(self._rng.normal(mean, std))

    def bernoulli(self, p: float) -> bool:
        """
        Bernoulli trial with probability p.

        Args:
            p: Probability of success (True)

        Returns:
            True with probability p, False otherwise
        """
        return self._rng.random() < p

    def choice(self, items: List, size: Optional[int] = None, replace: bool = True) -> Any:
        """
        Random choice from items.

        Args:
            items: List to choose from
            size: Number of items to choose (None = single item)
            replace: Whether to sample with replacement

        Returns:
            Single item if size is None, else array of items
        """
        return self._rng.choice(items, size=size, replace=replace)

    def shuffle(self, items: List) -> List:
        """
        Return shuffled copy of items.

        Args:
            items: List to shuffle

        Returns:
            New shuffled list (original unchanged)
        """
        result = list(items)
        self._rng.shuffle(result)
        return result

    def integers(self, low: int, high: int) -> int:
        """
        Generate random integer in [low, high).

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)

        Returns:
            Random integer in [low, high)
        """
        return int(self._rng.integers(low, high))

    def beta(self, a: float, b: float, size: Optional[int] = None) -> Any:
        """
        Generate beta distribution sample(s).

        Args:
            a: Alpha parameter (shape)
            b: Beta parameter (shape)
            size: Number of samples (None = single value)

        Returns:
            Single float or array of floats from Beta(a, b)
        """
        result = self._rng.beta(a, b, size)
        if size is None:
            return float(result)
        return result

    def truncated_normal(
        self,
        mean: float,
        std: float,
        lo: float = -1.0,
        hi: float = 1.0,
        max_attempts: int = 100,
    ) -> float:
        """
        Rejection-sample from Normal(mean, std) truncated to [lo, hi].

        Draws from Normal(mean, std) and rejects samples outside [lo, hi].
        After max_attempts without a valid sample, falls back to
        np.clip(mean, lo, hi) — never returns an out-of-bounds value.

        Args:
            mean: Mean of the underlying normal distribution
            std: Standard deviation of the underlying normal distribution
            lo: Lower bound (inclusive)
            hi: Upper bound (inclusive)
            max_attempts: Maximum rejection-sampling attempts before fallback

        Returns:
            Float in [lo, hi] sampled from the truncated distribution
        """
        for _ in range(max_attempts):
            sample = self._rng.normal(mean, std)
            if lo <= sample <= hi:
                return float(sample)
        return float(np.clip(mean, lo, hi))

    def normal_array(self, mean: float, std: float, size: int) -> np.ndarray:
        """
        Generate array of normal distribution samples.

        Args:
            mean: Mean of distribution
            std: Standard deviation of distribution
            size: Number of samples

        Returns:
            NumPy array of samples from Normal(mean, std)
        """
        return self._rng.normal(mean, std, size)

    def uniform_array(self, low: float, high: float, size: int) -> np.ndarray:
        """
        Generate array of uniform distribution samples.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            size: Number of samples

        Returns:
            NumPy array of samples from Uniform(low, high)
        """
        return self._rng.uniform(low, high, size)

    def beta_array(self, a: float, b: float, size: int) -> np.ndarray:
        """
        Generate array of beta distribution samples.

        Args:
            a: Alpha parameter (shape)
            b: Beta parameter (shape)
            size: Number of samples

        Returns:
            NumPy array of samples from Beta(a, b)
        """
        return self._rng.beta(a, b, size)

    @overload
    def spawn(self) -> "SeededRNG":
        ...

    @overload
    def spawn(self, n: int) -> Union["SeededRNG", List["SeededRNG"]]:
        ...

    def spawn(self, n: int = 1) -> Union["SeededRNG", List["SeededRNG"]]:
        """
        Create new independent RNG(s) from this one.

        Useful for parallel operations that need separate RNG streams.

        Args:
            n: Number of child RNGs to create (default 1)

        Returns:
            Single SeededRNG if n=1, else list of SeededRNG instances
        """
        if n == 1:
            new_seed = self.integers(0, 2**31)
            return SeededRNG(new_seed)
        seeds = self._rng.integers(0, 2**31, n)
        return [SeededRNG(int(s)) for s in seeds]


def compute_seed_from_graph(graph: "GraphV2", *, version: Optional[int] = None) -> int:
    """
    Compute deterministic seed from graph structure.

    Creates a canonical representation of the graph and hashes it
    to produce a reproducible seed. The same graph structure will
    always produce the same seed.

    IMPORTANT: Arrays must be sorted to ensure deterministic output.

    Args:
        graph: GraphV2 instance
        version: Seed hash version override.  None uses module-level
            SEED_HASH_VERSION (default 2).
            - Version 1: omits edge_type (original behaviour).
            - Version 2: includes edge_type so directed vs bidirected
              edges produce different seeds.

    Returns:
        32-bit unsigned integer seed

    Example:
        >>> graph = GraphV2(nodes=[...], edges=[...])
        >>> seed = compute_seed_from_graph(graph)
        >>> rng = SeededRNG(seed)
        >>> # Now rng will produce same sequence for same graph
    """
    effective_version = version if version is not None else SEED_HASH_VERSION

    # Sort nodes by id for deterministic ordering
    sorted_nodes = sorted(
        [{"id": n.id, "kind": n.kind} for n in graph.nodes], key=lambda x: x["id"]
    )

    # Build per-edge canonical dicts.
    # Version 1: omits edge_type (backward compat).
    # Version 2: includes edge_type so directed vs bidirected produce different seeds.
    edge_dicts = []
    for e in graph.edges:
        d = {
            "from": e.from_,
            "to": e.to,
            "exists_probability": e.exists_probability,
            "strength": {"mean": e.strength.mean, "std": e.strength.std},
        }
        if effective_version >= 2:
            # Include edge_type; default to "directed" when absent for determinism.
            d["edge_type"] = e.edge_type or "directed"
        edge_dicts.append(d)

    sorted_edges = sorted(edge_dicts, key=lambda x: (x["from"], x["to"]))

    # Create canonical JSON representation
    canonical = json.dumps(
        {"nodes": sorted_nodes, "edges": sorted_edges}, sort_keys=True, ensure_ascii=True
    )

    # Hash and convert to 32-bit unsigned integer
    hash_bytes = hashlib.sha256(canonical.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder="big", signed=False)

    return seed


def compute_seed_from_dict(data: dict) -> int:
    """
    Compute deterministic seed from arbitrary dictionary.

    Args:
        data: Dictionary to hash

    Returns:
        32-bit unsigned integer seed
    """
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
    hash_bytes = hashlib.sha256(canonical.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder="big", signed=False)
    return seed
