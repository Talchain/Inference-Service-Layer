"""
Seedable PRNG utilities for reproducible inference.

Provides deterministic random number generation that is critical for
reproducible Monte Carlo sampling in robustness analysis.

IMPORTANT: Do NOT use random.random() or np.random.random() directly
in inference code. Always use SeededRNG for reproducibility.
"""

import hashlib
import json
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.robustness_v2 import GraphV2


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

    def choice(self, items: List, size: Optional[int] = None, replace: bool = True):
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

    def spawn(self) -> "SeededRNG":
        """
        Create a new independent RNG from this one.

        Useful for parallel operations that need separate RNG streams.

        Returns:
            New SeededRNG instance with independent state
        """
        new_seed = self.integers(0, 2**31)
        return SeededRNG(new_seed)


def compute_seed_from_graph(graph: "GraphV2") -> int:
    """
    Compute deterministic seed from graph structure.

    Creates a canonical representation of the graph and hashes it
    to produce a reproducible seed. The same graph structure will
    always produce the same seed.

    IMPORTANT: Arrays must be sorted to ensure deterministic output.

    Args:
        graph: GraphV2 instance

    Returns:
        32-bit unsigned integer seed

    Example:
        >>> graph = GraphV2(nodes=[...], edges=[...])
        >>> seed = compute_seed_from_graph(graph)
        >>> rng = SeededRNG(seed)
        >>> # Now rng will produce same sequence for same graph
    """
    # Sort nodes by id for deterministic ordering
    sorted_nodes = sorted(
        [{"id": n.id, "kind": n.kind} for n in graph.nodes],
        key=lambda x: x["id"]
    )

    # Sort edges by (from, to) for deterministic ordering
    sorted_edges = sorted(
        [
            {
                "from": e.from_,
                "to": e.to,
                "exists_probability": e.exists_probability,
                "strength": {
                    "mean": e.strength.mean,
                    "std": e.strength.std
                }
            }
            for e in graph.edges
        ],
        key=lambda x: (x["from"], x["to"])
    )

    # Create canonical JSON representation
    canonical = json.dumps(
        {"nodes": sorted_nodes, "edges": sorted_edges},
        sort_keys=True,
        ensure_ascii=True
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
