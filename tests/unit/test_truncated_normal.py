"""Tests for SeededRNG.truncated_normal() — rejection-sampled truncated normal."""

import numpy as np
import pytest

from src.utils.rng import SeededRNG


class TestTruncatedNormalBounds:
    """All samples must lie within [lo, hi]."""

    def test_all_samples_within_bounds(self) -> None:
        rng = SeededRNG(42)
        samples = [rng.truncated_normal(0.0, 0.5) for _ in range(10_000)]
        assert all(-1.0 <= s <= 1.0 for s in samples)

    def test_all_samples_within_bounds_high_mean(self) -> None:
        """Edge near boundary: mean=0.9, std=0.3 — many raw draws exceed +1."""
        rng = SeededRNG(99)
        samples = [rng.truncated_normal(0.9, 0.3) for _ in range(10_000)]
        assert all(-1.0 <= s <= 1.0 for s in samples)

    def test_all_samples_within_bounds_negative_mean(self) -> None:
        rng = SeededRNG(77)
        samples = [rng.truncated_normal(-0.8, 0.4) for _ in range(10_000)]
        assert all(-1.0 <= s <= 1.0 for s in samples)

    def test_custom_bounds(self) -> None:
        rng = SeededRNG(42)
        samples = [rng.truncated_normal(0.0, 1.0, lo=-0.5, hi=0.5) for _ in range(5_000)]
        assert all(-0.5 <= s <= 0.5 for s in samples)


class TestTruncatedNormalMeanPreservation:
    """Truncated mean should NOT be biased toward zero."""

    def test_mean_near_boundary_not_biased_to_zero(self) -> None:
        """mean=0.9, std=0.3: truncated mean should stay well above zero.

        Theoretical truncated normal mean for N(0.9, 0.3) on [-1, 1] is ~0.72.
        This is lower than the raw mean because truncation removes the right
        tail, but it should NOT collapse toward zero.  Clipped normal would
        have a spike at +1 pushing the mean *higher* — truncation correctly
        avoids this artefact.
        """
        rng = SeededRNG(42)
        samples = [rng.truncated_normal(0.9, 0.3) for _ in range(10_000)]
        sample_mean = np.mean(samples)
        # Theoretical truncated mean ≈ 0.72; allow [0.65, 0.80]
        assert 0.65 < sample_mean < 0.80, f"Sample mean {sample_mean:.4f} outside expected range"

    def test_symmetric_distribution_mean_near_zero(self) -> None:
        rng = SeededRNG(42)
        samples = [rng.truncated_normal(0.0, 0.3) for _ in range(10_000)]
        sample_mean = np.mean(samples)
        assert abs(sample_mean) < 0.02, f"Symmetric truncated mean {sample_mean:.4f} should be ~0"


class TestTruncatedNormalFallback:
    """When bounds are unreachable, fallback to clamped mean."""

    def test_unreachable_bounds_returns_clamped_mean(self) -> None:
        """mean=10, std=0.001, bounds [-1,1] — virtually impossible to land
        in-bounds.  Fallback should return np.clip(10, -1, 1) = 1.0."""
        rng = SeededRNG(42)
        result = rng.truncated_normal(10.0, 0.001, lo=-1.0, hi=1.0)
        assert result == 1.0

    def test_unreachable_negative_returns_clamped_mean(self) -> None:
        rng = SeededRNG(42)
        result = rng.truncated_normal(-10.0, 0.001, lo=-1.0, hi=1.0)
        assert result == -1.0

    def test_fallback_within_bounds(self) -> None:
        """mean=0.5, std=0.001, bounds [0, 0.01] — mean is in-range so
        fallback should return 0.01 (clamped)... actually 0.5 > 0.01,
        so clip(0.5, 0, 0.01) = 0.01."""
        rng = SeededRNG(42)
        result = rng.truncated_normal(0.5, 0.001, lo=0.0, hi=0.01)
        assert result == 0.01


class TestTruncatedNormalHistogram:
    """Verify no boundary spike artefacts."""

    def test_no_boundary_spike_at_plus_one(self) -> None:
        """With clipping, N(0.9, 0.3) would have a spike of exact 1.0 values
        (all samples > 1.0 get clipped to exactly 1.0).  With truncation,
        no samples should be exactly at the boundary — the distribution is
        continuous up to the boundary."""
        rng = SeededRNG(42)
        samples = np.array([rng.truncated_normal(0.9, 0.3) for _ in range(10_000)])
        # Count samples at *exactly* 1.0 — truncation should produce zero
        exact_boundary = np.sum(samples == 1.0)
        assert exact_boundary == 0, (
            f"{exact_boundary} samples at exactly 1.0 — "
            f"truncation should not produce boundary spikes"
        )
        # Also verify all samples are in bounds
        assert np.all(samples >= -1.0) and np.all(samples <= 1.0)

    def test_no_boundary_spike_at_minus_one(self) -> None:
        """Symmetric check at -1.0 boundary."""
        rng = SeededRNG(42)
        samples = np.array([rng.truncated_normal(-0.9, 0.3) for _ in range(10_000)])
        exact_boundary = np.sum(samples == -1.0)
        assert exact_boundary == 0, (
            f"{exact_boundary} samples at exactly -1.0 — "
            f"truncation should not produce boundary spikes"
        )
        assert np.all(samples >= -1.0) and np.all(samples <= 1.0)


class TestTruncatedNormalDeterminism:
    """Same seed must produce identical sequences."""

    def test_deterministic_with_same_seed(self) -> None:
        samples_a = [SeededRNG(42).truncated_normal(0.5, 0.2) for _ in range(1)]
        # Re-create with same seed
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)
        seq1 = [rng1.truncated_normal(0.5, 0.2) for _ in range(100)]
        seq2 = [rng2.truncated_normal(0.5, 0.2) for _ in range(100)]
        assert seq1 == seq2

    def test_different_seeds_differ(self) -> None:
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(43)
        seq1 = [rng1.truncated_normal(0.5, 0.2) for _ in range(50)]
        seq2 = [rng2.truncated_normal(0.5, 0.2) for _ in range(50)]
        assert seq1 != seq2
