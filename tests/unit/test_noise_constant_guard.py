"""
I.2 — Noise constant golden-fixture guard.

The noise heuristic in RobustnessAnalyzerV2._apply_auto_scaled_noise() uses
``rng.normal(0, outcome_std)`` — an implicit factor of 1.0× on outcome_std.
This silently affects all downstream percentile and confidence computations.

This test locks the noise factor so that any change to the noise generation
line is caught immediately.
"""

import ast
import inspect
import textwrap

import pytest


_CHANGE_MSG = (
    "Noise constant factor changed — this affects ALL downstream percentile and "
    "confidence computations.  Any change requires re-validation of the full "
    "calibration suite.  Update this golden fixture if the change is intentional."
)


class TestNoiseConstantGuard:
    """Golden-fixture test that locks the noise heuristic."""

    def test_noise_uses_outcome_std_directly(self) -> None:
        """Noise is N(0, outcome_std * noise_multiplier) — default multiplier is 1.0×."""
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        source = inspect.getsource(RobustnessAnalyzerV2._apply_auto_scaled_noise)
        # The noise line should contain: rng.normal(0, outcome_std * noise_multiplier)
        # noise_multiplier defaults to 1.0 (no scaling by default)
        assert "rng.normal(0, outcome_std * noise_multiplier)" in source, (
            f"Expected noise generation to use rng.normal(0, outcome_std * noise_multiplier) "
            f"with default noise_multiplier=1.0.  {_CHANGE_MSG}"
        )

    def test_noise_multiplier_defaults_to_one(self) -> None:
        """The noise_multiplier parameter must default to 1.0 (no scaling)."""
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        source = inspect.getsource(RobustnessAnalyzerV2._apply_auto_scaled_noise)
        assert "noise_multiplier: float = 1.0" in source, (
            f"Expected noise_multiplier default to be 1.0.  {_CHANGE_MSG}"
        )

    def test_noise_variance_doubling_documented(self) -> None:
        """The √2 spread widening rationale must remain documented."""
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        source = inspect.getsource(RobustnessAnalyzerV2._apply_auto_scaled_noise)
        assert "2·var(X)" in source or "2*var(X)" in source or "2·var" in source, (
            "The variance-doubling rationale comment is missing.  "
            "This documents why the noise factor is 1.0x."
        )
