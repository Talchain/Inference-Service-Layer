"""VOI F-1 — auto-noise must not destroy a partially-non-finite option.

RobustnessAnalyzerV2._apply_auto_scaled_noise() scales the injected noise by the
standard deviation of an option's pre-noise samples. Computed over the WHOLE
sample array, that std is nan/inf if even one sample is non-finite, and
rng.normal(0, nan) then poisons EVERY sample — turning a mostly-finite option
into a wholly-dead (all-nan) one, dropping its downside block and corrupting its
validity metrics.

The std must be taken over the FINITE samples only (mirroring the downside
percentile family in api/robustness.py, which filters to finite_cleaned), leaving
non-finite entries as-is for the downstream finite-mask machinery to handle
(n_valid_samples / finite_cleaned percentiles / per-index downside regret).

Scope note: this hardens the noise MECHANISM. It does NOT by itself make an
inf/nan-retaining option return a 200 — outcome.mean/std (analyzer) are raw
np.mean/np.std over the post-noise samples and remain non-finite for such an
option, which the JSONResponse serializer rejects (allow_nan=False). Masking the
mean is a separate tracked row. Here we pin the mechanism + the finite-mask
downside consumer.

Arch step 1 (2026-07-26): auto-scaled noise is DEFAULT OFF
(``ENABLE_AUTO_SCALED_NOISE``), so every call below passes ``enabled=True``.
The subject of this file IS the noise path, and its pinned expectations are
unchanged — the opt-in restores the behaviour under test, it does not
re-baseline it.
"""

import numpy as np

from src.models.robustness_v2 import NodeV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.downside import cvar_from_samples
from src.utils.rng import SeededRNG


def _outcome_nodes():
    return [NodeV2(id="rev", kind="outcome", label="Rev")]


class TestAutoNoiseFiniteMask:
    def test_partial_non_finite_option_keeps_finite_std_and_downside(self):
        analyzer = RobustnessAnalyzerV2()
        # 100 finite samples + 3 inf (a partially-failed option).
        samples = [float(i) for i in range(100)] + [np.inf, np.inf, np.inf]
        option_outcomes = {"o1": list(samples)}

        out, applied = analyzer._apply_auto_scaled_noise(
            option_outcomes, "rev", _outcome_nodes(), SeededRNG(42), enabled=True
        )
        result = np.array(out["o1"])

        # Noise WAS applied — a finite std existed over the finite subset.
        assert applied is True
        # The 100 finite samples survive as finite values (noised, not destroyed).
        finite = result[np.isfinite(result)]
        assert finite.size == 100
        # The 3 non-finite entries are left as-is (still inf) — NOT turned to nan.
        # Pre-fix, np.std over the whole array is nan and normal(0, nan) destroyed
        # the WHOLE option to all-nan (finite.size == 0).
        assert int(np.sum(np.isinf(result))) == 3
        assert int(np.sum(np.isnan(result))) == 0

        # Downstream downside block (api/robustness.py finite_cleaned path) is now
        # computable from the finite subset; pre-fix that subset was empty.
        finite_cleaned = result[np.isfinite(result)]
        assert len(finite_cleaned) == 100
        assert np.isfinite(cvar_from_samples(finite_cleaned))
        assert np.isfinite(float(np.percentile(finite_cleaned, 5)))

    def test_all_finite_option_byte_identical_to_pre_fix(self):
        # Regression guard: the all-finite path (the norm) must be unchanged —
        # same std source, same RNG draw count (len(samples)) — so goldens are
        # unmoved. Reproduced independently with a fresh identical rng.
        analyzer = RobustnessAnalyzerV2()
        samples = [float(i) for i in range(50)]

        out, applied = analyzer._apply_auto_scaled_noise(
            {"o1": list(samples)}, "rev", _outcome_nodes(), SeededRNG(7), enabled=True
        )

        arr = np.array(samples)
        std = float(np.std(arr))  # over all-finite == over the finite mask
        ref_rng = SeededRNG(7)
        ref_noise = np.array([ref_rng.normal(0, std * 1.0) for _ in range(len(samples))])
        assert out["o1"] == (arr + ref_noise).tolist()
        assert applied is True

    def test_all_non_finite_option_skipped_no_crash(self):
        # Defensive edge: an option with NO finite samples has no std to scale
        # from — skip it (leave samples untouched) rather than crash.
        analyzer = RobustnessAnalyzerV2()
        option_outcomes = {"o1": [np.inf, np.nan, -np.inf]}
        out, applied = analyzer._apply_auto_scaled_noise(
            option_outcomes, "rev", _outcome_nodes(), SeededRNG(1), enabled=True
        )
        assert applied is False
        assert len(out["o1"]) == 3

    def test_finite_samples_with_overflow_std_not_destroyed(self):
        # VOI F-3 (hunter): the finite-mask protects against non-finite SAMPLES
        # but NOT against variance OVERFLOW of finite samples. np.std squares the
        # values, so all-finite draws of magnitude >= ~1.4e154 give
        # outcome_std = inf, and rng.normal(0, inf) = nan noise then DESTROYS every
        # finite sample the mask was built to protect.
        analyzer = RobustnessAnalyzerV2()
        # 100 ALL-FINITE samples of magnitude ~1e160 (max |x| < 1.8e308, so every
        # entry is finite) — but np.std over them overflows to inf.
        samples = list(np.random.default_rng(0).normal(0, 1e160, 100))
        arr_in = np.array(samples)
        assert np.isfinite(arr_in).all(), "precondition: pre-noise all-finite"
        assert not np.isfinite(np.std(arr_in)), "precondition: np.std overflows to inf"

        out, applied = analyzer._apply_auto_scaled_noise(
            {"o1": list(samples)}, "rev", _outcome_nodes(), SeededRNG(0), enabled=True
        )
        result = np.array(out["o1"])

        # Pre-fix: noise = rng.normal(0, inf) = all-nan → noised[mask] = finite + nan
        # → 0/100 finite (the finite majority the mask exists to protect is wiped).
        # Post-fix: the option is skipped (non-finite std, like the no-finite-sample
        # case), so all 100 finite samples survive unchanged.
        assert int(np.sum(np.isfinite(result))) == 100
        assert int(np.sum(np.isnan(result))) == 0
        # Noise was NOT applied to this option (no representable std to scale from).
        assert applied is False
        # Honest degrade: samples pass through untouched (no fabricated noise).
        assert out["o1"] == list(samples)
