"""Refused accumulation draws retain their meaning in each statistics consumer."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.rng import SeededRNG
from tests.unit.test_accumulation_identity import accumulation_request


class _DrawEvaluator:
    """Controlled evaluations keep refusal masks independent of production helpers."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def evaluate(self, **_kwargs: object) -> float:
        return next(self._values)


def test_refused_draw_consumers_use_only_informative_population() -> None:
    """One bounded row covers bands, wins, edge probes, buckets and robustness."""
    analyzer = RobustnessAnalyzerV2()
    request = RobustnessRequestV2.model_validate(accumulation_request())
    failures: list[str] = []

    def check(label: str, row: Callable[[], None]) -> None:
        try:
            row()
        except (AssertionError, IndexError, ValueError) as exc:
            failures.append(f"{label}: {exc}")

    def outcomes_and_win() -> None:
        small_request = request.model_copy(update={"n_samples": 4})
        outcomes = {"keep": [1.0, math.nan, 3.0, math.nan], "raise": [2.0] * 4}
        result = analyzer._compute_option_results(
            outcomes, {"keep": 1.0, "raise": 3.0}, small_request
        )[0]
        dist = result.outcome_distribution
        assert (dist.mean, dist.std, dist.median) == (2.0, 1.0, 2.0)
        assert (dist.ci_lower, dist.ci_upper) == pytest.approx((1.05, 2.95))
        assert result.win_probability == 0.5
        # Raw indices remain intact for common-random-number consumers.
        assert dist.samples is not None and len(dist.samples) == 4
        assert math.isnan(dist.samples[1]) and math.isnan(dist.samples[3])

    def edge_probe_populations() -> None:
        # Two evaluations in each arm: one finite, one refused. Each arm's mean
        # is independently measured from its informative draw, not NaN-clamped.
        small_request = request.model_copy(update={"n_samples": 20})
        edge = request.graph.edges[0]
        existence = analyzer._compute_existence_sensitivity(
            small_request,
            edge,
            2.0,
            SeededRNG(7),
            _DrawEvaluator([1.0, math.nan, 3.0, math.nan]),  # type: ignore[arg-type]
        )
        magnitude = analyzer._compute_magnitude_sensitivity(
            small_request,
            edge,
            2.0,
            SeededRNG(7),
            _DrawEvaluator([1.0, math.nan, 3.0, math.nan]),  # type: ignore[arg-type]
        )
        assert existence == -1.0
        assert magnitude == -0.5

    def empty_edge_probe_is_withheld() -> None:
        small_request = request.model_copy(update={"n_samples": 20})
        result = analyzer._compute_existence_sensitivity(
            small_request,
            request.graph.edges[0],
            2.0,
            SeededRNG(7),
            _DrawEvaluator([math.nan] * 4),  # type: ignore[arg-type]
        )
        assert result is None or math.isnan(result)

    def empty_sensitivity_is_withheld() -> None:
        small_request = request.model_copy(update={"n_samples": 20})
        rows = analyzer._compute_sensitivity(
            small_request,
            {"keep": [math.nan] * 20, "raise": [1.0] * 20},
            None,  # type: ignore[arg-type]
            SeededRNG(7),
            _DrawEvaluator([math.nan] * 100),  # type: ignore[arg-type]
        )
        assert rows == []

    def bucket_population() -> None:
        bucket = analyzer._compute_bucket_result(
            np.ones(5, dtype=bool),
            ["keep", None, "raise", None, "raise"],
            {"keep": "Keep", "raise": "Raise"},
            {"keep": 1.0, "raise": 2.0},
        )
        assert bucket is not None
        assert bucket.n_samples == 3
        assert bucket.winner_probability == 2.0 / 3.0
        assert bucket.runner_up_probability == 1.0 / 3.0

    def empty_bucket_is_withheld() -> None:
        assert analyzer._compute_bucket_result(
            np.ones(2, dtype=bool),
            [None, None],
            {"keep": "Keep", "raise": "Raise"},
            {"keep": 1.0, "raise": 2.0},
        ) is None

    def robustness_population() -> None:
        small_request = request.model_copy(update={"n_samples": 5})
        result = analyzer._compute_robustness(
            {"keep": 1.0, "raise": 2.0},
            ["keep", None, "raise", None, "raise"],
            [],
            small_request,
            [{}] * 5,
            _DrawEvaluator([]),  # type: ignore[arg-type]
            7,
        )
        assert result.recommendation_stability == 2.0 / 3.0

    check("outcome mean/median/interval and option win", outcomes_and_win)
    check("partial sensitivity probes", edge_probe_populations)
    check("empty sensitivity probe", empty_edge_probe_is_withheld)
    check("empty reference sensitivity", empty_sensitivity_is_withheld)
    check("conditional winner bucket", bucket_population)
    check("empty conditional winner bucket", empty_bucket_is_withheld)
    check("robustness informative denominator", robustness_population)
    assert not failures, "\n".join(failures)


def test_ancillary_draw_consumers_exclude_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture, regression and calibration use finite aligned measurements only."""
    from src.services import decision_flip, sbc_validator
    from src.utils.evppi import factor_evppi_estimate

    values = np.array([
        [1.0, math.nan, 3.0, math.nan],
        [2.0, 1.0, 2.0, math.nan],
        [math.nan] * 4,
    ])
    assert np.array_equal(decision_flip.p_best(values, "maximise", 4), [0.25, 0.5, 0.0])
    conditional = decision_flip.p_best(values, "maximise", 4, condition_on_informative=True)
    assert np.array_equal(conditional[:2], [0.5, 2.0 / 3.0])
    assert math.isnan(conditional[2])
    complete = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 2.0]])
    assert np.array_equal(
        decision_flip.p_best(complete, "maximise", 3),
        decision_flip.p_best(complete, "maximise", 3, condition_on_informative=True),
    )
    # A has only one measured draw, which it wins. B wins three of four;
    # A remains the conditional leader until its measured line crosses B.
    x0 = np.array([[1.0, math.nan, math.nan, math.nan], [0.5] * 4])
    x1 = np.array([[0.0, math.nan, math.nan, math.nan], [0.5] * 4])
    crossing = decision_flip.first_leader_change(
        x0, x1, ["A", "B"], "A", "maximise", 1.0, condition_on_informative=True
    )
    assert crossing is not None and crossing[:2] == (0.5, "B")

    theta = np.linspace(-1.0, 1.0, 64)
    first, second = theta.copy(), -theta.copy()
    first[[1, 7]] = math.nan
    second[21] = math.nan
    measured = np.isfinite(first) & np.isfinite(second)
    expected = factor_evppi_estimate(
        theta[measured], {"a": first[measured], "b": second[measured]}, seed=7
    )
    actual = factor_evppi_estimate(
        theta, {"a": first, "b": second, "unmeasured": np.full(64, math.nan)}, seed=7
    )
    assert actual == expected
    assert actual.n_samples == 61
    with pytest.raises(ValueError, match="no informative"):
        factor_evppi_estimate(theta, {"unmeasured": np.full(64, math.nan)}, seed=7)

    request = RobustnessRequestV2.model_validate(accumulation_request())
    monkeypatch.setattr(sbc_validator, "_compute_ground_truth", lambda *_args: 0.1)

    def calibration(populations: list[list[float]]):
        draws = iter(populations)
        analyzer = SimpleNamespace(analyze=lambda _request: SimpleNamespace(results=[
            SimpleNamespace(option_id="keep", outcome_distribution=SimpleNamespace(samples=next(draws)))
        ]))
        monkeypatch.setattr(sbc_validator, "RobustnessAnalyzerV2", lambda: analyzer)
        return sbc_validator.run_sbc_validation(
            request.graph, request.options, request.goal_node_id,
            n_trials=len(populations), n_samples=100, ci_levels=[0.9], seed=7,
        )

    calibrated = calibration([[0.1, math.nan, 0.1, math.nan], [math.nan] * 4])
    assert calibrated.coverage_results[0].observed_coverage == 1.0
    assert sum(calibrated.pit_histogram) == 1
    unknown = calibration([[math.nan] * 4])
    assert unknown.coverage_results == []
    assert unknown.pit_chi2_statistic is None and unknown.pit_chi2_p_value is None
    assert unknown.calibrated is None and unknown.pit_uniform is None
    assert unknown.to_dict()["calibrated"] is None
