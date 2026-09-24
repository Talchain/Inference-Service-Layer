"""Goal attainment follows the stated direction in the resolved sample frame."""

import math

import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


def request_for(direction, frame):
    return RobustnessRequestV2(
        graph={
            "nodes": [
                {
                    "id": "driver",
                    "kind": "factor",
                    "label": "Driver",
                    "observed_state": {"value": 0.5},
                },
                {
                    "id": "goal",
                    "kind": "outcome",
                    "label": "Goal",
                    "observed_state": {"value": 0.75, "baseline": 0.75},
                },
            ],
            "edges": [
                {
                    "from": "driver",
                    "to": "goal",
                    "exists_probability": 1.0,
                    "strength": {"mean": 1.0, "std": 0.0011},
                }
            ],
        },
        options=[
            {"id": "lower", "label": "Lower", "interventions": {"driver": 0.1}},
            {"id": "higher", "label": "Higher", "interventions": {"driver": 0.9}},
        ],
        goal_node_id="goal",
        goal_direction=direction,
        goal_threshold=0.5,
        goal_threshold_frame=frame,
        n_samples=100,
        seed=42,
        analysis_types=["comparison"],
    )


def probability_for(direction, frame, compared):
    request = request_for(direction, frame)
    analyser = RobustnessAnalyzerV2()
    plan, warning = analyser._resolve_goal_threshold_in_sample_frame(request)
    assert warning is None
    # In level frame B=.75 and status quo=.25: samples = levels - .5.
    samples = compared if frame == "delta" else [value - 0.5 for value in compared]
    results = analyser._compute_option_results(
        outcomes={option.id: samples for option in request.options},
        wins={option.id: 0 for option in request.options},
        request=request,
        goal_threshold_plan=plan,
        status_quo_outcomes=[0.25] * len(samples) if frame == "level" else None,
    )
    return results[0].probability_of_goal


@pytest.mark.parametrize("frame", ["delta", "level"])
@pytest.mark.parametrize(
    ("direction", "expected"),
    [("minimise", 0.75), ("maximise", 0.5), (None, 0.5), ("target", 0.5)],
)
def test_attainment_counts_the_requested_tail(frame, direction, expected):
    # Three at/below .5, two at/above .5. An inverted tail cannot pass.
    assert probability_for(direction, frame, [0.125, 0.25, 0.5, 0.875] * 25) == expected


@pytest.mark.parametrize("frame", ["delta", "level"])
@pytest.mark.parametrize("direction", ["minimise", "maximise", None])
def test_threshold_equality_is_included(frame, direction):
    assert probability_for(direction, frame, [0.5] * 100) == 1.0


@pytest.mark.parametrize("frame", ["delta", "level"])
def test_minimise_retains_non_finite_mask_and_partial_extremum_guard(frame):
    # Invalid draws are not evidence of success or failure.
    assert probability_for("minimise", frame, [0.25, 0.75, math.inf, -math.inf]) == 0.5
    assert probability_for("minimise", frame, [0.25, 0.25, math.inf, -math.inf]) is None
    assert probability_for("minimise", frame, [0.75, 0.75, math.inf, -math.inf]) is None


@pytest.mark.parametrize("frame", ["delta", "level"])
@pytest.mark.parametrize(
    ("direction", "lower", "higher"),
    [("minimise", 1.0, 0.0), ("maximise", 0.0, 1.0), (None, 0.0, 1.0)],
)
def test_real_analysis_preserves_direction_through_goal_probability(
    frame, direction, lower, higher
):
    response = RobustnessAnalyzerV2().analyze(request_for(direction, frame))
    results = {result.option_id: result for result in response.results}
    assert results["lower"].probability_of_goal == lower
    assert results["higher"].probability_of_goal == higher


@pytest.mark.parametrize("direction", ["minimise", "maximise", None])
def test_unattested_frame_still_withholds_probability(direction):
    response = RobustnessAnalyzerV2().analyze(request_for(direction, None))
    for result in response.results:
        assert result.probability_of_goal is None
        assert "probability_of_goal" not in result.model_dump(exclude_none=True)
