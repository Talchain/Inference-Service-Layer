"""Refused accumulation comparisons are absent from optional flip diagnostics."""

import copy
import math
import time

import pytest

import src.services.robustness_analyzer_v2 as rav2
from tests.unit.test_accumulation_identity import RATE, accumulation_request
from tests.unit.test_factor_flip_values import _control_graph
from tests.unit.test_r3_identity_evaluation import v2_body


def test_flip_consumers_exclude_refused_options_and_backgrounds(monkeypatch):
    analyzer = rav2.RobustnessAnalyzerV2()
    assert analyzer._argmax_option({"invalid": math.nan, "good": 1.0}) == "good"
    with pytest.raises(rav2.AccumulationDrawRefusedError):
        analyzer._argmax_option({"invalid": math.nan})

    request_dict = accumulation_request()
    request_dict["options"][0]["interventions"][RATE] = 1.2
    request = rav2.RobustnessRequestV2.model_validate(request_dict)
    evaluator = rav2.SCMEvaluatorV2(request.graph)
    rows = analyzer._compute_edge_e_values(request, evaluator)
    assert rows
    assert all(row["baseline_winner_id"] == "raise" for row in rows)
    assert all(row["alternative_winner_id"] != "keep" for row in rows)

    entries = [copy.deepcopy(rows[0])]
    monkeypatch.setattr(analyzer, "_sample_flip_backgrounds", lambda *args: [{}, {}, {}])
    flips = iter([0.25, "refused", None])

    def next_flip(*args):
        value = next(flips)
        if value == "refused":
            raise rav2.AccumulationDrawRefusedError("no informative background")
        return value

    monkeypatch.setattr(analyzer, "_flip_mean_under_background", next_flip)
    assert analyzer._attach_flip_stability_bands(request, evaluator, entries, 7)
    assert entries[0]["stability"] == {
        "n_seeds": 2, "n_seeds_flipped": 1, "seed_flip_means": [0.25, None],
        "band_min": 0.25, "band_median": 0.25, "band_max": 0.25, "band_width": 0.0,
    }

    at_min, at_max = {"invalid": math.nan, "good": 1.0}, {"invalid": math.nan, "good": 2.0}
    intercepts, slopes = analyzer._affine_coefficients(at_min, at_max)
    assert intercepts == {"good": 1.0} and slopes == {"good": 1.0}
    with pytest.raises(rav2.AccumulationDrawRefusedError):
        analyzer._affine_coefficients({"bad": 1.0}, {"bad": math.nan})

    legacy = _control_graph()
    lever = next(node for node in legacy.graph.nodes if node.id == "fac_lever")
    calls = iter([False, True, False])
    real_coefficients = analyzer._affine_coefficients

    def coefficients(*args):
        if next(calls):
            raise rav2.AccumulationDrawRefusedError("refused factor background")
        return real_coefficients(*args)

    monkeypatch.setattr(analyzer, "_affine_coefficients", coefficients)
    band = analyzer._factor_flip_band(
        legacy, rav2.SCMEvaluatorV2(legacy.graph), lever, 0.3,
        [{(edge.from_, edge.to): edge.strength.mean for edge in legacy.graph.edges}] * 3,
        time.monotonic(), 8000.0,
    )
    assert band["n_seeds"] == 2 and band["n_seeds_flipped"] == 2
    assert band["seed_flip_values"] == [0.8, 0.8]


def test_uninformative_flip_probes_reuse_unavailable_warning():
    request = accumulation_request()
    for option in request["options"]:
        option["interventions"][RATE] = 1.2
    request["include_e_values"] = True
    request["include_factor_flips"] = True
    body = v2_body(request)
    warnings = {warning["code"]: warning for warning in body["inference_warnings"]}
    for code in ("E_VALUES_UNAVAILABLE", "FACTOR_FLIPS_UNAVAILABLE"):
        assert warnings[code]["detail"]["reason"] == "accumulation_draw_refused"
    assert body.get("factor_flip_values") is None


def test_legacy_flip_control_remains_exact():
    request = _control_graph()
    rows = rav2.RobustnessAnalyzerV2()._compute_factor_flip_values(
        request, rav2.SCMEvaluatorV2(request.graph), 4242
    )
    lever = next(row for row in rows if row["factor_id"] == "fac_lever")
    assert lever["flip_value"] == 0.8
    assert lever["baseline_winner_id"] == "opt_a"
    assert lever["alternative_winner_id"] == "opt_c"
    assert lever["direction"] == "increase"
