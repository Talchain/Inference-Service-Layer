"""
Decision-level flip threshold (SCIENCE ROBUSTNESS step 1, EXPERIMENT): the pure grid + bisection search in
``src/services/decision_flip.py``. ``leader_at`` stands in for "the analyser's recommendation at link value x".
"""

import pytest

from src.services.decision_flip import decision_flip_threshold


def step_at(t, below="B", above="A"):
    return lambda x: above if x >= t else below


def test_finds_the_threshold_within_tol_and_names_the_new_leader():
    res = decision_flip_threshold(step_at(0.0625), current=0.25, bound=0.0, tol=0.001)
    assert res["exists"] is True and res["leader"] == "A" and res["to_option_id"] == "B"
    lo, hi = res["bracket"]
    assert lo < 0.0625 <= hi and hi - lo <= 0.001
    assert abs(res["threshold"] - 0.0625) <= 0.001


def test_no_change_before_the_bound_is_an_honest_none():
    res = decision_flip_threshold(lambda x: "A", current=0.25, bound=0.0)
    assert res == {"exists": False, "leader": "A", "threshold": None, "bracket": None, "to_option_id": None,
                   "evaluations": 9}


def test_the_nearest_change_wins_when_the_curve_changes_twice():
    # A on [0.2, 0.25], B on [0.05, 0.2), A again below 0.05: a bisection between current and bound alone would see
    # A at both ends and report "no change"; the grid finds the nearest change at 0.2.
    leader = lambda x: "A" if x >= 0.2 or x < 0.05 else "B"
    res = decision_flip_threshold(leader, current=0.25, bound=0.0, tol=0.001)
    assert res["exists"] is True and res["to_option_id"] == "B"
    assert abs(res["threshold"] - 0.2) <= 0.001


def test_a_change_exactly_at_the_bound_is_found():
    res = decision_flip_threshold(lambda x: "A" if x > 0 else "B", current=0.25, bound=0.0, tol=0.001)
    assert res["exists"] is True and res["bracket"][0] == 0.0 and res["bracket"][1] <= 0.001


def test_stronger_direction_searches_upwards():
    res = decision_flip_threshold(lambda x: "B" if x > 0.7 else "A", current=0.25, bound=1.0, tol=0.001)
    assert res["exists"] is True and abs(res["threshold"] - 0.7) <= 0.001


@pytest.mark.parametrize("kw", [{"grid": 0}, {"tol": 0}])
def test_bad_parameters_are_refused(kw):
    with pytest.raises(ValueError, match="DECISION_FLIP_BAD_PARAMS"):
        decision_flip_threshold(lambda x: "A", 0.25, 0.0, **kw)


# ── Step 2: the affine fast path ───────────────────────────────────────────────────────────────────────────────────
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402

import numpy as np  # noqa: E402

from src.models.robustness_v2 import DecisionFlipRequestV2, RobustnessRequestV2  # noqa: E402
from src.services import decision_flip as df  # noqa: E402
from src.services.robustness_analyzer_v2 import ObjectivePlan, RobustnessAnalyzerV2  # noqa: E402

D1 = json.loads((pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "robustness" / "d1-isl-request.json").read_text())
L1 = ("sprint_capacity_for_ai_reporting", "ai_reporting_module_availability")
SIGNING = "enterprise_prospect_signing_likelihood"


def d1(n=1000, seed="7", **edits):
    q = copy.deepcopy(D1)
    q.update(df.STRIPPED)
    q.update({"n_samples": n, "seed": seed})
    for node in q["graph"]["nodes"]:
        if node["id"] in edits.get("eps", {}):
            node["epsilon_std"] = edits["eps"][node["id"]]
    return q


def capture(q, link=None, mean=None, on=True):
    q = copy.deepcopy(q)
    if link is not None:
        for e in q["graph"]["edges"]:
            if (e.get("from") or e.get("from_"), e["to"]) == link:
                e["strength"]["mean"] = mean
    r = RobustnessRequestV2(**q)
    r._capture_draws = on
    out = RobustnessAnalyzerV2().analyze(r)
    return out._mc_draws if on else out


@pytest.mark.parametrize("sense", ["maximise", "minimise"])
def test_vectorised_p_best_is_the_canonical_winner_rule(sense):
    rng = np.random.default_rng(3)
    v = np.round(rng.normal(size=(3, 400)), 1)  # rounding makes exact ties common
    v[1, ::17] = math.nan
    v[:, 5] = math.nan  # a draw with no finite option credits nobody
    plan = ObjectivePlan(sense=sense, attested=True)
    wins = np.zeros(3)
    for d in range(v.shape[1]):
        finite = {str(i): float(v[i, d]) for i in range(3) if math.isfinite(v[i, d])}
        w = RobustnessAnalyzerV2._winners_for_draw(finite, plan, None) if finite else []
        for o in w:
            wins[int(o)] += 1.0 / len(w)
    assert np.allclose(df.p_best(v, sense, 400), wins / 400, atol=1e-12, rtol=0)


def test_affine_threshold_finds_the_crossing_on_the_grid():
    x0 = np.array([[1.0, 1.0], [0.4, 0.4]])  # A at the current mean (1.0), B constant
    x1 = np.array([[0.0, 0.0], [0.4, 0.4]])  # A at mean 0: A(x) = x
    res = df.affine_threshold(x0, x1, 1.0, ["A", "B"], "A", "maximise", 2, step=0.01)
    assert res["exists"] and res["to_option_id"] == "B"
    assert res["hold"] >= 0.4 > res["flip"] and res["hold"] - res["flip"] <= 0.0100001


def test_capture_mode_gives_exact_common_random_numbers_and_ordinary_mode_does_not():
    q = d1()
    a, b = capture(q, L1, 0.25), capture(q, L1, 0.125)
    # The status-quo option sets the AI capacity to 0, so it never feels this link: identical draws in capture mode.
    assert a["option_outcomes"]["continue_current_plan"] == b["option_outcomes"]["continue_current_plan"]
    z = capture(q, L1, 0.0)
    ai = [np.array(x["option_outcomes"]["ai_reporting_module_sprint"]) for x in (a, b, z)]
    assert np.abs(ai[1] - (ai[0] + ai[2]) / 2).max() < 1e-12  # each draw is affine in the mean


def test_ordinary_mode_still_breaks_ties_from_the_edge_stream():
    # Negative control for the row above: today's analysis is unchanged, ties still consume the edge stream.
    q = d1()
    a = capture(q, L1, 0.25, on=False).results
    b = capture(copy.deepcopy(q), L1, 0.125, on=False).results
    assert [o.win_probability for o in a] != [o.win_probability for o in b]


def test_a_clamp_downstream_of_the_link_makes_it_an_honest_absence():
    clean = RobustnessRequestV2(**d1())
    assert df.downstream_nonlinearity(clean, *L1) is None
    clamped = RobustnessRequestV2(**d1(eps={SIGNING: 0.05}))
    assert df.downstream_nonlinearity(clamped, *L1) == f"clamp:{SIGNING}"
    upstream = RobustnessRequestV2(**d1(eps={"sprint_capacity_for_ai_reporting": 0.05}))
    assert df.downstream_nonlinearity(upstream, *L1) is None  # a clamp UPSTREAM of the link is not in its cone


def test_an_evaluated_product_identity_downstream_is_an_honest_absence(monkeypatch):
    class Plan:
        evaluated, operation = True, "product"

    import src.services.robustness_analyzer_v2 as rav2
    monkeypatch.setattr(rav2, "_resolve_structural_identity_plans", lambda g: {SIGNING: Plan()})
    assert df.downstream_nonlinearity(RobustnessRequestV2(**d1()), *L1) == f"identity:{SIGNING}"
    Plan.operation = "sum"
    assert df.downstream_nonlinearity(RobustnessRequestV2(**d1()), *L1) is None


def test_block_on_d1_quotes_clean_links_and_withholds_a_clamped_one():
    links = [{"from_id": L1[0], "to_id": L1[1]}, {"from_id": SIGNING, "to_id": "quarterly_revenue"}]
    clean = df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate(
        {"request": d1(n=2000), "links": links, "replicates": 2}))
    assert clean.leader_option_id == "ai_reporting_module_sprint"
    for link in clean.links:
        assert link.status in ("quoted", "absent", "no_change")
        assert link.replicate_thresholds is not None and len(link.replicate_thresholds) == 2  # one per replicate
        if link.status == "quoted":
            assert link.to_option_id == "integration_bug_fix_sprint" and 0 < link.threshold < link.current_mean
    clamped = df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate(
        {"request": d1(n=2000, eps={SIGNING: 0.05}), "links": links, "replicates": 2}))
    first = clamped.links[0]
    assert first.status == "absent" and first.reason == f"nonlinear_downstream:clamp:{SIGNING}"
    assert first.replicate_thresholds is None  # no replicate ran: null, never an empty or all-null list


def test_unknown_link_is_refused():
    with pytest.raises(ValueError, match="DECISION_FLIP_UNKNOWN_LINK"):
        df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate(
            {"request": d1(), "links": [{"from_id": "nope", "to_id": "quarterly_revenue"}], "replicates": 2}))


def test_the_exact_rerun_withholds_a_link_the_straight_line_got_wrong(monkeypatch):
    # Force the affine path THROUGH clamped cones (the static guard bypassed): the real re-run at the quoted point must
    # catch the bent line and withhold the link, never quote it.
    q = d1(n=2000, eps={SIGNING: 0.05, "ai_reporting_module_availability": 0.05})
    monkeypatch.setattr(df, "downstream_nonlinearity", lambda *a, **k: None)
    block = df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate({"request": q, "replicates": 2, "links": [
        {"from_id": "ai_reporting_module_availability", "to_id": SIGNING}]}))
    assert (block.links[0].status, block.links[0].reason) == ("absent", "affine_check_failed")
