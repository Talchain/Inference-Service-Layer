"""R3-B correctness tests (Task 4 primary tests 1-5, plus EVPPI validation and integrity).

Correctness before finding-rate: every numeric expectation below is a closed-form hand
calculation from corpus values whose semantics are identical to the engine's.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest
from helpers import A1, A2, A3, A4, C2, C3, C4, E3, load, model, run, static

from sim import frames
from sim.corpus import graph_ids, load_graph, verify_corpus
from sim.dynamic import run_model
from sim.frames import Level, held_level
from sim.metrics import validate_evppi_estimator
from sim.run import TIME_PATH_GAP, frozen_check, run_graph
from sim.spec import load_spec, validate
from sim.static import NodeVal, XEffect, evaluate_all, point_draws, template_draws

A1_ZERO = {
    "X_price_churn_direct": 0.0,
    "X_price_churn_via_sensitivity": 0.0,
    "X_feature_competitive_mrr": 0.0,
}

# ------------------------------------------------------------------ integrity


def test_corpus_and_frozen_hashes_match() -> None:
    assert all(verify_corpus().values())
    assert frozen_check()["all_match"]


@pytest.mark.parametrize("gid", graph_ids())
def test_every_spec_validates_against_its_graph(gid: str) -> None:
    assert validate(load_spec(gid), load_graph(gid)) == []


# ------------------------------------------------------------------ 1. deterministic relationships


def test_month0_pro_mrr_identity_in_user_units() -> None:
    ev = static(A1, 1)
    assert float(ev["keep_current_pricing"].nodes["pro_plan_mrr"].value[0]) == pytest.approx(
        49 * 1200
    )


def test_price_rise_at_unchanged_subscribers_is_conditional_plus_12000() -> None:
    ev = static(A1, 1)
    nv = ev["59_with_feature_release"].nodes["pro_plan_mrr"]
    assert nv.value is None  # subscribers are tainted by the unquantified price -> churn link
    assert nv.conditional_on == ["pro_paying_subscribers"]
    assert float(nv.conditional_value[0]) - 49 * 1200 == pytest.approx(10 * 1200)


def test_mutant_product_on_normalised_operands_misses_the_hand_value(monkeypatch: Any) -> None:
    def normalised(graph: Any, node_id: str) -> Level | None:
        lvl = held_level_orig(graph, node_id)
        os_ = graph.nodes[node_id].get("observed_state") or {}
        if lvl is None or os_.get("value") is None:
            return lvl
        return Level(float(os_["value"]), lvl.source, lvl.tier, "mutant", ())

    held_level_orig = frames.held_level
    import sim.static as st

    monkeypatch.setattr(st, "held_level", normalised)
    mutated = float(static(A1, 1)["keep_current_pricing"].nodes["pro_plan_mrr"].value[0])
    assert mutated != pytest.approx(49 * 1200)  # 0.245 x 0.24: the R3-8 mutant goes RED


def test_salary_products_hand_values() -> None:
    ev = static(E3, 1)
    assert (
        float(ev["hire_2_senior_engineers"].nodes["senior_annual_salary_spend"].value[0])
        == 2 * 120000
    )
    assert (
        float(ev["hire_4_junior_engineers"].nodes["junior_annual_salary_spend"].value[0])
        == 4 * 65000
    )


def test_label_sum_reproduces_held_mrr_only_in_mode_x() -> None:
    ev = static(A1, 3, active_derived=("X_sum_mrr",))
    assert float(ev["keep_current_pricing"].nodes["mrr"].value[0]) == pytest.approx(
        49 * 1200 + 16200
    )
    assert 49 * 1200 + 16200 == 75000


def test_static_churn_verdict_matches_typed_natural_effects() -> None:
    ev = static(C2, 0)  # churn 3% is user_assumption, but the effects are Olumi estimates
    assert ev["features_59_pro_price"].nodes["monthly_churn"].value is None
    ev1 = static(C2, 1)
    assert float(ev1["features_59_pro_price"].nodes["monthly_churn"].value[0]) == pytest.approx(
        3 + 0.4 - 0.5
    )


# ------------------------------------------------------------------ 2. held starting values


@pytest.mark.parametrize("gid", graph_ids())
def test_status_quo_reproduces_every_admissible_held_level(gid: str) -> None:
    g, spec = load(gid)
    ev = evaluate_all(g, spec, 1, point_draws(g))[g.baseline_option]
    for node in g.quantity_ids:
        lvl = held_level(g, node)
        nv = ev.nodes[node]
        if lvl is None or lvl.value is None or nv.value is None:
            continue
        if node in spec["declared_identities"] or any(
            d["target"] == node for d in spec["derived_identities"]
        ):
            continue
        assert float(nv.value[0]) == pytest.approx(lvl.value), node


def test_dynamic_month0_equals_held_goal_level() -> None:
    t = run(A4, "T2_growth", {})["keep_pro_at_49"]
    assert float(t.y[0, 0]) == 75000
    c = run(
        C4,
        "X_inversion",
        {"X_price_churn_via_sensitivity": 0.0, "X_spend_total_to_acquisition": 0.0},
    )
    assert float(c["continue_as_now"].y[0, 0]) == pytest.approx(72000)


# ------------------------------------------------------------------ 3. time coherence


def test_growth_rate_compounds_exactly() -> None:
    y = run(A4, "T2_growth", {})["keep_pro_at_49"].y
    assert y.shape == (1, 13)
    assert np.allclose(y[0], 75000 * 1.015 ** np.arange(13))


def test_net_reading_accumulates_linearly() -> None:
    y = run(A1, "X_net_reading", A1_ZERO)["keep_current_pricing"].y
    subs = 1200 + 40 * np.arange(13)
    assert np.allclose(y[0], 49 * subs + 16200)
    assert float(y[0, -1]) == pytest.approx(98520)


def test_gross_reading_matches_churn_decay_closed_form() -> None:
    y = run(A1, "X_gross_reading", A1_ZERO)["keep_current_pricing"].y
    s_star = (40 + 0.03 * 1200) / 0.03
    t = np.arange(13)
    subs = s_star + (1200 - s_star) * 0.97**t
    assert np.allclose(y[0], 49 * subs + 16200)
    assert float(y[0, 1]) == pytest.approx(49 * 1240 + 16200)  # H = 1 by hand


def test_inversion_stock_decays_toward_inflow_over_churn() -> None:
    sc = {"X_price_churn_via_sensitivity": 0.0, "X_spend_total_to_acquisition": 0.0}
    y = run(C4, "X_inversion", sc)["continue_as_now"].y
    s0, s_star = 72000 / 49, 20 / 0.03
    assert np.allclose(y[0], 49 * (s_star + (s0 - s_star) * 0.97 ** np.arange(7)))


def test_parameters_are_drawn_once_per_trajectory() -> None:
    g, spec = load(A1)
    m = copy.deepcopy(model(spec, "X_net_reading"))
    m["flows"] = [m["flows"][0]]  # level inflow only
    m["algebraic"] = []
    m["output"] = "pro_paying_subscribers"
    level = np.array([10.0, 20.0, 30.0])
    nodes = {
        "pro_paying_subscribers": NodeVal(value=np.full(3, 1200.0), delta=np.zeros(3)),
        "monthly_net_pro_additions": NodeVal(value=level, delta=level - 40),
    }
    tr = run_model(g, m, "keep_current_pricing", nodes, nodes, 12, 3, 3)
    assert np.allclose(np.diff(tr.y, axis=1), level[:, None])  # same increment every month


def test_zero_flows_hold_a_stock_constant() -> None:
    g, spec = load(A4)
    m = copy.deepcopy(model(spec, "T2_growth"))
    m["flows"] = []
    tr = run(A4, "T2_growth", {})  # sanity: model runs
    assert tr["keep_pro_at_49"].ok
    ev = evaluate_all(g, spec, 2, point_draws(g), replaced_edges=frozenset(m["replaces_edges"]))
    base = ev[g.baseline_option].nodes
    flat = run_model(g, m, "keep_pro_at_49", base, base, 12, 2, 1)
    assert np.allclose(flat.y, 75000)


# ------------------------------------------------------------------ 4. unsupported calculations withheld


@pytest.mark.parametrize("gid", graph_ids())
def test_strict_tiers_never_emit_a_deadline_verdict(gid: str) -> None:
    res = run_graph(load_graph(gid), {})
    for tier in ("T0", "T1"):
        goal = res["tiers"][tier]["goal"]
        assert goal["status"] == "withheld"
        assert TIME_PATH_GAP in goal["gaps"]
        assert res["tiers"][tier]["evppi"]["status"] == "NOT_COMPUTABLE"


def test_prototype_assumptions_are_refused_below_mode_x() -> None:
    g, spec = load(A1)
    xe = (XEffect("X_price_churn_direct", ("pro_plan_price", "monthly_churn"), 0.5, 10.0),)
    with pytest.raises(ValueError):
        evaluate_all(g, spec, 1, point_draws(g), x_effects=xe)
    with pytest.raises(ValueError):
        evaluate_all(g, spec, 2, point_draws(g), active_derived=("X_sum_mrr",))


def test_spec_validator_rejects_assumptions_in_t2_and_inversion_outside_x() -> None:
    g, spec = load(A4)
    bad = copy.deepcopy(spec)
    model(bad, "T2_growth")["assumptions"] = ["X_pro_mrr_in_mrr"]
    assert any("T2 models may not carry" in e for e in validate(bad, g))
    g4, spec4 = load(C4)
    bad4 = copy.deepcopy(spec4)
    model(bad4, "X_inversion")["tier"] = "T2"
    assert any("inversion is Mode X only" in e for e in validate(bad4, g4))
    bad_sum = copy.deepcopy(spec)
    bad_sum["derived_identities"] = [
        {**copy.deepcopy(load(A1)[1]["derived_identities"][0]), "tier": "T1"}
    ]
    assert any("must be tier X" in e for e in validate(bad_sum, load_graph(A1)))


def test_unquantified_link_taints_instead_of_counting_as_zero() -> None:
    for tier in (1, 2):
        nv = static(A1, tier)["59_with_feature_release"].nodes["monthly_churn"]
        assert nv.value is None
        assert "STATIC_COEFFICIENT_NO_TEMPORAL_MEANING:pro_plan_price->monthly_churn" in nv.gaps


def test_removing_a_default_withholds_the_dependent_claim() -> None:
    at_t1 = static(A1, 1)["49_with_feature_release"].nodes["monthly_churn"]
    assert at_t1.value is None and any("linear_scaling" in gp for gp in at_t1.gaps)
    at_t2 = static(A1, 2)["49_with_feature_release"].nodes["monthly_churn"]
    assert float(at_t2.value[0]) == pytest.approx(3 - 0.5) and "linear_scaling" in at_t2.defaults


def test_inconsistent_identity_and_missing_frame_are_withheld() -> None:
    g, spec = load(C3)
    assert spec["declared_identities"]["mrr"]["status"] == "withheld"
    admitted = copy.deepcopy(spec)
    admitted["declared_identities"]["mrr"]["status"] = "admitted"
    sq = evaluate_all(g, admitted, 1, point_draws(g))["continue_as_now"].nodes["mrr"]
    assert sq.value is None  # the engine itself refuses 49 x 1,000 = 49,000 against held 72,000
    assert "IDENTITY_INCONSISTENT_WITH_HELD_LEVEL:mrr" in sq.gaps
    assert static(A3, 1)["carry_on_as_now"].nodes["pro_plan_mrr"].value is None


def test_missing_baseline_is_never_invented() -> None:
    res = run_graph(load_graph(C2), {})
    assert res["tiers"]["T2"]["goal_models"] == []
    assert "BASELINE_MISSING" in res["tiers"]["T2"]["goal"]["gaps"]
    assert res["tiers"]["X"]["models"] == []


def test_mutant_horizon_for_journey_e_still_emits_no_deadline() -> None:
    g, spec = load(E3)
    mutant = copy.deepcopy(spec)
    mutant["horizon"] = {
        "status": "resolved",
        "months": 3,
        "evidence": spec["horizon"]["evidence"],
        "gaps": [],
    }
    assert (
        mutant["goal"]["threshold_ptr"] is None
        and mutant["goal"]["temporal_semantics"] == "withheld"
    )
    assert mutant["goal_model_t2"]["status"] == "withheld" and mutant["dynamic_models"] == []


def test_options_without_levels_block_whole_decision_ranking() -> None:
    res = run_graph(load_graph(A1), {})
    for m in res["tiers"]["X"]["models"]:
        assert m["ranking"]["status"] == "RANKING_WITHHELD"
        assert set(m["ranking"]["blocking_options"]) >= {"625ec80a", "d1ca58d4", "40bb45e7"}


# ------------------------------------------------------------------ 5. option direction


def test_higher_price_raises_pro_mrr_at_unchanged_subscribers() -> None:
    ev = static(A2, 1)
    v54 = ev["raise_pro_to_54_at_release"].nodes["pro_mrr_at_month_12"].conditional_value
    v59 = ev["raise_pro_to_59_at_release"].nodes["pro_mrr_at_month_12"].conditional_value
    assert v54 is not None and v59 is not None and float(v59[0]) > float(v54[0]) > 49 * 1360


def test_higher_churn_response_lowers_the_deadline_outcome() -> None:
    low = run(A1, "X_gross_reading", A1_ZERO)["59_with_feature_release"].y
    high = run(A1, "X_gross_reading", {**A1_ZERO, "X_price_churn_direct": 4.25})[
        "59_with_feature_release"
    ].y
    assert float(high[0, -1]) < float(low[0, -1])


def test_higher_acquisition_raises_the_stock() -> None:
    sc = {"X_price_churn_via_sensitivity": 0.0, "X_spend_total_to_acquisition": 0.0}
    t = run(C4, "X_inversion", sc)
    assert float(t["advertising_investment"].y[0, -1]) > float(t["continue_as_now"].y[0, -1])


# ------------------------------------------------------------------ EVPPI and reproducibility


def test_isl_evppi_estimator_reproduces_analytic_value_and_zero_control() -> None:
    res = validate_evppi_estimator()
    assert res["all_pass"], json.dumps(res, indent=1)


def test_same_seed_same_draws_and_same_graph_result() -> None:
    g = load_graph(A1)
    a, b = template_draws(g, 500, 7), template_draws(g, 500, 7)
    assert all(np.array_equal(a.amount[k], b.amount[k]) for k in a.amount)
    r1 = json.dumps(run_graph(load_graph(A4), {}), sort_keys=True)
    r2 = json.dumps(run_graph(load_graph(A4), {}), sort_keys=True)
    assert r1 == r2


def test_monte_carlo_converges_between_2k_and_50k_draws() -> None:
    from sim.run import convergence

    runs = convergence()["runs"]
    small, large = runs[0]["options"], runs[-1]["options"]
    for opt, row in small.items():
        p_small, p_large = row["p_by_H"], large[opt]["p_by_H"]
        # SE at n=2,000 implied by the 50k estimate (a 2k estimate of exactly 0 or 1 has SE 0).
        se = float(np.sqrt(p_large * (1 - p_large) / runs[0]["n_draws"]))
        assert abs(p_small - p_large) <= 3 * se + 1e-6, opt


# ------------------------------------------------------------------ MVP-B break-even (Mode X)


@pytest.mark.parametrize("model_id", ["X_net_reading", "X_gross_reading"])
def test_breakeven_threshold_is_where_each_verdict_flips(model_id: str) -> None:
    from sim.breakeven import bisect, margins

    for verdict, margin in margins(model_id).items():
        t = bisect(margin)
        assert margin(0.0) > 0, verdict
        assert margin(t - 1e-6) > 0 >= margin(t + 1e-6), verdict
    # Hand: churn 3 % (held) - 0.2 pp per +10 perception x (75 - 50) = 2.5 %; limit 4 % -> 1.5.
    assert bisect(margins(model_id)["churn_limit"]) == pytest.approx(1.5, abs=1e-9)


def test_breakeven_is_labelled_mode_x_and_matches_the_committed_file() -> None:
    from sim.breakeven import compute
    from sim.corpus import ROOT

    out = compute()
    assert (out["tier"], out["label"], out["excluded_from_claims"]) == (
        "X",
        "prototype_assumption",
        True,
    )
    assert out["goal"]["semantics"] == "attain_by_H"
    assert json.loads((ROOT / "breakeven.json").read_text()) == json.loads(json.dumps(out))


# ------------------------------------------------------------------ reporting (brief format)


def test_sensitivity_rank_matches_a_direct_recomputation_and_the_summary() -> None:
    import re
    import statistics

    from sim.corpus import ROOT
    from sim.report import sensitivity_rank

    results = json.loads((ROOT / "results.json").read_text())
    tops = {}
    for gid, g in results["graphs"].items():
        models = g["tiers"]["X"]["models"]
        ranks = sensitivity_rank(models)
        direct: dict[tuple[str, str], list[float]] = {}
        for m in models:
            if m["template_draws"] and m["evppi"]["status"] == "computed":
                for pid, v in m["evppi"]["parameters"].items():
                    rho = [abs(x) for x in v["spearman_vs_outcome_at_H"].values() if x is not None]
                    if rho:
                        direct.setdefault((m["model"], pid), []).append(max(rho))
        for model, rows in ranks.items():
            for pid, med, n in rows:
                assert med == round(statistics.median(direct[(model, pid)]), 2)
                assert n == len(direct[(model, pid)])
            tops[gid] = max(tops.get(gid, 0.0), rows[0][1])
    # The hand-written summary in EVALUATION.md states these top values.
    assert tops == {
        "pj-20260927T180910Z-A": 0.42,
        "pj-20260927T181846Z-A": 0.66,
        "pj-20260927T183807Z-A": 0.67,
        "pj-20260927T183807Z-C": 0.52,
    }
    text = (ROOT / "EVALUATION.md").read_text()
    for v in tops.values():
        assert re.search(rf"\({v:.2f}\)", text)


def test_evaluation_ends_with_at_most_five_unnested_bullets() -> None:
    from sim.corpus import ROOT

    text = (ROOT / "EVALUATION.md").read_text().rstrip("\n")
    last = text[text.rindex("\n## ") + 1 :].splitlines()[1:]
    bullets = [ln for ln in last if ln.strip()]
    assert 1 <= len(bullets) <= 5
    assert all(ln[:1].isdigit() or ln.startswith("- ") for ln in bullets)
