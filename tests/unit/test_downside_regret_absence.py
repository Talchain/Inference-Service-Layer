"""B2 downside — an option with NO finite MC draw has NO honest expected regret.

Defect (ROADMAP 2.449 follow-up, found by the B2 ISL lane):
``expected_regret_per_option`` assigned a **fabricated 0.0** to an option whose
every pre-noise draw is non-finite. 0.0 is not a neutral placeholder for this
statistic — it is the value of the option that WINS EVERY SAMPLE, i.e. the most
favourable regret expressible. The fabrication then had a SECOND reader that the
"it never reaches the wire" argument at ``api/robustness.py`` did not cover:

    robustness_analyzer_v2.py:2507
        finite_regrets = [r for r in pre_noise_expected_regret.values()
                          if math.isfinite(r)]
        decision_evpi_bound = min(finite_regrets) if finite_regrets else None

``math.isfinite(0.0)`` is True, so the fabricated value entered the population
and ``min`` collapsed the whole-decision EVPI bound to 0.0. That bound is the cap
every per-factor EVPPI is clamped to (``:6492`` and ``:6507``), so a single
option whose draws overflowed would silently tell the user that **no factor is
worth learning about** — a fabricated scientific claim, in the direction that
suppresses the capability rather than the one a reader would notice.

The honest value is ABSENCE. It is also the policy the codebase already states
for the wire ``decision_evpi`` (``api/robustness.py:1168-1173``: an option with
no samples "carr[ies] no honest regret, [was] never in the population"), so this
makes the two readers agree instead of silently disagreeing.

Every assertion binds to its option by IDENTITY (option id), never by a value
predicate a sibling could satisfy (trap #19), and every absence assertion is
preceded by a PRESENCE the same instrument can see (trap #13).
"""

import math

import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.downside import expected_regret_per_option

# NOTE: `decision_evpi_from_regrets` is imported INSIDE the tests that use it,
# not at module scope. At pristine it does not exist, and a module-level import
# would abort COLLECTION of this whole file — every other test here would report
# as an error rather than as its own specific RED. Per-test imports keep each
# defect's RED signature legible. (Same convention as
# tests/integration/test_numerics_honesty_batch.py.)

INF = float("inf")
NAN = float("nan")

# ---------------------------------------------------------------------------
# Analytic fixture — the truth is computed BY HAND here, never snapshotted from
# the implementation (a snapshot of our own output proves nothing about whether
# the statistic is right).
#
#   opt_a  = [ 1,  2,  3,  4, 10]
#   opt_b  = [10,  9,  8,  7,  1]
#   best_i = [10,  9,  8,  7, 10]      (per-sample max over the finite options)
#
#   regret_a = mean(10-1, 9-2, 8-3, 7-4, 10-10) = mean(9,7,5,3,0) = 24/5 = 4.8
#   regret_b = mean(10-10, 9-9, 8-8, 7-7, 10-1) = mean(0,0,0,0,9) =  9/5 = 1.8
#
# The options CROSS (each wins somewhere), so the true whole-decision EVPI
# min_o regret_o = 1.8 is strictly POSITIVE — which is what makes the fabricated
# 0.0 observable. With a dominant option the true min is 0.0 anyway and the
# defect hides.
# ---------------------------------------------------------------------------
OPT_A = [1.0, 2.0, 3.0, 4.0, 10.0]
OPT_B = [10.0, 9.0, 8.0, 7.0, 1.0]
TRUE_REGRET_A = 4.8
TRUE_REGRET_B = 1.8
TRUE_DECISION_EVPI = 1.8


class TestAbsentRegretForAnOptionWithNoFiniteDraw:
    """The statistic is ABSENT, never a fabricated 0.0."""

    @pytest.mark.parametrize("dead_value", [INF, -INF, NAN], ids=["inf", "-inf", "nan"])
    def test_dead_option_regret_is_absent_while_siblings_stay_exact(self, dead_value):
        out = expected_regret_per_option(
            {"opt_a": OPT_A, "opt_b": OPT_B, "opt_dead": [dead_value] * 5}
        )

        # POSITIVE CONTROL (trap #13): the same call, same instrument, SEES the
        # two present regrets — and sees them at their hand-computed values, so
        # the absence below cannot be an instrument that reports nothing.
        assert out["opt_a"] == pytest.approx(TRUE_REGRET_A), (
            "positive control failed: opt_a's regret must be the analytic 4.8"
        )
        assert out["opt_b"] == pytest.approx(TRUE_REGRET_B), (
            "positive control failed: opt_b's regret must be the analytic 1.8"
        )

        # The claim, bound to opt_dead BY ID.
        assert out["opt_dead"] is None, (
            f"an option with no finite draw has no honest expected regret; "
            f"got {out['opt_dead']!r} (0.0 would be the regret of an option that "
            f"WINS EVERY SAMPLE — the most favourable value, fabricated)"
        )

    def test_a_partially_finite_option_still_gets_a_real_regret(self):
        """Absence is for NO finite draw — not for a merely degraded one.

        Guards the fix against over-reach: an option with one finite draw is
        still measurable and must keep a number.
        """
        out = expected_regret_per_option(
            {"opt_a": OPT_A, "opt_b": OPT_B, "opt_partial": [INF, INF, INF, INF, 2.0]}
        )
        # At sample 4 the finite options are a=10, b=1, partial=2 -> best=10,
        # so partial's regret averages over the ONE sample where it is finite:
        # 10 - 2 = 8.0
        assert out["opt_partial"] == pytest.approx(8.0)


class TestDecisionEvpiFromRegrets:
    """min over the options that HAVE an honest regret — absence never lowers it."""

    def test_min_over_present_regrets_is_the_analytic_value(self):
        from src.utils.downside import decision_evpi_from_regrets

        assert decision_evpi_from_regrets(
            {"opt_a": TRUE_REGRET_A, "opt_b": TRUE_REGRET_B}
        ) == pytest.approx(TRUE_DECISION_EVPI)

    def test_an_absent_regret_does_not_collapse_the_bound(self):
        """The defect, at the reader that the wire-level guard never covered."""
        from src.utils.downside import decision_evpi_from_regrets

        assert decision_evpi_from_regrets(
            {"opt_a": TRUE_REGRET_A, "opt_b": TRUE_REGRET_B, "opt_dead": None}
        ) == pytest.approx(TRUE_DECISION_EVPI)

    @pytest.mark.parametrize("bad", [INF, -INF, NAN], ids=["inf", "-inf", "nan"])
    def test_a_non_finite_regret_is_excluded(self, bad):
        from src.utils.downside import decision_evpi_from_regrets

        assert decision_evpi_from_regrets(
            {"opt_a": TRUE_REGRET_A, "opt_bad": bad}
        ) == pytest.approx(TRUE_REGRET_A)

    def test_no_honest_regret_anywhere_is_absent_not_zero(self):
        from src.utils.downside import decision_evpi_from_regrets

        assert decision_evpi_from_regrets({"opt_dead": None}) is None
        assert decision_evpi_from_regrets({}) is None


# ---------------------------------------------------------------------------
# The real seam: the production analyzer, not a re-implementation of its
# expression. A test that re-inlines `min(...)` would be a guard agreeing with
# itself (trap 13b) — this drives `RobustnessAnalyzerV2.analyze` end-to-end.
# ---------------------------------------------------------------------------

HUGE_FINITE = 1.7e308  # one contribution stays finite; two sum to +inf


def _crossing_graph_request(include_dead_option: bool) -> RobustnessRequestV2:
    """Two CROSSING live options (positive true EVPI) plus, optionally, an
    option whose every draw overflows to non-finite.

    ``f_huge_a`` + ``f_huge_b`` both land on the goal at ~float64 max, so any
    option that does NOT intervene on both of them overflows on every draw.
    ``f_x`` / ``f_y`` have exists_probability 0.5, so the two live options each
    win on some draws (the crossing that makes the true EVPI positive).
    ``f_info`` is the non-lever uncertain factor EVPPI is computed for.
    """
    nodes = [
        {"id": "f_huge_a", "kind": "factor", "label": "HA", "observed_state": {"value": HUGE_FINITE}},
        {"id": "f_huge_b", "kind": "factor", "label": "HB", "observed_state": {"value": HUGE_FINITE}},
        {"id": "f_x", "kind": "factor", "label": "X", "observed_state": {"value": 1.0}},
        {"id": "f_y", "kind": "factor", "label": "Y", "observed_state": {"value": 1.0}},
        {"id": "f_info", "kind": "factor", "label": "INFO", "observed_state": {"value": 1.0}},
        {"id": "goal_out", "kind": "outcome", "label": "Goal"},
    ]
    edges = [
        {"from": "f_huge_a", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 1.0, "std": 0.05}},
        {"from": "f_huge_b", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 1.0, "std": 0.05}},
        {"from": "f_x", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 3.0, "std": 0.5}},
        {"from": "f_y", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 3.0, "std": 0.5}},
        {"from": "f_info", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 2.0, "std": 0.3}},
    ]
    options = [
        {"id": "opt_x", "label": "Push X",
         "interventions": {"f_huge_a": 0.5, "f_huge_b": 0.5, "f_x": 5.0, "f_y": 0.0}},
        {"id": "opt_y", "label": "Push Y",
         "interventions": {"f_huge_a": 0.5, "f_huge_b": 0.5, "f_x": 0.0, "f_y": 5.0}},
    ]
    if include_dead_option:
        # Does NOT intervene on the two huge factors -> every draw overflows.
        options.append({"id": "opt_dead", "label": "Dead", "interventions": {"f_x": 1.0}})

    return RobustnessRequestV2.model_validate(
        {
            "graph": {"nodes": nodes, "edges": edges},
            "options": options,
            "goal_node_id": "goal_out",
            "n_samples": 200,
            "seed": 42,
            "parameter_uncertainties": [
                {"node_id": "f_info", "distribution": "normal", "std": 2.0}
            ],
            "include_voi": True,
        }
    )


ROUTE_B_BIG = 5.0e306
ROUTE_B_SEED = 99


def _route_b_request() -> RobustnessRequestV2:
    """ROUTE B — an UNKNOWN decision-EVPI bound must not be applied as a CAP OF ZERO.

    This is the fixture that makes the bound's absence OBSERVABLE, and it is a
    different regime from every other test here. There is **no dead option**:
    every draw of every option is finite. What overflows is the REGRET MEAN —
    `mean_i(best_i - o_i)` sums 200 strictly-positive terms of ~1e306, so both
    regrets are `inf` and the bound is honestly `None`.

    The trick that makes `evppi_raw` FINITE at the same time is that each option's
    outcomes **alternate sign** (+BIG / 0 / -BIG, via paired coin-flip edges with
    +1 and -1 strengths), so its own partial sums CANCEL — the estimator's
    baseline stays finite and the least-squares fit succeeds. A dead option can
    never produce this: non-finite draws make `evppi_raw` nan, and then
    `max(0.0, nan) == 0.0` on both sides of the mutant, so nothing discriminates.
    That is precisely why an earlier sweep of this lane's — which only ever used
    dead options — found 0 discriminating cases and must NOT have been read as
    evidence that none exist.

    Measured at this shape: 8 of 12 (BIG, seed) settings discriminate.
    """
    nodes = [
        {"id": "f_ap", "kind": "factor", "label": "AP", "observed_state": {"value": ROUTE_B_BIG}},
        {"id": "f_an", "kind": "factor", "label": "AN", "observed_state": {"value": ROUTE_B_BIG}},
        {"id": "f_bp", "kind": "factor", "label": "BP", "observed_state": {"value": ROUTE_B_BIG}},
        {"id": "f_bn", "kind": "factor", "label": "BN", "observed_state": {"value": ROUTE_B_BIG}},
        {"id": "f_info", "kind": "factor", "label": "INFO", "observed_state": {"value": 1.0}},
        {"id": "goal_out", "kind": "outcome", "label": "Goal"},
    ]
    edges = [
        {"from": "f_ap", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 1.0, "std": 0.01}},
        {"from": "f_an", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": -1.0, "std": 0.01}},
        {"from": "f_bp", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 1.0, "std": 0.01}},
        {"from": "f_bn", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": -1.0, "std": 0.01}},
        {"from": "f_info", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 2.0, "std": 0.3}},
    ]
    options = [
        {"id": "opt_p", "label": "P", "interventions": {"f_bp": 0.0, "f_bn": 0.0}},
        {"id": "opt_q", "label": "Q", "interventions": {"f_ap": 0.0, "f_an": 0.0}},
    ]
    return RobustnessRequestV2.model_validate(
        {
            "graph": {"nodes": nodes, "edges": edges},
            "options": options,
            "goal_node_id": "goal_out",
            "n_samples": 200,
            "seed": ROUTE_B_SEED,
            "parameter_uncertainties": [
                {"node_id": "f_info", "distribution": "normal", "std": 2.0}
            ],
            "include_voi": True,
        }
    )


class TestAnUnknownBoundIsNotAppliedAsACapOfZero:
    """The property `decision_evpi_from_regrets`' docstring asserts and nothing
    else tests: `None` means "no cap", NEVER "cap of zero".

    Found by adversarial review, which refuted this lane's claim that the `None`
    branch was unreachable. It is reachable, and the consequence is user-visible:
    every factor's EVPPI clamped to zero — "nothing is worth finding out" — on a
    graph with no dead option and no failed sample.
    """

    def test_unknown_bound_leaves_a_real_evppi_intact(self):
        response = RobustnessAnalyzerV2().analyze(_route_b_request())

        by_id = {r.option_id: r for r in response.results}
        assert set(by_id) == {"opt_p", "opt_q"}, f"fixture drift: {sorted(by_id)}"

        # IN-RUN GUARD 1: every draw is finite. Without this the fixture could
        # silently drift into the dead-option regime, where `evppi_raw` is nan and
        # the discriminating assertion below would pass for the WRONG reason.
        for option_id in ("opt_p", "opt_q"):
            samples = by_id[option_id].outcome_distribution.samples or []
            assert len(samples) == 200, f"{option_id}: expected 200 draws"
            assert all(math.isfinite(s) for s in samples), (
                f"{option_id}: this fixture REQUIRES an entirely finite sample "
                f"population — it is testing route B, not the dead-option route"
            )

        # IN-RUN GUARD 2: every regret is non-finite, so the bound really is None.
        regrets = {oid: r.pre_noise_expected_regret for oid, r in by_id.items()}
        for option_id, regret in regrets.items():
            assert regret is not None and not math.isfinite(regret), (
                f"{option_id}: fixture requires a NON-finite regret; got {regret!r}"
            )

        from src.utils.downside import decision_evpi_from_regrets

        assert decision_evpi_from_regrets(regrets) is None, (
            "precondition: with every regret non-finite the bound must be absent"
        )

        rows = {r["factor_id"]: r for r in (response.factor_evppi or [])}
        assert "f_info" in rows, "fixture must produce an EVPPI row for f_info"
        row = rows["f_info"]

        # POSITIVE CONTROL: a real, strictly-positive EVPPI exists on this run —
        # so a harness that could only ever yield 0.0 is excluded.
        assert math.isfinite(row["evppi_raw"]) and row["evppi_raw"] > 0.0, (
            f"positive control failed: this fixture must produce a FINITE, "
            f"strictly-positive raw EVPPI; got {row['evppi_raw']!r}"
        )

        # THE DISCRIMINATING ASSERTION, bound to f_info BY ID: an absent bound
        # applies NO cap. Coalescing it to 0.0 clamps this to zero.
        assert row["evppi"] == pytest.approx(row["evppi_raw"], rel=1e-12), (
            f"an UNKNOWN decision-EVPI bound was applied as a CAP OF ZERO: "
            f"evppi={row['evppi']!r} but evppi_raw={row['evppi_raw']!r}"
        )
        assert row["evppi"] > 0.0, (
            "the user asked what is worth finding out and was told 'nothing', "
            "because one bound could not be computed"
        )
        assert row["clamped_high"] is False, (
            "no cap exists, so the per-factor <= total-EVPI clamp must not fire"
        )


OVERFLOW_GAP = 1e307  # finite per draw; the per-sample gap sums past float64 max


def _overflowing_gap_request() -> RobustnessRequestV2:
    """Every sample FINITE, every regret NON-FINITE — the second route to an
    absent bound, with no dead option anywhere.

    `f_p` and `f_q` both sit at ~1e307 with independent exists_probability 0.5
    edges. Each option zeroes the OTHER one, so at a draw where only `f_p`'s edge
    fires `opt_p` leads by ~1e307, and where only `f_q`'s fires `opt_q` does — the
    lead alternates. Each option's own outcomes stay finite (~1e307), but the
    accumulated per-sample regret overflows, so BOTH regrets are `inf`.
    """
    nodes = [
        {"id": "f_p", "kind": "factor", "label": "P", "observed_state": {"value": OVERFLOW_GAP}},
        {"id": "f_q", "kind": "factor", "label": "Q", "observed_state": {"value": OVERFLOW_GAP}},
        {"id": "f_info", "kind": "factor", "label": "INFO", "observed_state": {"value": 1.0}},
        {"id": "goal_out", "kind": "outcome", "label": "Goal"},
    ]
    edges = [
        {"from": "f_p", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 1.0, "std": 0.01}},
        {"from": "f_q", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 1.0, "std": 0.01}},
        {"from": "f_info", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 2.0, "std": 0.3}},
    ]
    options = [
        {"id": "opt_p", "label": "Push P", "interventions": {"f_q": 0.0}},
        {"id": "opt_q", "label": "Push Q", "interventions": {"f_p": 0.0}},
    ]
    return RobustnessRequestV2.model_validate(
        {
            "graph": {"nodes": nodes, "edges": edges},
            "options": options,
            "goal_node_id": "goal_out",
            "n_samples": 200,
            "seed": 42,
            "parameter_uncertainties": [
                {"node_id": "f_info", "distribution": "normal", "std": 2.0}
            ],
            "include_voi": True,
        }
    )


class TestAnalyzerEmitsAbsentRegretForADeadOption:
    def test_dead_option_carries_absent_regret_live_siblings_carry_numbers(self):
        """Identity-bound: opt_dead ABSENT, opt_x/opt_y PRESENT and finite.

        The two live options are the in-run POSITIVE CONTROL — they prove this
        assertion can see a regret when one exists, so the absence on opt_dead
        is specific to opt_dead and not a blanket 'nothing was computed'.
        """
        response = RobustnessAnalyzerV2().analyze(_crossing_graph_request(True))
        by_id = {r.option_id: r.pre_noise_expected_regret for r in response.results}

        assert set(by_id) == {"opt_x", "opt_y", "opt_dead"}, (
            f"fixture drift — expected exactly the three constructed options, got {sorted(by_id)}"
        )

        # POSITIVE CONTROL, by identity.
        for live_id in ("opt_x", "opt_y"):
            assert by_id[live_id] is not None, f"{live_id} must carry a regret"
            assert math.isfinite(by_id[live_id]), f"{live_id} regret must be finite"

        # The claim, by identity.
        assert by_id["opt_dead"] is None, (
            f"opt_dead has no finite draw and must carry NO regret; got {by_id['opt_dead']!r}"
        )

    def test_all_regrets_non_finite_is_a_SECOND_route_to_an_absent_bound(self):
        """The bound is absent on TWO routes, not one — this is the second.

        ⚠ WHY THIS TEST EXISTS. An earlier version of this lane's evidence claimed
        `decision_evpi_from_regrets` can only return None when every option is
        DEAD (no finite draw), and that such input raises in
        `_compute_factor_sensitivity` before EVPPI is reached — so the None branch
        was called unreachable. **That was a demonstration of ONE route, wrongly
        generalised to the branch.** The helper also returns None when every
        regret is NON-FINITE, which needs no dead option at all: every sample can
        be finite while the per-sample regret `best_i - o_i` (or its mean)
        overflows, giving `inf` regrets from a perfectly healthy population.

        Here every option has 200/200 FINITE samples — the in-run positive
        control that this is NOT the dead-option route — and yet every regret is
        non-finite, so the bound is honestly ABSENT. `analyze()` completes; no
        raise, so the "it raises first" argument does not cover this route.
        """
        response = RobustnessAnalyzerV2().analyze(_overflowing_gap_request())
        by_id = {r.option_id: r for r in response.results}
        assert set(by_id) == {"opt_p", "opt_q"}, f"fixture drift: {sorted(by_id)}"

        for option_id in ("opt_p", "opt_q"):
            samples = by_id[option_id].outcome_distribution.samples or []
            assert len(samples) == 200, f"{option_id}: expected 200 draws, got {len(samples)}"
            # POSITIVE CONTROL: every draw is finite. This option is NOT dead.
            assert all(math.isfinite(s) for s in samples), (
                f"{option_id} must have an entirely FINITE sample population — "
                f"otherwise this test would be re-testing the dead-option route"
            )
            regret = by_id[option_id].pre_noise_expected_regret
            assert regret is not None and not math.isfinite(regret), (
                f"{option_id}: this fixture must produce a NON-FINITE regret "
                f"(the overflowing-gap route); got {regret!r}"
            )

        from src.utils.downside import decision_evpi_from_regrets

        regrets = {oid: r.pre_noise_expected_regret for oid, r in by_id.items()}
        assert decision_evpi_from_regrets(regrets) is None, (
            "every regret is non-finite, so there is no honest bound — it must be "
            "ABSENT. Coalescing it to 0.0 would clamp every factor's EVPPI to zero."
        )

    def test_the_live_options_regrets_are_unchanged_by_the_dead_options_presence(self):
        """No-regression + the reason the defect mattered: the dead option must
        not perturb the honest population it was polluting."""
        with_dead = {
            r.option_id: r.pre_noise_expected_regret
            for r in RobustnessAnalyzerV2().analyze(_crossing_graph_request(True)).results
        }
        without_dead = {
            r.option_id: r.pre_noise_expected_regret
            for r in RobustnessAnalyzerV2().analyze(_crossing_graph_request(False)).results
        }

        assert set(without_dead) == {"opt_x", "opt_y"}
        for live_id in ("opt_x", "opt_y"):
            assert with_dead[live_id] == pytest.approx(without_dead[live_id], rel=1e-12), (
                f"{live_id}'s regret changed when a dead sibling was added"
            )
        from src.utils.downside import decision_evpi_from_regrets

        # And the control's own EVPI is strictly positive — otherwise this whole
        # fixture would be vacuous (a zero true bound cannot witness a collapse).
        assert decision_evpi_from_regrets(without_dead) > 0.0
