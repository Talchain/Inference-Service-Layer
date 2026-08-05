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
