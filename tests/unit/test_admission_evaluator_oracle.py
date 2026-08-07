"""ROADMAP 2.356 — the compute-admission ORACLE: advertised cost must UPPER-BOUND
the evaluator work a request actually performs.

WHY THIS FILE EXISTS, AND WHY IT IS DIFFERENT FROM EVERY OTHER ADMISSION TEST.

`test_admission_calibration.py` proves the formula is SELF-CONSISTENT (its terms
agree with the advertisement, its versions are pinned, its parameters are
sufficient). `TestPhasePricingInventory` proves every phase has ANSWERED the
pricing question. Neither one ever RUNS an analysis, so neither can notice that
an answer is WRONG — that a phase's real evaluate() count exceeds what the
formula charges for it. That is precisely how ROADMAP 2.356 happened twice:

  * `_run_monte_carlo` is attributed `priced:base_mc` (S*O*W). Since ROADMAP
    2.286 it ALSO runs a complete status-quo SCM evaluation per draw for a
    level-framed goal (`sq_evaluator.evaluate(...)`), which base_mc does not
    price. The attribution was true when written and became false underneath it.
  * `_compute_alternative_winners` is attributed `bounded:` with an HONEST
    "KNOWN-UNDERCHARGE" confession — but a confession in a registry is not a
    bound, and the phase's evaluate() count is genuinely unbounded in the fragile
    edge count.

So this file closes the loop the other two leave open: it INSTRUMENTS the real
`SCMEvaluatorV2` and counts every invocation an end-to-end `analyze()` performs,
then asserts

    compute_weighted_cost(request).total  >=  counted_invocations * W

which is the admission gate's own stated convention (BASE_COST_COEF's comment:
"1 unit per sample x option x (nodes+edges) evaluate()", i.e. one evaluate()
costs W units).

⚠ HONEST SCOPE — read before trusting a green here.

 1. This is an UPPER-BOUND oracle over the WHOLE request, not a per-term
    equality. Terms charged flat and generously (bands at 200*E*O, the
    influence walk pool at 400k) create headroom that can MASK an undercharge
    elsewhere. The boundary shapes below are therefore built to switch those
    phases OFF, so the inequality is tight and an undercharge has nowhere to
    hide. A green on a shape with every phase enabled proves much less; that is
    why the sharp shapes are the ones that gate.
 2. It counts evaluate() invocations, so it is blind to work that is not an SCM
    evaluation (matrix solves inside EVPPI, path enumeration). Those phases are
    priced by their own non-evaluate terms and are out of this oracle's scope.
 3. `evaluate_multi` is counted as ONE invocation: it is a single topological
    pass that reads several target nodes out of the same computed state.
"""
from typing import Dict, List, Tuple

import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
    compute_weighted_cost,
)

from tests.unit.test_admission_calibration import (
    _graph,
    _request_dict,
    _sufficiency_shape_grid,
)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------
class _EvaluateCounter:
    """Counts every SCMEvaluatorV2.evaluate/evaluate_multi call made anywhere.

    Patched onto the CLASS, so it sees evaluators the analyzer constructs
    internally — including the ones a caller never gets a reference to, which is
    exactly where both 2.356 residuals live (`sq_evaluator` is constructed inside
    `_run_monte_carlo` and never escapes it).
    """

    def __init__(self) -> None:
        self.evaluate = 0
        self.evaluate_multi = 0

    @property
    def total(self) -> int:
        return self.evaluate + self.evaluate_multi


@pytest.fixture
def count_evaluates(monkeypatch):
    """Yield a counter wired into SCMEvaluatorV2 for the duration of one test."""
    counter = _EvaluateCounter()
    real_evaluate = SCMEvaluatorV2.evaluate
    real_evaluate_multi = SCMEvaluatorV2.evaluate_multi

    def counting_evaluate(self, *args, **kwargs):
        counter.evaluate += 1
        return real_evaluate(self, *args, **kwargs)

    def counting_evaluate_multi(self, *args, **kwargs):
        counter.evaluate_multi += 1
        return real_evaluate_multi(self, *args, **kwargs)

    monkeypatch.setattr(SCMEvaluatorV2, "evaluate", counting_evaluate)
    monkeypatch.setattr(SCMEvaluatorV2, "evaluate_multi", counting_evaluate_multi)
    return counter


def _run(body: dict, counter: _EvaluateCounter) -> Tuple[int, int, int]:
    """Analyse `body` under the counter. Returns (advertised, actual, W)."""
    request = RobustnessRequestV2(**body)
    advertised = compute_weighted_cost(request).total
    RobustnessAnalyzerV2().analyze(request)
    W = len(request.graph.nodes) + len(request.graph.edges)
    return advertised, counter.total * W, W


# ---------------------------------------------------------------------------
# Boundary shapes
# ---------------------------------------------------------------------------
def _level_framed_body(n_nodes: int, n_edges: int, n_samples: int, n_options: int) -> dict:
    """A LEVEL-framed goal threshold with EVERY optional phase off.

    Sharpness: with sensitivity, VOI, e-values, flips and paths all absent, the
    advertised cost is EXACTLY base_mc = S*O*W and there is ZERO headroom, so the
    per-draw status-quo evaluation has nothing to hide behind.

    The goal must carry an attested observed_state.BASELINE for the level frame
    to resolve — `value` alone is not enough, and the resolver refuses without
    it, leaving `needs_status_quo_reference` False so the shape silently stops
    testing anything (a trap-13 vacuity: the assertion would pass by exercising
    the un-instrumented path). This is not a hypothetical: the first cut of this
    fixture set `value` only, and the positive control below is what caught it.
    """
    body = _request_dict(n_nodes, n_edges, n_samples, n_options, sensitivity=False)
    body["analysis_types"] = ["comparison"]
    goal_id = f"n{n_nodes - 1}"
    for node in body["graph"]["nodes"]:
        if node["id"] == goal_id:
            node["observed_state"] = {"value": 0.7, "baseline": 0.7, "std": 0.05}
    body["goal_threshold"] = 0.9
    body["goal_threshold_frame"] = "level"
    return body


def _fragile_edge_body(n_nodes: int, n_edges: int, n_samples: int, n_options: int) -> dict:
    """A sensitivity request whose edges are fragile, so the marginal-switch
    sweep runs on as many edges as the graph has.

    Sharpness: the sensitivity term is charged at 4*E*min(CAP, S//DIV)*W, which
    is close to 1:1 with the sub-sweep's real evaluate() count, so it contributes
    almost no headroom. Whatever the alternative-winner sweep spends is therefore
    visible as excess.
    """
    body = _request_dict(n_nodes, n_edges, n_samples, n_options, sensitivity=True)
    # Wide, uncertain edges → high elasticity → the fragile set fills up.
    for edge in body["graph"]["edges"]:
        edge["strength"] = {"mean": 0.8, "std": 0.3}
        edge["exists_probability"] = 0.6
    return body


def _root_factor_graph(n_drivers: int, n_mediators: int) -> dict:
    """``n_drivers`` PARENTLESS factors -> ``n_mediators`` mediators -> one outcome.

    ⚠ WHY THIS EXISTS, AND WHY `_graph` CANNOT BE USED FOR A FLIPS SHAPE.

    `_graph` lays a n0->n1->...->n_last chain down FIRST, so every node except n0
    has a parent: the fixture graph has exactly ONE root, and that root carries an
    `observed_state` only when `evpi_factors` is set. The factor-flip phase screens
    only ROOT factors (`_compute_factor_flip_values`: a non-root's `factor_values`
    would ADD to its parent contribution rather than replace a base), so on a
    `_graph` shape the phase finds **0 eligible factors** and returns an empty list
    after a single baseline-winner evaluation.

    Measured at 686fcb7f: `_request_dict(12, 20, 2000, 3)` + `include_factor_flips`
    yields roots=['n0'], eligible=[], **0 flip rows** and 3 extra evaluate() calls,
    while the term charges 25,440 units. So simply SETTING the flag on the existing
    fixtures would have priced a phase that never runs — the oracle would pass by
    testing nothing (trap 13). The topology, not just the flag, is the fix.

    ⚠ WHY THE MEDIATORS, AND WHY A PLAIN FAN-IN IS NOT ENOUGH. A candidate is a
    screened factor whose SLOPE SPREAD across options exceeds epsilon. With every
    driver wired straight to the goal and every option intervening on the same
    node, each driver transmits IDENTICALLY under every option, so the spread is 0
    and the phase yields exactly ONE candidate however many drivers you add: the
    candidate cap, the crossing confirmations and the stability bands never run.
    Measured — that fixture made "remove the candidate cap" an EQUIVALENT mutant.

    Routing drivers through mediators that the options pin one apiece gives each
    driver a genuinely different slope under the option that blocks its path, so
    the spread is real, candidates scale with the driver count, and the cap binds.
    Measured on this shape: the advertised/actual ratio tightens from ~13.7x to
    ~1.3x, i.e. the oracle can now see a ~30% under-charge instead of needing 13x.

    Every driver carries an `observed_state.value`, which is the other half of the
    eligibility test, so eligible == every driver; mediators have parents and are
    correctly never eligible.

    ⚠ AND `m0` CARRIES AN OBSERVED VALUE ON PURPOSE — it is the NEGATIVE CONTROL
    for the producer's ROOT conjunct. Without it, no node in this file satisfies
    "observed value AND not the goal AND has a parent", i.e. the fixture contains
    no member of the class the root rule exists to EXCLUDE, so deleting
    `and not evaluator._parents.get(node.id)` from `_compute_factor_flip_values`
    is an EQUIVALENT mutant: measured at ad6651c3, that deletion left 73/73 GREEN
    across this file, `test_factor_flip_values.py` and `test_flip_stability_bands.py`.
    A predicate can be true as stated and have its BREADTH untested (trap 13b —
    a guard agreeing with itself).

    Giving `m0` an observed value costs nothing and changes nothing NUMERICALLY:
    the evaluator seeds `observed_state.value` as a node's base for ROOT nodes
    only (`robustness_analyzer_v2.py`, `is_root` gate in `_evaluate`), so a
    mediator's observed value is never read by the SCM. Verified empirically —
    every flip shape's eligible set, row ids, flip reasons, evaluate() count and
    advertised total are BYTE-IDENTICAL before and after this line. What it does
    change is that `m0` now satisfies every eligibility conjunct EXCEPT
    parentlessness, so the root conjunct becomes observable and its deletion REDs
    `test_flip_shape_screens_every_root_factor` on the set equality.
    `test_the_fixture_can_observe_the_root_conjunct` PINS this precondition, so
    a tidy-up that drops the observed value fails loudly instead of silently
    restoring the blind spot.
    """
    nodes: List[dict] = [
        {
            "id": f"n{i}",
            "kind": "factor",
            "label": f"N{i}",
            "observed_state": {"value": 0.5, "std": 0.1},
        }
        for i in range(n_drivers)
    ]
    mediators: List[dict] = [
        {"id": f"m{j}", "kind": "factor", "label": f"M{j}"} for j in range(n_mediators)
    ]
    # NEGATIVE CONTROL — see the docstring. `m0` always has at least one parent
    # (driver n0 wires to `m{0 % n_mediators}` == m0 for every n_drivers >= 1),
    # so it is a non-root, non-goal node WITH an observed value: exactly the
    # class the producer's root conjunct exists to exclude, and the only thing
    # that makes deleting that conjunct a biting mutant rather than an
    # equivalent one.
    mediators[0]["observed_state"] = {"value": 0.5, "std": 0.1}
    nodes += mediators
    nodes.append({"id": "goal", "kind": "outcome", "label": "GOAL"})
    # Varied strengths so the slope spreads differ and the candidate ranking is a
    # real ordering rather than an arbitrary tie-break on id.
    edges = [
        {
            "from": f"n{i}",
            "to": f"m{i % n_mediators}",
            "exists_probability": 0.9,
            "strength": {"mean": 0.2 + 0.05 * (i % 7), "std": 0.1},
        }
        for i in range(n_drivers)
    ]
    edges += [
        {
            "from": f"m{j}",
            "to": "goal",
            "exists_probability": 0.9,
            "strength": {"mean": 0.3 + 0.03 * j, "std": 0.1},
        }
        for j in range(n_mediators)
    ]
    return {"nodes": nodes, "edges": edges}


def _eligible_driver_ids(body: dict) -> List[str]:
    """The flip-eligible set, DERIVED from the request rather than hand-listed.

    Mirrors `_compute_factor_flip_values`'s own eligibility test: parentless, not
    the goal, and carrying an observed value or a parameter_uncertainties entry.
    Derived so this helper cannot drift from the graph a test builds (trap 12).
    """
    parents: Dict[str, List[str]] = {n["id"]: [] for n in body["graph"]["nodes"]}
    for e in body["graph"]["edges"]:
        parents[e["to"]].append(e["from"])
    uncertain = {u["node_id"] for u in body.get("parameter_uncertainties") or []}
    return [
        n["id"]
        for n in body["graph"]["nodes"]
        if n["id"] != body["goal_node_id"]
        and not parents[n["id"]]
        and (
            (n.get("observed_state") or {}).get("value") is not None or n["id"] in uncertain
        )
    ]


def _flip_body(n_drivers: int, n_mediators: int, n_samples: int, n_options: int) -> dict:
    """`include_factor_flips` AS PLoT SENDS IT (unconditionally true), on a graph
    that can actually exercise the phase.

    Sharpness: sensitivity, VOI, e-values and paths are all OFF, so the advertised
    total is exactly ``base_mc + factor_flips``. base_mc is EXACTLY tight (S*O
    evaluations, priced S*O*W), so the whole inequality collapses to

        factor_flips term  >=  the flip phase's real evaluate() count x W

    with no other phase's headroom available to hide an undercharge behind. That
    is the sharpest possible shape for this term, and it is the one the six
    pre-existing shapes could not build.
    """
    graph = _root_factor_graph(n_drivers, n_mediators)
    return {
        "graph": graph,
        # Each option pins ONE mediator, which is what gives the drivers behind it
        # a different slope under that option and makes them real candidates.
        "options": [
            {"id": f"o{k}", "label": f"O{k}", "interventions": {f"m{k % n_mediators}": 0.2 + 0.1 * k}}
            for k in range(n_options)
        ],
        "goal_node_id": "goal",
        "n_samples": n_samples,
        "seed": 7,
        "analysis_types": ["comparison", "robustness"],
        "include_factor_flips": True,
    }


#: The oracle's shapes, in ONE place so the run below and the term-coverage guard
#: cannot disagree about what is actually exercised (trap 12 — derive, don't mirror).
_ORACLE_SHAPES = [
    ("representative_base", _request_dict(12, 20, 2000, 3, sensitivity=False)),
    ("representative_sensitivity", _request_dict(12, 20, 2000, 3)),
    ("representative_voi", _request_dict(12, 20, 2000, 3, evpi_factors=4)),
    ("boundary_level_frame", _level_framed_body(12, 20, 400, 3)),
    ("boundary_fragile_edges", _fragile_edge_body(14, 26, 1000, 4)),
    ("boundary_many_options", _fragile_edge_body(10, 18, 1000, 8)),
    # ROADMAP 2.745 — flips, the always-on zero-headroom term.
    # 9 drivers < FACTOR_FLIP_MAX_CANDIDATES, so every candidate is banded.
    ("flips_sharp", _flip_body(9, 3, 400, 3)),
    # 16 drivers > the cap (10), so the cap TRUNCATES the banding set: a different
    # limb of the closed form, and one that reads `candidate_cap_exceeded` rows.
    ("flips_above_candidate_cap", _flip_body(16, 4, 400, 4)),
    # Options multiply the crossing-confirmation limb 2*C*(O-1).
    ("flips_many_options", _flip_body(16, 4, 400, 8)),
]

#: Priced terms that NO oracle shape currently CHARGES, with the reason each is
#: still open. ROADMAP 2.745 removed `factor_flips` from this list by adding the
#: shapes above; the remaining four are rowed, not resolved.
#:
#: ⚠ THIS IS A RATCHET, NOT A MIRROR. The guard below compares it against the term
#: set DERIVED from the sibling sufficiency grid and fails in BOTH directions:
#: a newly-priced term that no shape charges is not in the list -> RED; a term that
#: gains coverage is still in the list -> RED. So it cannot silently drift, and it
#: can only ever shrink.
#:
#: ⚠ AND THE RATCHET IS ABOUT FLAGS, NOT EXECUTION — retiring an entry here
#: requires a precondition pin proving the phase actually runs on the new shape,
#: or 2.745's defect recurs under a green suite. See
#: `TestPricedTermsAreChargedBySomeShape`.
_KNOWN_UNRUN_BY_ORACLE = {
    # Charged flat and generously (200*E*O and 20*E*O), so they carry large
    # headroom rather than zero — the opposite of the factor_flips hazard.
    "bands",
    "e_values",
    # A do()-grid phase; needs a control_candidates shape of its own.
    "evpc",
    # Analytic, not an SCM evaluation, so this oracle's instrument (which counts
    # evaluate() calls) is structurally blind to it — see HONEST SCOPE note 2.
    "path_decomposition",
}


class TestPricedTermsAreChargedBySomeShape:
    """ROADMAP 2.745 — every priced term must be CHARGED by some oracle shape.

    ⚠ READ THE NAME LITERALLY: this proves FLAG REACHABILITY, NOT EXECUTION.
    Both sides of the comparison are derived from `compute_weighted_cost`, which
    gates on request FLAGS and cannot know whether a phase ever ran. A shape that
    charges a term while the phase does nothing scores as covered. Measured at
    ad6651c3: the pre-existing `_graph` shape plus `include_factor_flips` charges
    `factor_flips` 25,440 units and produces **0** flip rows (eligible == []), and
    this guard would count that as `factor_flips` being covered.

    That is the exact vacuity 2.745 exists to close, so DO NOT read a green here
    as evidence a phase executes. Execution is pinned ONLY by the hand-written
    preconditions in `TestFactorFlipShapeIsNotVacuous`. **Retiring any of the four
    remaining entries from `_KNOWN_UNRUN_BY_ORACLE` therefore requires shipping a
    matching precondition pin alongside the new shape** — `evpc` in particular is
    a do()-grid gated on `control_candidates` presence, so a shape can carry
    `control_candidates`, charge the term, exercise nothing, and drop the
    exemption with a fully green suite. Without the pin, 2.745's defect recurs
    under a green ratchet.

    WHAT IT DOES CLOSE. `factor_flips` was priced with W_FACTOR_FLIP_COEF = 1
    (zero headroom) and sent by PLoT on EVERY live request, yet no oracle shape and
    no calibration cell ever ENABLED it: the one always-on term with no slack was
    the one term the oracle that exists to catch mis-pricing never charged. Nothing
    failed, because nothing was checking that the priced set and the charged set
    agree. That flag-absent class is now non-recurring; the
    charged-but-not-executed class is not, and is the reason the pins exist.

    The check is DERIVED on both sides — the priced set from the sibling
    sufficiency grid, the charged set from `_ORACLE_SHAPES` — so adding a term to
    `compute_weighted_cost` without a shape that enables it turns this RED.
    """

    def test_every_priced_term_is_exercised_by_some_shape(self):
        priced = set()
        for body in _sufficiency_shape_grid().values():
            priced |= set(compute_weighted_cost(RobustnessRequestV2(**body)).terms)

        # NOT "run" — `compute_weighted_cost` gates on FLAGS, so this is the set of
        # terms some oracle shape CHARGES. See the class docstring.
        charged = set()
        for _label, body in _ORACLE_SHAPES:
            charged |= set(compute_weighted_cost(RobustnessRequestV2(**body)).terms)

        uncharged = priced - charged
        assert uncharged <= _KNOWN_UNRUN_BY_ORACLE, (
            f"priced term(s) {sorted(uncharged - _KNOWN_UNRUN_BY_ORACLE)} are charged "
            f"by compute_weighted_cost on the sufficiency grid but by NO oracle shape, "
            f"so nothing measures whether the charge is enough. Add a shape that "
            f"enables the phase — AND a precondition pin proving the phase actually "
            f"executes on it (see TestFactorFlipShapeIsNotVacuous), because THIS guard "
            f"cannot tell execution from a flag — or justify it in "
            f"_KNOWN_UNRUN_BY_ORACLE."
        )

        stale = _KNOWN_UNRUN_BY_ORACLE & charged
        assert not stale, (
            f"{sorted(stale)} now HAS oracle coverage but is still listed in "
            f"_KNOWN_UNRUN_BY_ORACLE. Remove it — a stale exception list is exactly "
            f"the hand-maintained mirror this guard exists to prevent. Removing an "
            f"entry also requires the precondition pin described in the class "
            f"docstring; a charged term is not a run phase."
        )

    def test_factor_flips_is_no_longer_exempt(self):
        """The specific regression 2.745 closes: `factor_flips` must never go back
        onto the exemption list."""
        assert "factor_flips" not in _KNOWN_UNRUN_BY_ORACLE, (
            "factor_flips was re-exempted. It is charged at W_FACTOR_FLIP_COEF = 1 "
            "(zero pricing headroom) and PLoT sends include_factor_flips=true "
            "unconditionally, so it must stay under the oracle."
        )


class TestFactorFlipShapeIsNotVacuous:
    """POSITIVE CONTROL for every flips assertion below (trap 13 / trap 13b).

    An upper-bound oracle on a phase that never runs is satisfied trivially. These
    tests PIN THE PRECONDITION in-test, so a flips shape that silently stops
    exercising the phase fails HERE rather than passing quietly downstream.
    """

    def test_the_fixture_can_observe_the_root_conjunct(self):
        """PRECONDITION PIN for the ROOT half of the eligibility rule (trap 13b).

        `test_flip_shape_screens_every_root_factor` asserts a SET EQUALITY against
        the producer's eligible set, but that equality can only DISCRIMINATE the
        root conjunct if the fixture contains a node that would become eligible
        without it — a non-goal node that has a parent AND carries an observed
        value. Nothing in the graph builder forces such a node to exist, and its
        absence is invisible: every test stays green while the conjunct goes
        unobserved. Measured at ad6651c3, before `m0` was given an observed value,
        deleting `and not evaluator._parents.get(node.id)` from
        `_compute_factor_flip_values` left 73/73 GREEN across this file,
        `test_factor_flip_values.py` and `test_flip_stability_bands.py`.

        So this pins the fixture's discriminating power itself, rather than
        trusting it to survive a tidy-up.
        """
        body = _flip_body(9, 3, 400, 3)
        parents: Dict[str, List[str]] = {n["id"]: [] for n in body["graph"]["nodes"]}
        for e in body["graph"]["edges"]:
            parents[e["to"]].append(e["from"])

        excluded_only_by_the_root_rule = [
            n["id"]
            for n in body["graph"]["nodes"]
            if n["id"] != body["goal_node_id"]
            and parents[n["id"]]
            and (n.get("observed_state") or {}).get("value") is not None
        ]
        assert excluded_only_by_the_root_rule, (
            "no node in the flips fixture is a non-goal node that HAS a parent and "
            "carries observed_state.value, so nothing in this file belongs to the "
            "class `_compute_factor_flip_values`'s root conjunct exists to exclude. "
            "Deleting that conjunct from the producer would be an EQUIVALENT mutant "
            "and the flip corpus would stay green. Restore an observed_state on a "
            "mediator in `_root_factor_graph` — never relax this assertion."
        )

        # ...and the derived mirror must still exclude it, or the sibling set
        # equality would pass for the wrong reason.
        eligible = set(_eligible_driver_ids(body))
        assert not (set(excluded_only_by_the_root_rule) & eligible), (
            f"{sorted(set(excluded_only_by_the_root_rule) & eligible)} is parented "
            f"yet _eligible_driver_ids returned it: the helper has stopped mirroring "
            f"the producer's parentless test."
        )

    def test_flip_shape_screens_every_root_factor(self, count_evaluates):
        """RED on the pre-existing fixture: `_graph` yields 0 eligible factors.

        Binds by IDENTITY (the factor_id SET), never by a count or a value
        predicate another object could satisfy (trap 19).
        """
        body = _flip_body(9, 3, 400, 3)
        eligible = _eligible_driver_ids(body)
        assert len(eligible) == 9, (
            f"the flips fixture must supply MANY root factors, got {len(eligible)}: "
            f"{eligible}. A single-root graph (like `_graph`) makes every flips "
            f"assertion in this file vacuous."
        )

        response = RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**body))
        rows = response.factor_flip_values
        assert rows is not None, (
            "factor_flip_values is None — the phase declined (budget) or the flag "
            "did not reach it; the flips oracle below would pass by testing nothing."
        )
        # Rows are dicts on the wire-building path (same access pattern as
        # tests/unit/test_factor_flip_values.py), not FactorFlipValueV2 instances.
        assert {r["factor_id"] for r in rows} == set(eligible), (
            f"flip rows must cover exactly the eligible root factors. "
            f"rows={sorted(r['factor_id'] for r in rows)} eligible={sorted(eligible)}"
        )
        # The factors must not be provably inert, or the phase stops after the
        # screen and the crossing/banding limbs of the term go unmeasured.
        reasons = [r["flip_reason"] for r in rows]
        assert reasons.count("structurally_invariant") < len(rows), (
            f"every factor screened as structurally invariant, so no candidate was "
            f"ever probed or banded: the flips shapes measure the SCREEN only. "
            f"reasons={reasons}"
        )

    def test_candidate_cap_shape_actually_exceeds_the_cap(self, count_evaluates):
        """PRECONDITION PIN for `flips_above_candidate_cap` (trap 13b).

        That shape's whole purpose is to drive the phase PAST
        FACTOR_FLIP_MAX_CANDIDATES so the cap-truncation limb of the closed form is
        the one under test. If the fixture stops producing more candidates than the
        cap — a strength tweak would do it — the shape silently degrades into a
        duplicate of `flips_sharp` and nothing says so.
        """
        body = _flip_body(16, 4, 400, 4)
        rows = RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**body)).factor_flip_values
        assert rows is not None
        capped = [r for r in rows if r["flip_reason"] == "candidate_cap_exceeded"]
        assert capped, (
            f"no row reported `candidate_cap_exceeded`, so the candidate cap "
            f"(FACTOR_FLIP_MAX_CANDIDATES="
            f"{RobustnessAnalyzerV2.FACTOR_FLIP_MAX_CANDIDATES}) never bound and this "
            f"shape is a duplicate of flips_sharp. "
            f"reasons={[r['flip_reason'] for r in rows]}"
        )

    def test_enabling_the_flag_costs_real_evaluations(self, count_evaluates):
        """The flag must CHANGE the work done — a comparator, not an assertion
        about an absolute number.

        The screen alone is 2 evaluations per eligible factor per option, so the
        delta cannot be smaller than that if the phase ran at all.
        """
        on = _flip_body(9, 3, 400, 3)
        off = {k: v for k, v in on.items() if k != "include_factor_flips"}

        RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**off))
        off_calls = count_evaluates.total

        count_evaluates.evaluate = 0
        count_evaluates.evaluate_multi = 0
        RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**on))
        on_calls = count_evaluates.total

        n_eligible, n_options = len(_eligible_driver_ids(on)), len(on["options"])
        screen_floor = 2 * n_eligible * n_options
        assert on_calls - off_calls >= screen_floor, (
            f"include_factor_flips added only {on_calls - off_calls} evaluations; the "
            f"candidate screen alone is 2*{n_eligible}*{n_options}={screen_floor}. The "
            f"phase did not run, so the flips oracle is VACUOUS — repair the shape, "
            f"never relax the assertion."
        )


class TestEvaluatorCallCountOracle:
    """The advertised price must never be less than the work performed."""

    def test_factor_flips_are_within_the_advertised_price(self, count_evaluates):
        """ROADMAP 2.745 — the term with ZERO pricing headroom, under the oracle.

        `W_FACTOR_FLIP_COEF = 1` charges the formula's own "1 evaluate = W units"
        convention exactly, so an error in this term cannot be absorbed by slack —
        and PLoT sends `include_factor_flips: true` UNCONDITIONALLY
        (`plot-lite-service` `src/integrations/isl/translator-v3.ts:913`, verified at
        `staging` 38bc3826 on 2026-08-07 — `:295` is only the optional interface
        field), so it is priced on every live request. Before this test, it was the
        one always-on priced phase the oracle never ran.
        """
        body = _flip_body(9, 3, 400, 3)
        advertised, actual, W = _run(body, count_evaluates)
        assert advertised >= actual, (
            f"UNDER-PRICED by {actual - advertised} units "
            f"({(actual / advertised - 1) * 100:.1f}%): advertised={advertised}, "
            f"actual={actual} ({count_evaluates.total} evaluate() calls x W={W}). "
            f"The factor-flip phase performs more evaluate() work than the "
            f"factor_flips term charges for."
        )

    def test_level_frame_shape_actually_runs_the_status_quo_evaluator(self, count_evaluates):
        """POSITIVE CONTROL (trap 13) — before asserting the level-framed shape is
        UNDER-priced, prove the shape does the extra work at all.

        A delta-framed twin of the same graph is the comparator: the ONLY
        difference is the frame, so any gap between their evaluate() counts is
        the status-quo series and nothing else. Without this control, a resolver
        that quietly refused the level threshold would make the undercharge test
        below pass by testing nothing.
        """
        level = _level_framed_body(12, 20, 400, 3)
        RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**level))
        level_calls = count_evaluates.total

        count_evaluates.evaluate = 0
        count_evaluates.evaluate_multi = 0
        delta = dict(level)
        delta["goal_threshold_frame"] = "delta"
        RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**delta))
        delta_calls = count_evaluates.total

        assert level_calls - delta_calls == level["n_samples"], (
            f"the level-framed shape did not run one extra evaluation per draw "
            f"(level={level_calls}, delta={delta_calls}, S={level['n_samples']}). "
            f"Either the resolver refused the threshold — in which case this "
            f"file's status-quo assertions are VACUOUS and the shape must be "
            f"repaired, not the assertion relaxed — or the phase moved."
        )

    def test_status_quo_evaluation_is_within_the_advertised_price(self, count_evaluates):
        """RED at pristine: base_mc charges S*O*W; the request performs
        (S*O + S) evaluations, so it is under-priced by exactly S*W."""
        body = _level_framed_body(12, 20, 400, 3)
        advertised, actual, W = _run(body, count_evaluates)
        assert advertised >= actual, (
            f"UNDER-PRICED by {actual - advertised} units "
            f"({(actual / advertised - 1) * 100:.1f}%): advertised={advertised}, "
            f"actual={actual} ({count_evaluates.total} evaluate() calls x W={W}). "
            f"The per-draw status-quo reference (ROADMAP 2.286) is not in the "
            f"cost formula."
        )

    def test_alternative_winner_sweep_is_within_the_advertised_price(self, count_evaluates):
        """RED at pristine: the marginal-switch sweep spends (K+1)*O evaluations
        per fragile edge and the fragile set is not count-capped."""
        body = _fragile_edge_body(14, 26, 1000, 4)
        advertised, actual, W = _run(body, count_evaluates)
        assert advertised >= actual, (
            f"UNDER-PRICED by {actual - advertised} units "
            f"({(actual / advertised - 1) * 100:.1f}%): advertised={advertised}, "
            f"actual={actual} ({count_evaluates.total} evaluate() calls x W={W}). "
            f"_compute_alternative_winners runs an unpriced, uncapped "
            f"marginal-switch sweep per fragile edge."
        )

    @pytest.mark.parametrize("label,body", _ORACLE_SHAPES)
    def test_advertised_cost_upper_bounds_evaluator_calls(self, label, body, count_evaluates):
        """The oracle across representative AND boundary shapes."""
        advertised, actual, W = _run(body, count_evaluates)
        assert advertised >= actual, (
            f"[{label}] UNDER-PRICED by {actual - advertised} units: "
            f"advertised={advertised}, actual={actual} "
            f"({count_evaluates.total} evaluate() calls x W={W})"
        )
