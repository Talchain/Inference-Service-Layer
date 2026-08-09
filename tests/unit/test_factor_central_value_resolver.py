"""ROADMAP 2.1020 — one resolver for "the factor's central value".

THE DEFECT
----------
``_compute_factor_sensitivity`` and its bootstrap twin both did, in effect::

    mean_value = observed_value if observed_value is not None else 0.0

and never consumed the sampled draws. For a PRIOR-ONLY factor — one with a
stated prior range but no ``observed_state.value`` — that is 0.0, so for a
stated ``Uniform[0.6, 1.0]`` the sensitivity probe perturbed at -0.04 / +0.04
(BOTH outside the factor's own declared support) and divided by
``factor_denom = max(|0.0|, 0.01) = 0.01`` instead of 0.8. The factor's
reported elasticity was therefore ~80x suppressed: it read as near-zero
influence in "what matters most" and fell below the ``|elasticity| >= 0.01``
flip-candidate filter.

WHY A RESOLVER AND NOT A PATCH
------------------------------
The factor's central value was read INDEPENDENTLY at five sites (register row
2.520). Patching only the sensitivity site would ship an engine whose
SENSITIVITY disagreed with its own SAMPLER — a hand-maintained mirror inside a
single file (CLAUDE.md trap 12), and a worse defect than the original, because
the two halves would then be confidently inconsistent.

THE PRODUCER'S DECLARED SEMANTICS (derived at the bytes, not from opinion)
-------------------------------------------------------------------------
``resolve_factor_central_value`` is defined as **the expectation of the
distribution FactorSampler actually draws from**:

* ``uniform``   — ``FactorSampler._sample_from_distribution`` draws
  ``rng.uniform(range_min, range_max)`` and IGNORES ``mean`` entirely
  (``_copula_transform`` likewise). E[U(a,b)] = (a+b)/2, so the central value
  is the midpoint REGARDLESS of ``observed_state``.
* ``normal``    — draws ``rng.normal(mean, std)`` where mean is
  ``observed_state.value`` else 0.0. E = mean.
* ``point_mass``— returns ``mean`` exactly. E = mean.
* no uncertainty entry — the factor is never sampled; its value is
  ``observed_state.value`` else 0.0.

That definition is TESTABLE AGAINST THE PRODUCER rather than against a copy of
it: ``test_resolver_matches_the_sampler_it_claims_to_summarise`` asserts the
resolver equals the empirical mean of real FactorSampler draws. A resolver that
drifts from the sampler fails on the sampler's own output, not on a mirror.

Corroboration that a prior-only factor is NOT meant to be 0.0 comes from the
producer too: the ``ROOT_NODE_DEFAULT_VALUE`` warning (analyzer :1815-1831)
deliberately EXCLUDES nodes carrying a ``parameter_uncertainties`` entry, on
the stated grounds that such an entry "would provide sampling via
FactorSampler". The engine already declares these factors un-defaulted.

TRAP DISCIPLINE OBSERVED HERE
-----------------------------
* trap 19 (bind by IDENTITY): every behavioural assertion selects its factor by
  ``node_id``. The graph carries a SECOND factor, ``fac_obs``, whose elasticity
  is of the same order — so an assertion that merely found "a factor with a
  big elasticity" would pass on the wrong object. The discriminating mutant
  pair in the lane report exercises exactly that.
* trap 13 (an absence/property assertion needs a positive control): the
  "perturbs within the declared support" test asserts the spy SAW a non-zero
  number of perturbations before asserting anything about where they landed.
* trap 12d (a derived guard proves agreement, never completeness): the two
  conformance tests below are HAND-WRITTEN corpora, deliberately NOT derived
  from the resolver's own call list, so a NEW independent read fails them.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.constants import FACTOR_SENSITIVITY_VALUE_EPSILON
from src.models.robustness_v2 import ParameterUncertainty, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import SCMEvaluatorV2

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

SEED = 4242
N_SAMPLES = 400

# The flip-candidate filter the decision-review path applies downstream. A
# factor below this reads as "barely matters" and is never offered as a lever.
FLIP_CANDIDATE_ELASTICITY_FLOOR = 0.01

# StrengthDistribution.std is constrained gt=0.001, so a truly deterministic
# edge cannot be expressed; this is the smallest legal value. Every hand
# computation below uses mean * exists_probability only, which is std-free.
NEAR_ZERO_STD = 0.002

# --- identities under test -------------------------------------------------
FAC_PRIOR = "fac_prior"  # prior-only: Uniform[0.6, 1.0], NO observed_state
FAC_OBS = "fac_obs"  # control: observed_state.value = 0.8, Normal(std=0.05)
LEVER = "lever"  # both options intervene here (validator needs a target)
GOAL = "outcome"

PRIOR_MIN = 0.6
PRIOR_MAX = 1.0
PRIOR_MIDPOINT = 0.8  # E[U(0.6, 1.0)]
PRIOR_DELTA = (PRIOR_MAX - PRIOR_MIN) * 0.1  # 0.04 — the analyzer's uniform delta


def _node(node_id: str, kind: str, value: Optional[float] = None) -> Dict[str, Any]:
    node: Dict[str, Any] = {"id": node_id, "kind": kind, "label": node_id}
    if value is not None:
        node["observed_state"] = {"value": value}
    return node


def _edge(src: str, dst: str, mean: float) -> Dict[str, Any]:
    return {
        "from": src,
        "to": dst,
        "strength": {"mean": mean, "std": NEAR_ZERO_STD},
        "exists_probability": 1.0,
    }


def _request_dict() -> Dict[str, Any]:
    """The lane's graph. Every constant is load-bearing for a hand computation.

    SCM (epsilon noise is disabled before every post-MC structural analysis)::

        outcome = lever*0.6 + fac_prior*0.5 + fac_obs*0.4

    Reference option (options[0]) is ``opt_a``: do(lever = 1.0).

    HAND COMPUTATION for fac_prior's elasticity, delta = 0.04:

        outcome_high - outcome_low = 0.5 * (2 * 0.04)          = 0.04
        baseline_mean  = 0.6 + 0.5*E[U(.6,1.)] + 0.4*E[N(.8,.05)]
                       = 0.6 + 0.5*0.8 + 0.4*0.8               = 1.32
        pct_outcome_change = 0.04 / 1.32                       = 0.030303

        BEFORE (mean_value = 0.0):
            pct_factor_change = 0.08 / max(0, 0.01) = 8.0
            elasticity        = 0.030303 / 8.0      = 0.003788   <- BELOW the
                                                        0.01 flip-candidate floor
        AFTER  (mean_value = 0.8):
            pct_factor_change = 0.08 / 0.8          = 0.1
            elasticity        = 0.030303 / 0.1      = 0.30303    <- 80x higher

    The 80x is exactly ``PRIOR_MIDPOINT / FACTOR_SENSITIVITY_VALUE_EPSILON``
    and is independent of the baseline, which is MC-estimated and therefore
    carries sampling noise. Assertions below pin the exact quantities
    (perturbation points, resolver value) and bound the noisy one.
    """
    return {
        "request_id": "factor-central-value-2p1020",
        "graph": {
            "nodes": [
                _node(LEVER, "factor", 0.0),
                _node(FAC_PRIOR, "factor"),  # NO observed_state — prior only
                _node(FAC_OBS, "factor", 0.8),
                _node(GOAL, "goal"),
            ],
            "edges": [
                _edge(LEVER, GOAL, 0.6),
                _edge(FAC_PRIOR, GOAL, 0.5),
                _edge(FAC_OBS, GOAL, 0.4),
            ],
        },
        "options": [
            {"id": "opt_a", "label": "A", "interventions": {LEVER: 1.0}},
            {"id": "opt_b", "label": "B", "interventions": {LEVER: 0.0}},
        ],
        "goal_node_id": GOAL,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "include_factor_flips": True,
        "parameter_uncertainties": [
            {
                "node_id": FAC_PRIOR,
                "distribution": "uniform",
                "range_min": PRIOR_MIN,
                "range_max": PRIOR_MAX,
            },
            {"node_id": FAC_OBS, "distribution": "normal", "std": 0.05},
        ],
    }


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def response_body(client: TestClient) -> Dict[str, Any]:
    resp = client.post(ENDPOINT, json=_request_dict(), headers=V2_HEADERS)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()


def _factor(body: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """Select a factor BY IDENTITY. Never by a value predicate (trap 19)."""
    rows = body.get("factor_sensitivity") or []
    match = next((r for r in rows if r["node_id"] == node_id), None)
    assert match is not None, f"no factor '{node_id}' in {[r['node_id'] for r in rows]}"
    return match


# ===========================================================================
# 1. The resolver itself, checked against the PRODUCER it claims to summarise
# ===========================================================================


class TestResolverAgreesWithTheSampler:
    def test_prior_only_uniform_resolves_to_the_declared_midpoint(self) -> None:
        """A stated Uniform[0.6, 1.0] with no observed value centres on 0.8."""
        from src.services.robustness_analyzer_v2 import resolve_factor_central_value

        request = RobustnessRequestV2(**_request_dict())
        node = next(n for n in request.graph.nodes if n.id == FAC_PRIOR)
        uncertainty = next(
            u for u in (request.parameter_uncertainties or []) if u.node_id == FAC_PRIOR
        )

        resolved = resolve_factor_central_value(node, uncertainty)

        assert resolved.value == pytest.approx(PRIOR_MIDPOINT)
        assert resolved.source == "prior_midpoint"
        assert node.observed_state is None, "fixture precondition: this factor is prior-only"

    def test_observed_value_resolves_to_the_observed_value(self) -> None:
        from src.services.robustness_analyzer_v2 import resolve_factor_central_value

        request = RobustnessRequestV2(**_request_dict())
        node = next(n for n in request.graph.nodes if n.id == FAC_OBS)
        uncertainty = next(
            u for u in (request.parameter_uncertainties or []) if u.node_id == FAC_OBS
        )

        resolved = resolve_factor_central_value(node, uncertainty)

        assert resolved.value == pytest.approx(0.8)
        assert resolved.source == "observed_state"

    def test_no_observed_value_and_no_prior_range_defaults_to_zero(self) -> None:
        """The honest fallback survives: a normal factor with no observed value
        genuinely IS centred on 0.0, because that is what the sampler draws."""
        from src.services.robustness_analyzer_v2 import resolve_factor_central_value

        request = RobustnessRequestV2(**_request_dict())
        node = next(n for n in request.graph.nodes if n.id == FAC_PRIOR)
        resolved = resolve_factor_central_value(
            node, ParameterUncertainty(node_id=FAC_PRIOR, distribution="normal", std=0.05)
        )

        assert resolved.value == 0.0
        assert resolved.source == "default_zero"

    @pytest.mark.parametrize(
        "uncertainty_kwargs,observed,expected",
        [
            ({"distribution": "uniform", "range_min": 0.6, "range_max": 1.0}, None, 0.8),
            ({"distribution": "uniform", "range_min": 0.0, "range_max": 1.0}, None, 0.5),
            # A uniform factor's draws IGNORE observed_state, so the central
            # value must too — this is the site-agreement the lane exists for.
            ({"distribution": "uniform", "range_min": 0.2, "range_max": 0.4}, 0.9, 0.3),
            ({"distribution": "normal", "std": 0.05}, 0.7, 0.7),
            ({"distribution": "normal", "std": 0.05}, None, 0.0),
            ({"distribution": "point_mass"}, 0.42, 0.42),
            ({"distribution": "point_mass"}, None, 0.0),
        ],
    )
    def test_resolver_matches_the_sampler_it_claims_to_summarise(
        self, uncertainty_kwargs: Dict[str, Any], observed: Optional[float], expected: float
    ) -> None:
        """The resolver IS E[the distribution FactorSampler draws from].

        Checked against the real sampler's output — not against a restatement
        of the sampler's rules — so a resolver that drifts from the sampler
        fails on the sampler's own draws.
        """
        from src.services.robustness_analyzer_v2 import (
            FactorSampler,
            SeededRNG,
            resolve_factor_central_value,
        )

        node_dict = _node("probe", "factor", observed)
        request = RobustnessRequestV2(
            **{
                **_request_dict(),
                "graph": {
                    "nodes": [*_request_dict()["graph"]["nodes"], node_dict],
                    "edges": [*_request_dict()["graph"]["edges"], _edge("probe", GOAL, 0.1)],
                },
            }
        )
        node = next(n for n in request.graph.nodes if n.id == "probe")
        uncertainty = ParameterUncertainty(node_id="probe", **uncertainty_kwargs)

        resolved = resolve_factor_central_value(node, uncertainty)
        assert resolved.value == pytest.approx(expected)

        sampler = FactorSampler([node], [uncertainty], SeededRNG(SEED))
        draws = [sampler.sample_factor_values()["probe"] for _ in range(8000)]
        empirical_mean = sum(draws) / len(draws)

        # Tolerance covers MC error only; the point is agreement with the
        # producer, and a 0.0-vs-0.8 disagreement is 20x outside it.
        assert resolved.value == pytest.approx(empirical_mean, abs=0.02), (
            f"resolver says {resolved.value} but the sampler's own draws average "
            f"{empirical_mean} for {uncertainty_kwargs}"
        )


# ===========================================================================
# 2. The user-visible defect
# ===========================================================================


class TestPriorOnlyFactorIsNotSuppressed:
    def test_sensitivity_perturbs_within_the_declared_support(self, monkeypatch) -> None:
        """Both probe points must lie inside the factor's own stated range.

        BEFORE: perturbed at -0.04 and +0.04 — both outside [0.6, 1.0].

        Driven through ``RobustnessAnalyzerV2.analyze`` directly, NOT the route:
        the API offloads the analysis to a process pool (``compute_governor``),
        so an in-process monkeypatch never reaches the code that runs. That was
        measured, not assumed — the spy recorded zero calls via the route.

        ``include_factor_flips`` is OFF here on purpose. The flip probe
        deliberately evaluates the whole [FACTOR_VALUE_MIN, FACTOR_VALUE_MAX]
        domain to map the affine family — a different operation from centring a
        perturbation, and its 0.0 endpoint is legitimately outside this
        factor's support. Including it would make this assertion measure the
        wrong thing.
        """
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        request_dict = {**_request_dict(), "include_factor_flips": False}
        seen: List[float] = []
        original = SCMEvaluatorV2.evaluate

        def _spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            factor_values = kwargs.get("factor_values") or {}
            if FAC_PRIOR in factor_values:
                seen.append(factor_values[FAC_PRIOR])
            return original(self, *args, **kwargs)

        monkeypatch.setattr(SCMEvaluatorV2, "evaluate", _spy)

        RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**request_dict))

        # trap 13: prove the instrument can SEE a presence before asserting
        # anything about what it saw.
        assert seen, "spy observed no perturbation of fac_prior — instrument is blind"

        out_of_support = [v for v in seen if not (PRIOR_MIN <= v <= PRIOR_MAX)]
        assert not out_of_support, (
            f"{len(out_of_support)} of {len(seen)} perturbations of '{FAC_PRIOR}' fell "
            f"OUTSIDE its declared support [{PRIOR_MIN}, {PRIOR_MAX}]: "
            f"{sorted(set(out_of_support))[:8]}"
        )
        assert PRIOR_MIDPOINT - PRIOR_DELTA in seen or any(
            abs(v - (PRIOR_MIDPOINT - PRIOR_DELTA)) < 1e-9 for v in seen
        ), f"expected a probe at {PRIOR_MIDPOINT - PRIOR_DELTA}; saw {sorted(set(seen))[:8]}"
        assert any(
            abs(v - (PRIOR_MIDPOINT + PRIOR_DELTA)) < 1e-9 for v in seen
        ), f"expected a probe at {PRIOR_MIDPOINT + PRIOR_DELTA}; saw {sorted(set(seen))[:8]}"

    def test_prior_only_factor_survives_the_flip_candidate_filter(
        self, response_body: Dict[str, Any]
    ) -> None:
        """THE OUTCOME METRIC. Below 0.01 the factor is dropped from the
        decision-review flip candidates and reads as "barely matters"."""
        prior = _factor(response_body, FAC_PRIOR)
        assert abs(prior["elasticity"]) >= FLIP_CANDIDATE_ELASTICITY_FLOOR, (
            f"'{FAC_PRIOR}' elasticity {prior['elasticity']} is below the "
            f"{FLIP_CANDIDATE_ELASTICITY_FLOOR} flip-candidate floor"
        )

    def test_prior_only_elasticity_matches_the_hand_computation(
        self, response_body: Dict[str, Any]
    ) -> None:
        """Pins the magnitude, not just the sign of the inequality above.

        Suppression factor is exactly PRIOR_MIDPOINT / epsilon = 80.
        """
        prior = _factor(response_body, FAC_PRIOR)
        suppression = PRIOR_MIDPOINT / FACTOR_SENSITIVITY_VALUE_EPSILON
        assert suppression == pytest.approx(80.0)
        # 0.030303 / 0.1; baseline_mean is MC-estimated so allow 10%.
        assert prior["elasticity"] == pytest.approx(0.30303, rel=0.10)

    def test_observed_value_factor_is_unchanged(self, response_body: Dict[str, Any]) -> None:
        """DISCRIMINATION: the control factor carries an observed value, so the
        resolver must leave it exactly where it always was. A fix that simply
        inflated every factor would fail here."""
        obs = _factor(response_body, FAC_OBS)
        # 0.04/1.32 / (0.1/0.8) = 0.2424
        assert obs["elasticity"] == pytest.approx(0.24242, rel=0.10)

    def test_flip_row_current_value_is_within_the_declared_support(
        self, response_body: Dict[str, Any]
    ) -> None:
        """The flip row's ``current_value`` is the same quantity, published on
        the wire. Leaving it at 0.0 while sensitivity centres on 0.8 would make
        one response state two different current values for one factor."""
        rows = response_body.get("factor_flip_values") or []
        assert rows, "expected factor_flip_values rows (include_factor_flips=True)"
        row = next((r for r in rows if r["factor_id"] == FAC_PRIOR), None)
        assert row is not None, f"no flip row for '{FAC_PRIOR}' in {[r['factor_id'] for r in rows]}"
        assert row["current_value"] == pytest.approx(PRIOR_MIDPOINT)

    def test_value_defaulted_is_not_claimed_for_a_prior_backed_factor(
        self, response_body: Dict[str, Any]
    ) -> None:
        """``value_defaulted`` declares "no observed value was provided, so it
        fell back to 0.0". Once the value comes from the declared prior, that
        sentence is false and the flag must not be published."""
        prior = _factor(response_body, FAC_PRIOR)
        assert "value_defaulted" not in prior or prior["value_defaulted"] is False


# ===========================================================================
# 3. Site completeness — a HAND-WRITTEN corpus, deliberately not derived
# ===========================================================================
#
# trap 12d: deriving a guard from a list proves the copies agree, never that
# the list is right. These two rules are written by hand from a full AST census
# of the tree, so a NEW independent read fails them and has to be justified.

_SCANNED_FILES = ("src/services/robustness_analyzer_v2.py", "src/api/robustness.py")

# Functions permitted to read ``.observed_state.value`` AS A VALUE. Each entry
# is a DIFFERENT question from "what is this factor's central value", and the
# reason is recorded so adding one is a deliberate act (CLAUDE.md trap 21:
# name the concepts apart rather than aligning their defaults).
_VALUE_READ_ALLOWED = {
    "resolve_factor_central_value": "THE resolver — the sanctioned single site.",
    "evaluate": (
        "SCMEvaluatorV2 root-node BASE for one deterministic evaluation. A "
        "different concept with its own published doctrine (analyzer :3615, "
        ":3770); deliberately NOT folded into the central-value resolver."
    ),
    "evaluate_multi": "As `evaluate` — the multi-goal twin of the same base rule.",
    "_build_goal_node_disclosures": (
        "The GOAL node's observed value, for the goal-frame disclosure. Not a "
        "factor central value."
    ),
    "_resolve_threshold_in_sample_frame": (
        "The goal's observed_state.baseline/value for threshold frame "
        "conversion. Not a factor central value."
    ),
}

# Every function anywhere in the scanned files that mentions ``observed_state``
# at all. This is the blunt union assertion: it also catches the alias form
# (``obs = node.observed_state`` then ``obs.value``) that an attribute-chain
# scan walks straight past.
_OBSERVED_STATE_MENTION_MANIFEST = {
    # --- the resolver and its consumers -----------------------------------
    "resolve_factor_central_value",
    # --- different concepts, see _VALUE_READ_ALLOWED ----------------------
    "evaluate",
    "evaluate_multi",
    "_build_goal_node_disclosures",
    "_resolve_threshold_in_sample_frame",
    # --- presence / provenance checks (never a central value) -------------
    "analyze",  # ROOT_NODE_DEFAULT_VALUE eligibility
    "_compute_factor_flip_values",  # flip-row eligibility: does it carry data?
    "_compute_conditional_winners",  # reads observed_state.unit, not .value
    "_analyze_robustness_v2_legacy",  # counts nodes with a value, for logging
    "_analyze_robustness_v2_enhanced",  # Track S provenance echo (source/type)
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _functions_by_line(tree: ast.AST) -> List[Any]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _enclosing(funcs: List[Any], lineno: int) -> str:
    best = None
    for f in funcs:
        if f.lineno <= lineno <= (f.end_lineno or f.lineno):
            if best is None or f.lineno > best.lineno:
                best = f
    return best.name if best else "<module>"


def _parents(tree: ast.AST) -> Dict[int, Any]:
    table: Dict[int, Any] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            table[id(child)] = parent
    return table


class TestCentralValueSiteCompleteness:
    def test_only_the_resolver_reads_observed_state_value_as_a_factor_value(self) -> None:
        """RULE A. A new ``node.observed_state.value`` read that is not a mere
        presence check fails here until it is either routed through the
        resolver or justified as a different concept."""
        offenders: Dict[str, List[str]] = {}
        for rel in _SCANNED_FILES:
            path = _repo_root() / rel
            tree = ast.parse(path.read_text())
            funcs = _functions_by_line(tree)
            parents = _parents(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Attribute) and node.attr == "value"):
                    continue
                inner = node.value
                if not (isinstance(inner, ast.Attribute) and inner.attr == "observed_state"):
                    continue
                parent = parents.get(id(node))
                # `x.observed_state.value is not None` is a presence check, not
                # a central-value read — allowed anywhere.
                if isinstance(parent, ast.Compare) and any(
                    isinstance(op, (ast.Is, ast.IsNot)) for op in parent.ops
                ):
                    continue
                name = _enclosing(funcs, node.lineno)
                if name not in _VALUE_READ_ALLOWED:
                    offenders.setdefault(name, []).append(f"{rel}:{node.lineno}")

        assert not offenders, (
            "A factor's central value is read outside `resolve_factor_central_value`. "
            "Route it through the resolver, or add it to _VALUE_READ_ALLOWED with the "
            f"DIFFERENT question it answers. Offenders: {offenders}"
        )

    def test_no_unlisted_function_touches_observed_state(self) -> None:
        """RULE B. The union assertion. Catches the alias form that RULE A's
        attribute-chain scan cannot see, and any brand-new reader."""
        found: Dict[str, List[str]] = {}
        for rel in _SCANNED_FILES:
            path = _repo_root() / rel
            tree = ast.parse(path.read_text())
            funcs = _functions_by_line(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "observed_state"
                    and isinstance(node.ctx, ast.Load)
                ):
                    found.setdefault(_enclosing(funcs, node.lineno), []).append(
                        f"{rel}:{node.lineno}"
                    )

        # Positive control: the scan must be finding something at all, or an
        # empty diff would pass this test vacuously (trap 13).
        assert found, "AST scan found no observed_state readers at all — instrument is blind"

        unlisted = {k: v for k, v in found.items() if k not in _OBSERVED_STATE_MENTION_MANIFEST}
        assert not unlisted, (
            "New function(s) read observed_state. Decide deliberately whether this is "
            "the factor's central value (use the resolver) or a different question "
            f"(add to the manifest with a comment). Unlisted: {unlisted}"
        )
