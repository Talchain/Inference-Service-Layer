"""Sampled edge strengths carry no +/-1 magnitude bound (AIQ ruling #72 5868664986).

A normalised strength's magnitude depends on the node's frame (cap): doubling a source's frame
halves its normalised values and doubles the strengths out of it, and the user-unit answer must
not move. The former +/-1 cut on DRAWS broke that: journey A's GBP59 effect fell 30% under a x2
frame on the served tuple, and a 0.85 +/- 0.2 edge lost 23% of its mass. Finite means are preserved too; optional flip searches disclose when their fixed domain cannot represent a coefficient. There is still no sign constraint, but
P(sign flip) now follows the stated normal: 0.159 for 0.5 +/- 0.5, where truncation gave 0.187.
The frame transform is EXACT under the unbounded law (normal(2m, 2s) consumes the same z as
normal(m, s)), so the frame rows assert equality: a +/-2 bound passed a 3-SE window (DL 5871402629).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import src.services.robustness_analyzer_v2 as analyzer_mod
from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import DualUncertaintySampler, RobustnessAnalyzerV2
from src.utils.rng import SeededRNG


def _strengths(mean: float, std: float, n: int = 20_000, seed: int = 42) -> np.ndarray:
    edges = [
        EdgeV2(
            **{"from": "a", "to": "b"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=mean, std=std),
        )
    ]
    configs = DualUncertaintySampler(edges, SeededRNG(seed)).sample_n_configurations(n)
    return np.array([c[("a", "b")] for c in configs])


class TestSampledStrengthLaw:
    def test_very_high_edge_keeps_its_mean(self):
        """Paul's 'very high' edge, 0.85 +/- 0.2: truncation at +1 pulled the mean to ~0.80."""
        s = _strengths(0.85, 0.2)
        se = 0.2 / np.sqrt(s.size)
        assert abs(s.mean() - 0.85) < 4 * se, s.mean()
        assert abs(s.std() - 0.2) < 0.01

    def test_draws_may_exceed_one_in_magnitude(self):
        s = _strengths(0.85, 0.2)
        assert (s > 1.0).mean() > 0.2  # P(N(0.85, 0.2) > 1) = 0.227

    def test_no_sign_constraint_draws_may_cross_zero(self):
        s = _strengths(0.1, 0.3)
        assert 0.3 < (s < 0).mean() < 0.4  # P(N(0.1, 0.3) < 0) = 0.369, as before

    def test_sign_flip_probability_follows_the_normal_not_the_truncation(self):
        """0.5 +/- 0.5: Phi(-1) = 0.1587 under the normal; the +/-1 truncation gave 0.1873."""
        s = _strengths(0.5, 0.5)
        p = float((s < 0).mean())
        se = float(np.sqrt(0.1587 * (1 - 0.1587) / s.size))
        assert abs(p - 0.1587) < 4 * se, p
        assert abs(p - 0.1873) > 8 * se, p

    def test_draws_are_finite(self):
        assert np.isfinite(_strengths(0.9, 5.0)).all()


N_SAMPLES = 4000


def _two_level_request(
    mean: float, std: float, scale: float, n: int = N_SAMPLES
) -> RobustnessRequestV2:
    """x -> y with strength N(mean, std); options set x to two levels. ``scale`` is the frame
    factor applied to x: its normalised levels are divided by ``scale`` and the strength out of
    it multiplied by ``scale``, so the user-unit problem is identical for every ``scale``."""
    nodes = [
        NodeV2(
            id="x",
            kind="factor",
            label="X",
            observed_state=ObservedState(value=0.2 / scale),
        ),
        NodeV2(id="y", kind="outcome", label="Y", observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(
            **{"from": "x", "to": "y"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=mean * scale, std=std * scale),
        )
    ]
    return RobustnessRequestV2(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(id="high", label="High", interventions={"x": 0.8 / scale}),
            InterventionOption(id="low", label="Low", interventions={"x": 0.2 / scale}),
        ],
        goal_node_id="y",
        seed=4242,
        n_samples=n,
    )


def _effect(req: RobustnessRequestV2) -> tuple[float, float]:
    r = RobustnessAnalyzerV2().analyze(req)
    out = {o.option_id: o.outcome_distribution for o in r.results}
    diff = out["high"].mean - out["low"].mean
    se = float(np.hypot(out["high"].std, out["low"].std) / np.sqrt(N_SAMPLES))
    return diff, se


class TestFrameInvariance:
    """AIQ's contract test F in miniature: doubling the source's frame must not move the answer."""

    def test_x2_frame_gives_the_same_effect(self):
        base, se_b = _effect(_two_level_request(0.45, 0.3, scale=1.0))
        doubled, se_d = _effect(_two_level_request(0.45, 0.3, scale=2.0))
        assert doubled == pytest.approx(base, rel=1e-12, abs=1e-12), (base, doubled)


class TestByteIdenticalWhenNoDrawReachesTheOldBound:
    """A graph whose draws never reach +/-1 must serialise exactly as under the former sampler."""

    def test_response_identical_to_truncated_sampler(self, monkeypatch):
        req = _two_level_request(0.3, 0.05, scale=1.0, n=1000)
        new = json.loads(RobustnessAnalyzerV2().analyze(req).model_dump_json())
        monkeypatch.setattr(
            analyzer_mod,
            "_sample_edge_strength",
            lambda rng, mean, std: rng.truncated_normal(mean, std, -1.0, 1.0),
        )
        old = json.loads(RobustnessAnalyzerV2().analyze(req).model_dump_json())
        for d in (new, old):  # per-call identity and timing, not science
            d.pop("request_id", None)
            d.get("metadata", {}).pop("execution_time_ms", None)
        assert new == old


@pytest.mark.parametrize("mean,std", [(0.85, 0.2), (-0.9, 0.5)])
def test_mutant_bound_restored_is_red(mean, std, monkeypatch):
    """The discriminating control: with the +/-1 bound restored the mean is biased (RED)."""
    monkeypatch.setattr(
        analyzer_mod,
        "_sample_edge_strength",
        lambda rng, m, s: rng.truncated_normal(m, s, -1.0, 1.0),
    )
    s = _strengths(mean, std)
    assert abs(s.mean() - mean) > 4 * std / np.sqrt(s.size)


# ---------------------------------------------------------------------------------------------------------
# AIQ contract test F on the real body: Paul's a6ed1bff graph as PLoT sends it (the B1a fixture). Doubling
# pro_plan_price's frame must leave the GBP59 effect in place (AIQ measured -30% under the truncated sampler).
# ---------------------------------------------------------------------------------------------------------

from tests.unit.test_anchored_delta_levels import (  # noqa: E402
    P59,
    analyse,
    effect_gbp,
    effect_se_gbp,
    served_wire,
    with_the_served_truncated_sampler,
)

PRICE_NODE = "pro_plan_price"


def _double_the_frame(d: dict, node_id: str) -> dict:
    """AIQ's F transform: the node's cap x2; its normalised value, stds and option levels /2; the
    strengths OUT of it x2 and INTO it /2. Raw figures, limits and the goal are unchanged."""
    d = json.loads(json.dumps(d))
    for n in d["graph"]["nodes"]:
        if n["id"] == node_id:
            os_ = n["observed_state"]
            os_["cap"] = os_["cap"] * 2
            for k in ("value", "std", "baseline"):
                if os_.get(k) is not None:
                    os_[k] = os_[k] / 2
    for e in d["graph"]["edges"]:
        src = e.get("from", e.get("from_"))
        if src == node_id:
            e["strength"] = {k: v * 2 for k, v in e["strength"].items()}
        if e["to"] == node_id:
            e["strength"] = {k: v / 2 for k, v in e["strength"].items()}
    for pu in d.get("parameter_uncertainties", []):
        if pu["node_id"] == node_id and pu.get("std") is not None:
            pu["std"] = pu["std"] / 2
    for o in d["options"]:
        iv = o.get("interventions", {})
        if node_id in iv:
            iv[node_id] = iv[node_id] / 2
    return d


def _f_effects(analyse_fn=analyse):
    base_resp = analyse_fn(served_wire())
    doubled_resp = analyse_fn(_double_the_frame(served_wire(), PRICE_NODE))
    return (
        effect_gbp(base_resp, P59),
        effect_se_gbp(base_resp, P59),
        effect_gbp(doubled_resp, P59),
        effect_se_gbp(doubled_resp, P59),
    )


class TestContractFOnPaulsBody:
    def test_price_frame_x2_leaves_the_59_effect_exactly_unchanged(self):
        base, se_b, doubled, se_d = _f_effects()
        assert doubled == pytest.approx(base, rel=1e-12, abs=1e-9), (base, doubled)

    def test_mutant_truncated_sampler_is_frame_dependent(self):
        """The discriminating control: AIQ's measured -30% under the served sampler."""
        base, se_b, doubled, se_d = _f_effects(
            lambda d: with_the_served_truncated_sampler(analyse, d)
        )
        assert abs(doubled - base) > 3 * np.hypot(se_b, se_d), (base, doubled)


# Class 2: the parse-time mean bound must not undo the unbounded sampled law.
@pytest.mark.parametrize("sign", [1, -1])
def test_coefficient_keeps_natural_units_through_parse_and_scm(sign):
    from src.services.robustness_analyzer_v2 import SCMEvaluatorV2

    # Source frame 2, target frame 3; beta = +/-2 target units per source unit.
    # Setting source to 2 units must produce +/-4 target units, not +/-3.
    mean = sign * 2 * 2 / 3
    edge = EdgeV2(**{"from": "x", "to": "y"}, exists_probability=1,
                  strength={"mean": mean, "std": 0.2 * 2 / 3})
    graph = GraphV2(nodes=[NodeV2(id="x", kind="factor", label="X"),
                          NodeV2(id="y", kind="outcome", label="Y")], edges=[edge])
    assert edge.strength.mean == mean
    assert edge.strength.std == pytest.approx(0.2 * 2 / 3)
    outcome = SCMEvaluatorV2(graph).evaluate(
        edge_strengths={("x", "y"): edge.strength.mean},
        interventions={"x": 2 / 2}, goal_node="y",
    )
    assert outcome * 3 == pytest.approx(sign * 4)
    assert graph.collect_parse_warnings() == []


@pytest.mark.parametrize("sign", [1, -1])
def test_frame_invariance_when_rescaled_coefficient_exceeds_one(sign):
    base, _ = _effect(_two_level_request(sign * 0.45, 0.1, scale=1, n=100))
    wider, _ = _effect(_two_level_request(sign * 0.45, 0.1, scale=4, n=100))
    assert wider == pytest.approx(base, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mean", [2, -2])
def test_large_coefficient_preserves_base_analysis_and_discloses_unavailable_search(mean, monkeypatch):
    req = _two_level_request(mean, 0.1, scale=1, n=100)
    req.include_e_values = True
    analyzer = RobustnessAnalyzerV2()

    def unsupported_search(*args, **kwargs):
        pytest.fail("The [-1,1] optional search must not run on an out-of-domain coefficient")

    monkeypatch.setattr(analyzer, "_compute_edge_e_values", unsupported_search)
    response = analyzer.analyze(req)
    assert len(response.results) == 2
    assert all(r.outcome_distribution.mean is not None for r in response.results)
    assert response.edge_e_values is None
    by_code = {w.code: w for w in response.inference_warnings}
    assert "STRENGTH_MEAN_CLAMPED" not in by_code
    for code in ("E_VALUES_UNAVAILABLE", "STABILITY_BANDS_UNAVAILABLE"):
        assert by_code[code].severity == "warning"
        assert by_code[code].detail["reason"] == "coefficient_outside_search_domain"


def test_in_range_coefficient_still_runs_optional_search(monkeypatch):
    req = _two_level_request(0.45, 0.1, scale=1, n=100)
    req.include_e_values = True
    analyzer = RobustnessAnalyzerV2()
    calls = []

    def supported_search(*args, **kwargs):
        calls.append(True)
        return []

    monkeypatch.setattr(analyzer, "_compute_edge_e_values", supported_search)
    response = analyzer.analyze(req)
    assert calls == [True]
    assert not any(w.code == "E_VALUES_UNAVAILABLE" for w in response.inference_warnings)
