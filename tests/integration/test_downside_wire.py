"""B2 downside / tail-risk end-to-end wire tests (in-process TestClient).

Asserts the per-option ``downside`` object {cvar_10, p05, expected_regret}
reaches the V2 HTTP envelope, is gated by the emission rule (present EXACTLY
when the option's samples are available), respects the tail-ordering, and is
seed-deterministic. Additive-optional: absent (exclude_none) otherwise.

The regret+mean identity used below is exact ONLY when every draw is finite for
every option (validity_ratio == 1.0), so each such assertion is gated on that.
"""

import pytest

from fastapi.testclient import TestClient

from src.api.main import app

V2_URL = "/api/v1/robustness/analyze/v2"


@pytest.fixture
def client():
    return TestClient(app)


def base_request(**overrides):
    """3-node chain price -> demand -> revenue (+ price -> revenue), factor
    uncertainty on price. Two options with clearly different means so the
    downside metrics are non-degenerate."""
    request = {
        "graph": {
            "nodes": [
                {"id": "price", "kind": "factor", "label": "Price",
                 "observed_state": {"value": 0.5}},
                {"id": "demand", "kind": "chance", "label": "Demand"},
                {"id": "revenue", "kind": "outcome", "label": "Revenue"},
            ],
            "edges": [
                {"from": "price", "to": "demand", "exists_probability": 0.9,
                 "strength": {"mean": -0.6, "std": 0.1}},
                {"from": "demand", "to": "revenue", "exists_probability": 0.95,
                 "strength": {"mean": 0.8, "std": 0.1}},
                {"from": "price", "to": "revenue", "exists_probability": 0.8,
                 "strength": {"mean": 0.5, "std": 0.15}},
            ],
        },
        "options": [
            {"id": "low", "label": "Low price", "interventions": {"price": 0.3}},
            {"id": "high", "label": "High price", "interventions": {"price": 0.7}},
        ],
        "goal_node_id": "revenue",
        "n_samples": 400,
        "seed": 42,
    }
    request.update(overrides)
    return request


def near_equivalent_request(**overrides):
    """base_request's graph with two NEAR-EQUIVALENT options (price 0.50 vs
    0.51). Their true outcomes are almost identical, so the CRN decision regret
    is tiny -- which is what makes the post-noise inflation glaring (~80x): the
    max-over-independently-noised premium swamps the real regret."""
    request = base_request()
    request["options"] = [
        {"id": "low", "label": "Price 0.50", "interventions": {"price": 0.50}},
        {"id": "high", "label": "Price 0.51", "interventions": {"price": 0.51}},
    ]
    request.update(overrides)
    return request


def post_v2(client, request):
    resp = client.post(f"{V2_URL}?response_version=2", json=request)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestDownsideOnWire:
    def test_downside_present_and_shaped_per_option(self, client):
        body = post_v2(client, base_request())
        assert body.get("options"), "expected per-option results on the wire"
        for opt in body["options"]:
            d = opt.get("downside")
            assert d is not None, f"downside must be present for option {opt['id']}"
            assert set(d.keys()) == {"cvar_10", "p05", "expected_regret"}
            assert all(isinstance(d[k], (int, float)) for k in d)

    def test_emission_rule_iff_samples_available(self, client):
        """downside present EXACTLY when outcome.percentiles_source == 'samples'."""
        body = post_v2(client, base_request())
        for opt in body["options"]:
            has_downside = "downside" in opt
            has_samples = opt["outcome"].get("percentiles_source") == "samples"
            assert has_downside == has_samples, (
                f"emission rule violated for {opt['id']}: downside={has_downside}, "
                f"percentiles_source={opt['outcome'].get('percentiles_source')}"
            )

    def test_tail_ordering_on_wire(self, client):
        """cvar_10 <= p10 <= p50 — the THEOREM chain only (adversarial F1).

        cvar_10 <= p05 is deliberately NOT asserted here: it is NOT a theorem
        (a plateau in the lower tail inverts it — proven live-reachable via a
        point-mass-heavy chance-goal graph, cvar_10=0.4727 > p05=0.0, HTTP 200,
        values correct). Asserting the non-theorem on wire output would mis-fire
        RED on a legitimate future engine change (broken-alarm class). The full
        chain for the continuous MC regime is covered on seeded fixtures in
        test_downside_metrics.py.
        """
        body = post_v2(client, base_request())
        for opt in body["options"]:
            d = opt["downside"]
            o = opt["outcome"]
            assert d["cvar_10"] <= o["p10"] <= o["p50"], (
                opt["id"], d, o["p10"], o["p50"])
            assert d["p05"] <= o["p10"], (opt["id"], d, o["p10"])

    def test_p05_wire_value_pin(self, client):
        """Wire-exact p05 VALUE pin (adversarial O2): the emission constant has
        no other wire-exact guard — a 5→4 typo in the np.percentile call would
        pass every other test. The wire does not emit raw samples, so an
        in-test recompute is impossible without vacuity; instead pin the
        seed-42 deterministic values (captured from this exact request via an
        independent in-process run, module __file__ pinned to this tree).
        Mutation-proven: percentile 5→4 REDs this pin; determinism of the
        seeded request is covered by the existing determinism tests.
        """
        expected_p05 = {
            "low": -0.23213546863616197,
            "high": -0.48139036523706824,
        }
        body = post_v2(client, base_request())
        for opt in body["options"]:
            assert opt["downside"]["p05"] == pytest.approx(
                expected_p05[opt["id"]], rel=1e-12
            ), (opt["id"], opt["downside"]["p05"])

    def test_expected_regret_nonneg(self, client):
        body = post_v2(client, base_request())
        for opt in body["options"]:
            assert opt["downside"]["expected_regret"] >= 0.0

    def test_expected_regret_argmin_is_max_mean_utility(self):
        """The HONEST identity `argmin(expected_regret) == argmax(mean utility)`.

        F9 (A3, 2026-07-22): this REPLACES the false invariant
        argmin(regret)==argmax(win_probability). That equality is NOT a theorem
        (see the divergence test below). What IS a theorem — directly from the
        metric's own definition `regret_o = mean_i(best_i − o_i)` — is that when
        every draw is finite, `regret_o = E[best] − mean_o` with E[best] a single
        constant shared by all options; therefore the option with the LOWEST
        expected regret is exactly the one with the HIGHEST mean utility.

        Verified on the production metric (`expected_regret_per_option`, the exact
        code that produces the wire `downside.expected_regret`) over CRN-aligned
        PRE-noise per-option sample populations — the same population semantics as
        the fixture — not from the disagreeing scalar summaries.
        """
        import numpy as np

        from src.utils.downside import expected_regret_per_option

        rng = np.random.default_rng(42)
        n = 4000
        # Three CRN-aligned options (same n draws) with clearly separated means.
        base = rng.normal(0.0, 1.0, n)  # shared latent draw (CRN)
        samples = {
            "a": (base + rng.normal(0.0, 0.3, n) + 2.0).tolist(),  # highest mean
            "b": (base + rng.normal(0.0, 0.3, n) + 0.5).tolist(),
            "c": (base + rng.normal(0.0, 0.3, n) - 1.0).tolist(),  # lowest mean
        }
        regrets = expected_regret_per_option(samples)
        means = {k: float(np.mean(v)) for k, v in samples.items()}

        least_regret = min(regrets, key=lambda k: regrets[k])
        highest_mean = max(means, key=lambda k: means[k])
        assert least_regret == highest_mean == "a", (regrets, means)

    def test_regret_and_win_probability_may_legitimately_diverge(self):
        """Codex's counterexample: argmin(regret) and argmax(win_probability) can
        pick DIFFERENT options — proving the old equality invariant was never a
        theorem, and guarding against ever re-imposing it on the engine.

        Option A pays 100 on 49% of the CRN draws and 0 otherwise; option B pays 1
        on every draw. Then B wins the MOST samples (51% > 49%) so
        argmax(win_probability) = B, while A carries the LOWEST expected regret
        (E[regret_A]=0.51 vs E[regret_B]=48.51) so argmin(regret) = A. The two
        honest metrics answer "which is best?" differently, and both are correct
        for their own question.

        RED-first framing: on THIS data the old assertion
        `argmax(win_prob) == argmin(regret)` is `B == A` -> FALSE. If anyone ever
        made the engine force regret to track win_probability (re-imposing the
        false equality), the two would agree here and this divergence assertion
        would go RED.
        """
        import numpy as np

        from src.utils.downside import expected_regret_per_option

        n = 100
        a = [100.0] * 49 + [0.0] * 51  # 49% pay 100, else 0
        b = [1.0] * n  # always pays 1
        samples = {"A": a, "B": b}

        # win_probability: per-sample argmax winner (production semantics; no ties
        # here since A in {100, 0} and B == 1 are never equal).
        mat = np.vstack([np.asarray(a), np.asarray(b)])  # rows: A, B
        winners = mat.argmax(axis=0)  # 0 -> A wins, 1 -> B wins
        win_prob = {"A": float(np.mean(winners == 0)), "B": float(np.mean(winners == 1))}
        regrets = expected_regret_per_option(samples)

        best_by_winprob = max(win_prob, key=lambda k: win_prob[k])
        least_regret = min(regrets, key=lambda k: regrets[k])

        assert best_by_winprob == "B", win_prob
        assert least_regret == "A", regrets
        # The metrics DIVERGE — the old invariant would fail here.
        assert best_by_winprob != least_regret, (best_by_winprob, least_regret)

    def test_expected_regret_is_pre_noise_crn_value(self, client):
        """expected_regret is the PRE-noise CRN-joint regret, NOT the post-noise
        inflated value (F1 fix / CODE-REVIEW-ISL P1).

        The goal node 'revenue' is kind='outcome', so ``_apply_auto_scaled_noise``
        fires and adds INDEPENDENT per-option noise to the outcome samples,
        breaking the CRN alignment the metric's own docstring relies on. The
        JOINT expected_regret must be computed from the PRE-noise (CRN-aligned)
        population -- the same samples that produced win_probability.

        Seed-42/n=400 wire-exact pins (deterministic; captured from an
        independent pre-noise analyzer run, module __file__ pinned to this tree).
        BEFORE the fix the wire emitted the POST-noise values
        (low=0.11147263695963129, high=0.10749866011865553) -- ~2.5x these -- so
        this test is RED on the pre-fix code and GREEN after. cvar_10/p05 stay on
        the noised samples and are UNCHANGED by this fix (test_p05_wire_value_pin).
        """
        expected_regret = {
            "low": 0.04396019818411503,
            "high": 0.05405519879844565,
        }
        pre_fix_noised_bug = {  # what the pre-fix (post-noise) code emitted
            "low": 0.11147263695963129,
            "high": 0.10749866011865553,
        }
        body = post_v2(client, base_request())
        for opt in body["options"]:
            oid = opt["id"]
            r = opt["downside"]["expected_regret"]
            # Positive control (trap #13): the emitted value is materially
            # different from the pre-fix noised value -> the assertion can SEE
            # the correction, it is not vacuously green.
            assert abs(r - pre_fix_noised_bug[oid]) > 1e-2, (oid, r)
            assert r == pytest.approx(expected_regret[oid], rel=1e-12), (oid, r)

    def test_expected_regret_not_noise_inflated_wide_margin(self, client):
        """Wide-margin discriminator: two NEAR-EQUIVALENT options on a noised
        (outcome) goal. The true CRN decision regret is ~0.001; the pre-fix
        post-noise code inflated it ~80x to ~0.09 -- a pure max-over-independent-
        noise artifact for options that are causally near-identical. Pins the
        PRE-noise values and asserts the emitted regret is NOT in the noised
        band. RED on the pre-fix code (emits ~0.09), GREEN after.
        """
        expected_regret = {
            "low": 0.0010990049546028757,
            "high": 0.0013513799699611433,
        }
        pre_fix_noised_bug = {"low": 0.0965537399, "high": 0.0899453578}
        body = post_v2(client, near_equivalent_request())
        for opt in body["options"]:
            oid = opt["id"]
            r = opt["downside"]["expected_regret"]
            # Positive control: pre-fix emitted the ~80x-inflated value.
            assert abs(r - pre_fix_noised_bug[oid]) > 5e-2, (oid, r)
            # The honest regret is < 0.01; anything in the noised band (~0.09) fails.
            assert r < 0.01, (oid, r, "looks noise-inflated")
            assert r == pytest.approx(expected_regret[oid], rel=1e-12), (oid, r)

    def test_downside_seed_deterministic(self, client):
        """Same request + seed → identical downside values."""
        b1 = post_v2(client, base_request(seed=7))
        b2 = post_v2(client, base_request(seed=7))
        d1 = {o["id"]: o["downside"] for o in b1["options"]}
        d2 = {o["id"]: o["downside"] for o in b2["options"]}
        assert d1 == d2

    def test_existing_outcome_values_unchanged_additive(self, client):
        """Adding downside must not perturb the existing outcome stats: mean/std/
        p10/p50/p90 are still present and finite (additive-only guarantee)."""
        body = post_v2(client, base_request())
        for opt in body["options"]:
            o = opt["outcome"]
            for k in ("mean", "std", "p10", "p50", "p90"):
                assert isinstance(o[k], (int, float))


class TestDownsideOffloadSurvival:
    """Regret must survive the ProcessPoolExecutor OFFLOAD boundary (CODE-REVIEW
    F1 fix-of-fix). The analyzer runs inside the worker; run_offloaded returns the
    response as model_dump_json() and the endpoint reconstructs it with
    model_validate_json(). The pre-noise regret carrier MUST be a serialized field
    — a PrivateAttr is dropped across that boundary, so downside would be silently
    omitted on offloaded requests while present in in-process tests (test/prod
    divergence). These tests fail if the carrier ever reverts to a PrivateAttr.
    """

    def test_pre_noise_regret_survives_model_dump_validate_roundtrip(self):
        """The exact dump/validate boundary run_offloaded uses. A PrivateAttr
        carrier would come back None here (RED); a serialized field survives."""
        from src.models.robustness_v2 import OptionResult, OutcomeDistribution

        od = OutcomeDistribution(
            mean=1.0, std=0.5, median=1.0, ci_lower=0.2, ci_upper=1.8,
            samples=[0.1, 0.2, 0.3],
        )
        r = OptionResult(option_id="low", outcome_distribution=od, win_probability=0.5)
        r.pre_noise_expected_regret = 0.0439601982
        r2 = OptionResult.model_validate_json(r.model_dump_json())
        assert r2.pre_noise_expected_regret == pytest.approx(0.0439601982, rel=1e-12), (
            "pre_noise_expected_regret did not survive the offload dump/validate "
            "boundary — it must be a serialized field, not a PrivateAttr"
        )

    def test_worker_entry_emits_pre_noise_regret_across_boundary(self):
        """End-to-end via the EXACT ProcessPoolExecutor entry (run_robustness_v2:
        JSON in → JSON out). After reconstruction the results carry the correct
        PRE-noise regret (not the ~2.5x noised value), so an offloaded request
        builds downside identically to an in-process one."""
        import json as _json

        import src.services.robustness_worker as worker
        from src.models.robustness_v2 import RobustnessResponseV2

        out_json = worker.run_robustness_v2(_json.dumps(base_request()))
        resp = RobustnessResponseV2.model_validate_json(out_json)
        by_id = {r.option_id: r.pre_noise_expected_regret for r in resp.results}
        # PRE-noise targets (seed 42); the noised/pre-fix values were ~0.111/0.107.
        assert by_id["low"] == pytest.approx(0.0439601982, rel=1e-9), by_id
        assert by_id["high"] == pytest.approx(0.0540551988, rel=1e-9), by_id
        for v in by_id.values():
            assert v is not None
