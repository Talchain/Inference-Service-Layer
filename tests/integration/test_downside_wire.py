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

    def test_expected_regret_consistent_with_win_probability(self, client):
        """expected_regret and win_probability derive from the SAME pre-noise
        CRN population (F1 fix), so the option that wins the most samples
        (highest win_probability) also carries the lowest expected_regret --
        one consistent answer to "which option is best?".

        Pre-fix this FAILED on this exact fixture: win_probability is built from
        the PRE-noise winners and ranks 'low' highest (0.5025 > 0.4975), while
        expected_regret was computed from the POST-noise samples and ranked
        'high' lowest (0.1075 < 0.1115). The same response answered the same
        question two ways from two populations (CODE-REVIEW-ISL F1 aggravator).
        RED before the fix (low != high), GREEN after (low == low).

        NB: this REPLACES the old ``winner_has_lowest_regret_when_all_finite``
        check, which compared argmin(regret) to argmax(WIRE mean). The wire mean
        is POST-noise while the corrected regret is PRE-noise, so near-equivalent
        means that noise reorders would make that comparison mis-fire. The honest
        same-population winner is win_probability (also pre-noise), used here.
        """
        body = post_v2(client, base_request())
        opts = body["options"]
        best_by_winprob = max(opts, key=lambda o: o["win_probability"])["id"]
        least_regret = min(opts, key=lambda o: o["downside"]["expected_regret"])["id"]
        assert best_by_winprob == least_regret, (best_by_winprob, least_regret)

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
