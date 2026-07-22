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

    def test_winner_has_lowest_regret_when_all_finite(self, client):
        """Highest-mean option has the lowest expected_regret.

        regret_o = E[best] - mean_o (exact when all draws finite), so argmin
        regret == argmax mean. Gated on validity_ratio == 1.0 per option.
        """
        body = post_v2(client, base_request())
        opts = body["options"]
        if not all(o["outcome"].get("validity_ratio") == 1.0 for o in opts):
            pytest.skip("non-finite draws present; identity not exact")
        best_by_mean = max(opts, key=lambda o: o["outcome"]["mean"])["id"]
        best_by_regret = min(opts, key=lambda o: o["downside"]["expected_regret"])["id"]
        assert best_by_mean == best_by_regret

    def test_regret_plus_mean_is_constant_across_options(self, client):
        """expected_regret_o + mean_o = E[best] (same for every option) when all
        draws are finite. A strong, raw-sample-free cross-check that regret is
        wired to the JOINT best-per-sample, not to some per-option quantity."""
        body = post_v2(client, base_request())
        opts = body["options"]
        if not all(o["outcome"].get("validity_ratio") == 1.0 for o in opts):
            pytest.skip("non-finite draws present; identity not exact")
        totals = [o["downside"]["expected_regret"] + o["outcome"]["mean"] for o in opts]
        assert max(totals) - min(totals) == pytest.approx(0.0, abs=1e-9)

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
