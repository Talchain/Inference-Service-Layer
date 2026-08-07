"""Contract step-2 slice 6b — the READER half of `constraint_id` adoption.

PLoT sends `goal_constraints[].constraint_id` on every /robustness/analyze/v2
request (plot-lite-service `integrations/isl/translator-v3.ts:543`, flat on each
goal-constraint member). ISL's `GoalConstraint` did not declare it, and every
model in the `RobustnessRequestV2` tree sets `extra: "ignore"`, so the key was
dropped at parse: present on the wire, absent from the parsed model, invisible
in both directions. PLoT therefore had to re-key ISL's constraint results
POSITIONALLY (`routes/v2/run.ts:1936-1949`: "ISL preserves insertion order and
does not echo constraint_id, so positional") and CEE's `constraint_verdict` has
an `identity_unresolved` state precisely because ratified constraint IDs have
zero overlap with those reconstructed keys.

Codex adjudicated OQ-5 as ADOPT-into-ISL, reader-first (not delete). This suite
pins the reader half: ISL ACCEPTS the optional id and ECHOES it back on the
corresponding result member, so a consumer can key results by the ratified ID
instead of reconstructing them.

Reader-first is load-bearing for sequencing: PLoT already emits this field, so
ISL must declare it BEFORE any strictness flip — under `extra='forbid'` a
producer-first field 422s.

Every assertion goes through the REAL endpoint, so it covers the request model,
the analyzer threading, the internal->V2 emission layer AND serialisation
(`by_alias=True, exclude_none=True`) end to end. A unit test on the model alone
would pass while the value was still dropped somewhere in the middle.

RED-first: before the declaration landed, tests 1-4 and 8 failed — `extra='ignore'`
dropped the key at parse, so no echo could exist anywhere downstream.
"""

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

# Two constraints on the SAME node with the SAME operator. This is the case
# positional reconstruction cannot disambiguate and the echo can: a consumer
# keying off (node_id, operator) collapses these two into one, which is the
# concrete defect `identity_unresolved` reports.
CID_LOW = "c:revenue-floor:9f2a"
CID_HIGH = "compiled:revenue"


def _build_request(constraint_ids=None):
    """Two goal constraints on 'revenue'. When `constraint_ids` is None the
    request omits `constraint_id` entirely — the pre-adoption shape, and the
    positive control for "nothing changes when the field is not sent"."""
    # ROADMAP 2.798: `value_frame` is the TRUTHFUL attestation for these
    # thresholds — they are stated in the samples' own frame, which is exactly
    # what 'delta' declares. Without it the constraint block is refused outright
    # and there would be no results for an id to be echoed onto.
    constraints = [
        {
            "node_id": "revenue",
            "operator": ">=",
            "value": 40.0,
            "label": "Revenue floor",
            "value_frame": "delta",
        },
        {
            "node_id": "revenue",
            "operator": ">=",
            "value": 90.0,
            "label": "Revenue stretch",
            "value_frame": "delta",
        },
    ]
    if constraint_ids is not None:
        for constraint, cid in zip(constraints, constraint_ids):
            constraint["constraint_id"] = cid

    return {
        "graph": {
            "nodes": [
                {"id": "price", "kind": "factor", "label": "Price"},
                {"id": "revenue", "kind": "goal", "label": "Revenue"},
            ],
            "edges": [
                {"from": "price", "to": "revenue", "strength": {"mean": 0.6, "std": 0.15}},
            ],
        },
        "options": [
            {"id": "opt1", "label": "Raise price", "interventions": {"price": 120}},
            {"id": "opt2", "label": "Lower price", "interventions": {"price": 80}},
        ],
        "goal_node_id": "revenue",
        "seed": 42,
        "n_samples": 200,
    } | {"goal_constraints": constraints}


def _constraints_for_first_option(client, request):
    """POST and return the first option's constraint_analysis.constraints list."""
    resp = client.post(ENDPOINT, json=request, headers=V2_HEADERS)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    options = data.get("options") or []
    assert options, f"expected option results, got keys {sorted(data)}"
    analysis = options[0].get("constraint_analysis")
    assert analysis is not None, f"expected constraint_analysis on {sorted(options[0])}"
    return analysis["constraints"]


class TestConstraintIdEcho:
    """The reader half: accept the optional id, echo it on the result."""

    def test_constraint_id_echoed_on_each_result(self, v2_client):
        """1. RED-first core: the supplied ids come back on the results, in order."""
        constraints = _constraints_for_first_option(v2_client, _build_request([CID_LOW, CID_HIGH]))
        assert [c.get("constraint_id") for c in constraints] == [CID_LOW, CID_HIGH]

    def test_echo_disambiguates_same_node_same_operator(self, v2_client):
        """2. The point of the field: two constraints identical but for their
        threshold get DISTINCT ids, which (node_id, operator) cannot supply."""
        constraints = _constraints_for_first_option(v2_client, _build_request([CID_LOW, CID_HIGH]))
        reconstructed = {(c["node_id"], c["operator"]) for c in constraints}
        echoed = {c["constraint_id"] for c in constraints}
        assert len(reconstructed) == 1, "precondition: positional key collapses these two"
        assert len(echoed) == 2, "the echo must distinguish what the positional key cannot"

    def test_echo_is_exact_not_normalised(self, v2_client):
        """3. The id is opaque: echoed byte-for-byte, not lower-cased, slugified
        or otherwise 'cleaned'. PLoT mints ids like `compiled:<nodeId>` and CEE
        supplies arbitrary caller ids; any normalisation breaks the join."""
        weird = "C:Revenue_Floor-2026/Q3 (ratified)"
        constraints = _constraints_for_first_option(v2_client, _build_request([weird, CID_HIGH]))
        assert constraints[0]["constraint_id"] == weird

    def test_echo_survives_every_option(self, v2_client):
        """4. The echo is per-constraint inside EVERY option's analysis, not
        only the first — PLoT keys results per option."""
        resp = v2_client.post(
            ENDPOINT, json=_build_request([CID_LOW, CID_HIGH]), headers=V2_HEADERS
        )
        assert resp.status_code == 200, resp.text
        options = resp.json()["options"]
        assert len(options) == 2, "precondition: both options present"
        for option in options:
            ids = [c.get("constraint_id") for c in option["constraint_analysis"]["constraints"]]
            assert ids == [CID_LOW, CID_HIGH], f"option {option.get('id')} lost the echo"

    def test_omitted_constraint_id_is_absent_not_null(self, v2_client):
        """5. POSITIVE CONTROL / exclude_none: a request that does NOT send the
        field gets results with the key OMITTED — not serialised as null. The
        pre-adoption wire shape is preserved exactly for producers that never
        send it."""
        constraints = _constraints_for_first_option(v2_client, _build_request())
        for constraint in constraints:
            assert "constraint_id" not in constraint

    def test_omitting_constraint_id_changes_nothing_else(self, v2_client):
        """6. POSITIVE CONTROL, whole-payload: the response with ids supplied is
        IDENTICAL to the response without them once the new optional field is
        removed. Proves the addition is inert — it cannot perturb sampling,
        probabilities or any other field. Per-request identifiers and wall-clock
        timings are excluded because they vary between any two calls; the
        comparator's ability to SEE a difference is asserted below, so this is
        not an equality that passes by comparing nothing."""
        with_ids = v2_client.post(
            ENDPOINT, json=_build_request([CID_LOW, CID_HIGH]), headers=V2_HEADERS
        )
        without = v2_client.post(ENDPOINT, json=_build_request(), headers=V2_HEADERS)
        assert with_ids.status_code == without.status_code == 200

        volatile = {"request_id", "timestamp", "processing_time_ms"}

        def canonical(payload):
            payload = {k: v for k, v in payload.items() if k not in volatile}
            for option in payload["options"]:
                analysis = option.get("constraint_analysis")
                if analysis:
                    for constraint in analysis["constraints"]:
                        constraint.pop("constraint_id", None)
            return payload

        left, right = canonical(with_ids.json()), canonical(without.json())
        assert left == right

        # The comparator must be able to fail: perturb one number and prove the
        # equality flips. Without this the assertion above could be comparing
        # two payloads it had emptied.
        perturbed = canonical(with_ids.json())
        perturbed["options"][0]["win_probability"] = -1.0
        assert perturbed != right, "comparator cannot see a difference — assertion is vacuous"

    def test_partial_ids_do_not_forge_the_missing_one(self, v2_client):
        """7. Mixed input: one constraint carries an id, the other does not.
        The unidentified one must stay unidentified rather than inherit its
        neighbour's id or be given a fabricated one — a forged identity is worse
        than an absent one, because a consumer cannot tell it is a guess."""
        request = _build_request()
        request["goal_constraints"][0]["constraint_id"] = CID_LOW
        constraints = _constraints_for_first_option(v2_client, request)
        assert constraints[0]["constraint_id"] == CID_LOW
        assert "constraint_id" not in constraints[1]

    def test_constraint_id_survives_the_offload_boundary(self, v2_client):
        """8. The analyzer may run in a ProcessPoolExecutor worker that returns
        the response via model_dump_json() and rebuilds it with
        model_validate_json(). The echo must be a DECLARED field on every model
        it passes through, or it is silently dropped on offloaded requests only
        — green in-process, absent in prod. Re-validating the emitted JSON
        through the response model reproduces that round-trip."""
        import json

        from src.models.response_v2 import ISLResponseV2

        resp = v2_client.post(
            ENDPOINT, json=_build_request([CID_LOW, CID_HIGH]), headers=V2_HEADERS
        )
        assert resp.status_code == 200, resp.text
        round_tripped = ISLResponseV2.model_validate_json(json.dumps(resp.json()))
        assert round_tripped.options is not None
        rebuilt = round_tripped.options[0].constraint_analysis
        assert rebuilt is not None
        assert [c.constraint_id for c in rebuilt.constraints] == [CID_LOW, CID_HIGH]

    def test_unknown_sibling_key_still_ignored(self, v2_client):
        """9. NEGATIVE CONTROL: declaring constraint_id must not turn the model
        strict. An undeclared sibling is still dropped by extra='ignore' rather
        than rejected — this slice adopts one field, it does not flip strictness
        (that is a separate, Codex-gated slice)."""
        request = _build_request([CID_LOW, CID_HIGH])
        request["goal_constraints"][0]["definitely_not_a_declared_field"] = "xyz"
        resp = v2_client.post(ENDPOINT, json=request, headers=V2_HEADERS)
        assert resp.status_code == 200, f"strictness changed: {resp.status_code} {resp.text}"
        constraints = _constraints_for_first_option(v2_client, request)
        assert "definitely_not_a_declared_field" not in constraints[0]
