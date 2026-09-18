"""Naming and absence-reason disclosure for defaulted inputs and VOI.

TWO DISCLOSURES, DELIBERATELY NAMED APART (CLAUDE.md trap 21). They answer
different questions and neither may be derived from the other:

* ``ROOT_NODE_DEFAULT_VALUE`` — "this ROOT node had no observed value and no
  ParameterUncertainty, so it was genuinely defaulted to 0.0". Its population
  is DISJOINT from ``factor_sensitivity``, whose membership is exactly the set
  of nodes that HAVE a ParameterUncertainty
  (``_compute_factor_sensitivity``: ``for uncertainty in
  request.parameter_uncertainties``). So the factor-scoped
  ``value_defaulted`` flag can never speak for these nodes, and this warning
  is the ONLY honest carrier of "the analysis had to guess this one".

  It already carried the raw ``node_id``. A raw identifier is not a name a
  person can act on, and every downstream consumer that wants to say WHICH
  input was guessed has to either drop the disclosure or invent a name. The
  label travels here, at the producer, because that is the only layer that
  holds it.

* ``FACTOR_EVPPI_NOT_COMPUTED`` — "per-factor value-of-information was not
  computed, and here is which precondition was unmet". The gate is a
  three-conjunct precondition ISL evaluates itself
  (``include_voi`` AND ``factor_sampler.has_uncertainties()`` AND
  pre-noise outcomes available). Absence previously carried NO signal at all:
  no code, no status field, and ``isl_analysis_status`` reads ``computed`` in
  both arms — so nothing downstream could tell "not computed" from
  "suppressed" from "genuinely nothing to learn", and any surface that spoke
  would have been guessing. ISL knows the reason; it states it rather than
  leaving a consumer to reconstruct one.

RED-first: every assertion below fails at pristine (7781ca4f).
"""

from typing import Any, Dict, List

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.models.response_v2 import InferenceWarning
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

# The genuinely-defaulted root: no observed_state, no ParameterUncertainty,
# not intervened by every option. Its label and its id are DELIBERATELY
# different strings so an assertion cannot pass on the wrong one.
DEFAULTED_ID = "factor_c"
DEFAULTED_LABEL = "Billing Migration Complexity"


def _graph() -> GraphV2:
    nodes = [
        NodeV2(
            id="factor_a",
            kind="factor",
            label="Factor A",
            observed_state=ObservedState(value=0.5),
        ),
        NodeV2(
            id="factor_b",
            kind="factor",
            label="Factor B",
            observed_state=ObservedState(value=0.3),
        ),
        NodeV2(
            id=DEFAULTED_ID,
            kind="factor",
            label=DEFAULTED_LABEL,
            observed_state=None,  # deliberately missing → genuinely defaulted
        ),
        NodeV2(
            id="outcome",
            kind="outcome",
            label="Revenue",
            observed_state=ObservedState(value=0.0),
        ),
    ]
    edges = [
        EdgeV2(
            **{"from": "factor_a"},
            to="outcome",
            strength=StrengthDistribution(mean=0.8, std=0.1),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "factor_b"},
            to="outcome",
            strength=StrengthDistribution(mean=0.3, std=0.2),
            exists_probability=0.9,
        ),
        EdgeV2(
            **{"from": DEFAULTED_ID},
            to="outcome",
            strength=StrengthDistribution(mean=0.5, std=0.1),
            exists_probability=0.9,
        ),
    ]
    return GraphV2(nodes=nodes, edges=edges)


def _request(
    *, include_voi: bool = False, include_uncertainties: bool = False
) -> RobustnessRequestV2:
    uncertainties = None
    if include_uncertainties:
        uncertainties = [
            ParameterUncertainty(node_id="factor_b", distribution="normal", std=0.15),
        ]
    return RobustnessRequestV2(
        graph=_graph(),
        options=[
            InterventionOption(id="opt_high", label="High", interventions={"factor_a": 0.9}),
            InterventionOption(id="opt_low", label="Low", interventions={"factor_a": 0.1}),
        ],
        goal_node_id="outcome",
        seed=12345,
        n_samples=200,
        parameter_uncertainties=uncertainties,
        include_voi=include_voi,
    )


def _warnings(response: Any, code: str) -> List[Any]:
    return [w for w in response.inference_warnings if w.code == code]


# ===========================================================================
# 1. ROOT_NODE_DEFAULT_VALUE names the input a person can act on
# ===========================================================================


class TestRootDefaultWarningNamesTheInput:
    def test_detail_carries_the_human_node_label(self) -> None:
        """The disclosure's whole purpose is "go and set THIS input". A raw id
        cannot be acted on, and no downstream layer holds the label."""
        response = RobustnessAnalyzerV2().analyze(_request())
        warns = _warnings(response, "ROOT_NODE_DEFAULT_VALUE")
        assert len(warns) == 1, f"expected exactly one root-default warning, got {len(warns)}"
        detail: Dict[str, Any] = warns[0].detail
        # Bound by IDENTITY (the node_id), never by a value predicate another
        # node could satisfy (CLAUDE.md trap 19).
        assert detail["node_id"] == DEFAULTED_ID
        assert detail["node_label"] == DEFAULTED_LABEL

    def test_message_names_the_label_and_not_the_raw_identifier(self) -> None:
        """PLoT echoes this message VERBATIM (producer-owned wording) and the UI
        refuses any note that names a raw identifier. If the message keeps the id
        the disclosure stays unrenderable however good the detail is."""
        response = RobustnessAnalyzerV2().analyze(_request())
        message = _warnings(response, "ROOT_NODE_DEFAULT_VALUE")[0].detail["message"]
        assert DEFAULTED_LABEL in message, message
        assert DEFAULTED_ID not in message, message

    def test_label_falls_back_to_the_id_rather_than_inventing_one(self) -> None:
        """An unlabelled node must degrade to the id, never to a manufactured
        name. Honest and ugly beats fluent and false."""
        graph = _graph()
        for node in graph.nodes:
            if node.id == DEFAULTED_ID:
                node.label = ""
        request = _request()
        request = request.model_copy(update={"graph": graph})
        response = RobustnessAnalyzerV2().analyze(request)
        detail = _warnings(response, "ROOT_NODE_DEFAULT_VALUE")[0].detail
        assert detail["node_label"] == DEFAULTED_ID
        assert detail["message"].count(DEFAULTED_ID) >= 1


# ===========================================================================
# 2. factor_evppi absence states WHICH precondition was unmet
# ===========================================================================


class TestFactorEvppiAbsenceIsDisclosed:
    def test_absence_carries_a_code_and_a_reason_when_voi_not_requested(self) -> None:
        """include_voi=False is a REQUEST-level choice ISL knows exactly. Saying
        so lets a surface state something true instead of nothing."""
        response = RobustnessAnalyzerV2().analyze(_request(include_voi=False))
        assert response.factor_evppi is None
        warns = _warnings(response, "FACTOR_EVPPI_NOT_COMPUTED")
        assert len(warns) == 1, "absence of factor_evppi must carry a disclosure"
        assert warns[0].detail["reason"] == "voi_not_requested"

    def test_absence_reason_distinguishes_no_uncertainties(self) -> None:
        """A DIFFERENT unmet conjunct must read as a DIFFERENT reason — a probe
        that returns the same answer for every input is not discriminating
        (CLAUDE.md trap 20)."""
        response = RobustnessAnalyzerV2().analyze(
            _request(include_voi=True, include_uncertainties=False)
        )
        assert response.factor_evppi is None
        warns = _warnings(response, "FACTOR_EVPPI_NOT_COMPUTED")
        assert len(warns) == 1
        assert warns[0].detail["reason"] == "no_parameter_uncertainties"

    def test_no_absence_disclosure_when_factor_evppi_is_computed(self) -> None:
        """PRECONDITION PINNED IN-TEST (CLAUDE.md trap 13b): this arm asserts the
        quantity was actually produced, so the absent warning is provably the
        code's doing and not the fixture failing to reach the estimator."""
        response = RobustnessAnalyzerV2().analyze(
            _request(include_voi=True, include_uncertainties=True)
        )
        assert response.factor_evppi is not None, (
            "fixture precondition failed — this arm proves nothing unless "
            "factor_evppi was genuinely computed"
        )
        assert _warnings(response, "FACTOR_EVPPI_NOT_COMPUTED") == []


# ===========================================================================
# 3. A DISCLOSURE MAY NOT DENY A REASON THE PAYLOAD BESIDE IT STATES
#
# Added 18 Sep 2026 on an independent review finding against this PR.
#
# `FACTOR_EVPPI_NOT_COMPUTED`'s last arm says "The reason is not known at this
# layer and has deliberately not been inferred." Its suppression list excluded
# only FACTOR_EVPPI_UNAVAILABLE, so on the path where every requested factor is
# dropped in-loop -- `failed` non-empty, `results` empty, `return None` WITHOUT
# raising -- FACTOR_EVPPI_PARTIAL was emitted STATING the reason (failed factor
# ids and a per-factor category) and this warning was emitted beside it DENYING
# the reason is known.
#
# The block's own governing rule is NEVER SYNTHESISE A REASON. Denying a stated
# one is that rule's mirror, and is worse than inventing one: it teaches a
# reader that nothing further is knowable while the answer sits in the same
# response object.
#
# THE PAIR IS THE POINT (CLAUDE.md trap 19). The first test alone would pass if
# the suppression were widened to swallow the disclosure entirely; the second
# proves the disclosure still fires when nothing else explains the absence. One
# without the other shows nothing.
# ===========================================================================


class TestAbsenceDisclosureNeverContradictsAPayloadThatExplainsIt:
    @staticmethod
    def _force_evppi_none(monkeypatch: Any, *, emit_partial: bool) -> None:
        """Make the estimator return None, optionally having stated WHY first.

        Patches the real seam rather than the outcome: `_compute_factor_evppi`
        receives `inference_warnings` and returns None when `results` is empty,
        which is exactly the reachable path the finding names.

        ⚠ CALLERS MUST PASS `include_uncertainties=True`. `_request`'s default
        is False, so without it `factor_sampler.has_uncertainties()` is false,
        the guard short-circuits on its SECOND conjunct with
        `no_parameter_uncertainties`, and the estimator -- and therefore this
        patch -- is never reached at all. Both tests below would then be
        measuring a completely different arm while looking like they passed
        through this one. CI caught it, on the PRECONDITION assertions rather
        than on the behaviour, which is the only reason it was visible: a test
        that does not pin its own precondition reports success about a path it
        never walked.
        """

        def _fake(
            self: Any,
            *_args: Any,
            inference_warnings: Any = None,
            **_kwargs: Any,
        ) -> None:
            if emit_partial and inference_warnings is not None:
                inference_warnings.append(
                    InferenceWarning(
                        code="FACTOR_EVPPI_PARTIAL",
                        field="factor_evppi",
                        detail={
                            "failed_factor_ids": ["f_demand"],
                            "message": "1 factor could not be estimated.",
                        },
                    )
                )
            return None

        monkeypatch.setattr(RobustnessAnalyzerV2, "_compute_factor_evppi", _fake)

    def test_not_computed_is_silent_when_partial_already_states_the_reason(
        self, monkeypatch: Any
    ) -> None:
        self._force_evppi_none(monkeypatch, emit_partial=True)
        response = RobustnessAnalyzerV2().analyze(
            _request(include_voi=True, include_uncertainties=True)
        )

        assert response.factor_evppi is None
        partial = _warnings(response, "FACTOR_EVPPI_PARTIAL")
        assert len(partial) == 1, "precondition: the payload must state a reason"

        not_computed = _warnings(response, "FACTOR_EVPPI_NOT_COMPUTED")
        assert not_computed == [], (
            "FACTOR_EVPPI_NOT_COMPUTED must not claim the reason is unknown "
            "while FACTOR_EVPPI_PARTIAL states it in the same payload"
        )

    def test_not_computed_still_fires_when_nothing_else_explains_the_absence(
        self, monkeypatch: Any
    ) -> None:
        """DISCRIMINATING CONTROL. Identical path, no FACTOR_EVPPI_PARTIAL --
        the disclosure MUST still speak, or the fix above has simply deleted an
        honest signal instead of removing a contradictory one."""
        self._force_evppi_none(monkeypatch, emit_partial=False)
        response = RobustnessAnalyzerV2().analyze(
            _request(include_voi=True, include_uncertainties=True)
        )

        assert response.factor_evppi is None
        assert _warnings(response, "FACTOR_EVPPI_PARTIAL") == []

        not_computed = _warnings(response, "FACTOR_EVPPI_NOT_COMPUTED")
        assert len(not_computed) == 1, "the honest absence disclosure must survive"
        assert not_computed[0].detail["reason"] == "estimator_returned_no_rows"
