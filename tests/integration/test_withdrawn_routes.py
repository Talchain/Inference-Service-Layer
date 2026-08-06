"""
ROADMAP 2.704 — quarantine pins for withdrawn (fabricating) ISL capabilities.

Two dispositions are pinned here, and they need different kinds of evidence:

REFUSED routes must answer a typed 501 **from the handler**, so that mounting
the router later still cannot serve a number. Testing that requires MOUNTING
the dark router — a test that merely observed "the route 404s in production"
would prove only that the ``include_router`` line is commented out, which is
the state we are defending against being changed. So every refusal test builds
its own app, mounts the real dark router, and drives the real handler.

DELETED routes must be gone from the import surface with no dangling
reference. An absence assertion is vacuous unless the instrument can see a
presence, so every scan here is paired with a positive control that finds a
symbol known to exist.

Binding note (trap 19): each refusal asserts the route's OWN reason constant,
not merely "some 501". A mutant that restores fabrication in a DIFFERENT
handler must not RED another handler's pin, and the reason-identity assertion
is what makes that true.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from src.api import withdrawn
from src.models.responses import ErrorCode

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"


# ---------------------------------------------------------------------------
# The withdrawal register.
#
# (module, router attribute, mount prefix, path, production route id, reason)
#
# The production route id is what the handler passes to refuse_withdrawn and
# what lands in the metric label; asserting it here means a copy-paste of a
# handler into a neighbouring module cannot silently mislabel its telemetry.
# ---------------------------------------------------------------------------
WITHDRAWN_ROUTES = [
    (
        "src.api.analysis",
        "router",
        "/api/v1/analysis",
        "/sensitivity",
        "/api/v1/analysis/sensitivity",
        withdrawn.DEFAULT_OUTCOME_REASON,
    ),
    (
        "src.api.cee",
        "router",
        "/api/v1",
        "/sensitivity/detailed",
        "/api/v1/sensitivity/detailed",
        withdrawn.STAMPED_CONSTANTS_REASON,
    ),
    (
        "src.api.cee",
        "router",
        "/api/v1",
        "/contrastive",
        "/api/v1/contrastive",
        withdrawn.INVENTED_FEASIBILITY_REASON,
    ),
    (
        "src.api.cee",
        "router",
        "/api/v1",
        "/conformal",
        "/api/v1/conformal",
        withdrawn.ASSERTED_COVERAGE_REASON,
    ),
    # NOTE: cee `/validation/strategies` is deliberately ABSENT from this list.
    # The 2026-08-07 triage recorded all four cee routes as fabrication traps;
    # re-derived at the bytes for 2.704 that does not hold for this one, which
    # delegates to a real suggester and publishes no numeric estimate. It is
    # pinned as a positive control below instead — so if anyone later withdraws
    # it on the strength of the blanket claim, a test says why not to.
    (
        "src.api.teaching",
        "router",
        "/api/v1/teaching",
        "/teach",
        "/api/v1/teaching/teach",
        withdrawn.RANDOM_OUTCOMES_REASON,
    ),
    (
        "src.api.team",
        "router",
        "/api/v1/team",
        "/align",
        "/api/v1/team/align",
        withdrawn.KEYWORD_MATCH_REASON,
    ),
    (
        "src.api.preferences",
        "router",
        "/api/v1/preferences",
        "/update",
        "/api/v1/preferences/update",
        withdrawn.FABRICATED_QUERY_REASON,
    ),
    (
        "src.api.causal",
        "router",
        "/api/v1/causal",
        "/discover/from-data",
        "/api/v1/causal/discover/from-data",
        withdrawn.INDEX_DIRECTION_REASON,
    ),
    (
        "src.api.causal",
        "router",
        "/api/v1/causal",
        "/sensitivity/detailed",
        "/api/v1/causal/sensitivity/detailed",
        withdrawn.DEFAULT_OUTCOME_REASON,
    ),
]

# Live siblings on the SAME dark router as a withdrawn route.
#
# These are the positive controls. Each proves two things at once: the harness
# really reaches handlers on this router (so a 501 assertion is not passing
# because nothing was mounted), and the withdrawal is bound to the named ROUTE
# rather than applied to the whole router.
NOT_WITHDRAWN_SIBLINGS = [
    ("src.api.analysis", "router", "/api/v1/analysis", "/optimise"),
    ("src.api.preferences", "router", "/api/v1/preferences", "/elicit"),
    ("src.api.causal", "router", "/api/v1/causal", "/discover/from-knowledge"),
    # cee `/validation/strategies` — three of its four sibling routes ARE
    # withdrawn, so this is the sharpest control in the file: it proves the
    # refusal was applied per-route on evidence, not swept across a module.
    ("src.api.cee", "router", "/api/v1", "/validation/strategies"),
]


def _mount(module_name: str, router_attr: str, prefix: str) -> FastAPI:
    """Build an app around a real (production-dark) router.

    Production exception handlers are attached so the response envelope is the
    one callers would actually receive, not FastAPI's default.
    """
    from src.api import main as isl_main

    module = importlib.import_module(module_name)
    app = FastAPI()
    app.include_router(getattr(module, router_attr), prefix=prefix)
    app.add_exception_handler(HTTPException, isl_main.http_exception_handler)
    app.add_exception_handler(Exception, isl_main.global_exception_handler)
    return app


class TestWithdrawnRoutesRefuse:
    """Every withdrawn route answers a typed 501 from its own handler."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "module_name,router_attr,prefix,path,route_id,reason", WITHDRAWN_ROUTES
    )
    async def test_withdrawn_route_returns_typed_501(
        self, module_name, router_attr, prefix, path, route_id, reason
    ):
        app = _mount(module_name, router_attr, prefix)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(prefix + path, json={})

        assert response.status_code == 501, (
            f"{route_id} must refuse with 501, got {response.status_code}: "
            f"{response.text[:400]}"
        )
        body = response.json()
        assert body["code"] == ErrorCode.CAPABILITY_WITHDRAWN.value
        assert body["retryable"] is False
        # Identity binding: this route's OWN reason, not merely "a refusal".
        assert body["reason"] == "capability_withdrawn"
        assert reason in body["message"], (
            f"{route_id} must cite its own fabrication reason. Expected "
            f"{reason!r} in message, got {body['message']!r}"
        )
        assert route_id in body["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "module_name,router_attr,prefix,path,route_id,reason", WITHDRAWN_ROUTES
    )
    async def test_withdrawn_route_refuses_before_validation(
        self, module_name, router_attr, prefix, path, route_id, reason
    ):
        """A garbage body still gets 501, never 422.

        No input is "valid" for a capability that does not exist; answering 422
        would imply that a well-formed body would have been computed. This is
        also what makes the guarantee unconditional: there is no request shape
        that reaches fabricating code.
        """
        app = _mount(module_name, router_attr, prefix)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                prefix + path, json={"totally": ["un", "expected"], "shape": 12345}
            )

        assert response.status_code == 501, (
            f"{route_id} must refuse regardless of body shape, got "
            f"{response.status_code}"
        )
        assert response.json()["code"] == ErrorCode.CAPABILITY_WITHDRAWN.value

    @pytest.mark.asyncio
    @pytest.mark.parametrize("module_name,router_attr,prefix,path", NOT_WITHDRAWN_SIBLINGS)
    async def test_positive_control_sibling_route_is_not_withdrawn(
        self, module_name, router_attr, prefix, path
    ):
        """POSITIVE CONTROL — the harness can see a non-refusal.

        Without this, every 501 assertion above could pass on a harness that
        refused everything (or on one that mounted nothing and 404'd). The
        sibling shares a router with a withdrawn route, so a blanket
        router-level refusal would be caught here.
        """
        app = _mount(module_name, router_attr, prefix)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(prefix + path, json={})

        assert response.status_code != 501, (
            f"{prefix}{path} is NOT withdrawn and must not answer 501; a "
            f"router-wide refusal has leaked onto a live sibling"
        )
        assert response.status_code != 404, (
            f"{prefix}{path} did not mount — the refusal harness is not "
            f"reaching handlers, so the 501 assertions prove nothing"
        )


class TestWithdrawnHelperContract:
    """The refusal helper's own wire contract."""

    def test_capability_withdrawn_code_is_distinct(self):
        """The withdrawal code must not collide with any failure code.

        A withdrawn capability is not a request that failed; a caller must be
        able to tell "this will never work" from "this attempt did not work".
        """
        assert ErrorCode.CAPABILITY_WITHDRAWN.value == "ISL_CAPABILITY_WITHDRAWN"
        others = [c.value for c in ErrorCode if c is not ErrorCode.CAPABILITY_WITHDRAWN]
        assert ErrorCode.CAPABILITY_WITHDRAWN.value not in others

    def test_every_reason_constant_is_distinct_and_cites_evidence(self):
        """Reasons must be specific enough to be worth reading.

        A generic "not implemented" string would let two different
        fabrications share one description, which is how a register goes
        stale. Each names its own defect and cites where it was established.
        """
        reasons = [
            withdrawn.STAMPED_CONSTANTS_REASON,
            withdrawn.INVENTED_FEASIBILITY_REASON,
            withdrawn.RANDOM_OUTCOMES_REASON,
            withdrawn.KEYWORD_MATCH_REASON,
            withdrawn.FABRICATED_QUERY_REASON,
            withdrawn.DEFAULT_OUTCOME_REASON,
            withdrawn.INDEX_DIRECTION_REASON,
        ]
        assert len(set(reasons)) == len(reasons), "reason constants must be distinct"
        for reason in reasons:
            assert "triage 2026-08-07" in reason, f"reason must cite evidence: {reason!r}"

    def test_refusal_records_caller_telemetry(self):
        """Telemetry is why these are refused rather than deleted.

        A 404 would answer the probe and record nothing, so we would never
        learn that something upstream had started calling a withdrawn route.
        """
        from src.utils.business_metrics import capability_withdrawn_refusals_total

        route = "/api/v1/team/align"

        def _count() -> float:
            for metric in capability_withdrawn_refusals_total.collect():
                for sample in metric.samples:
                    if sample.labels.get("route") == route and sample.name.endswith(
                        "_total"
                    ):
                        return sample.value
            return 0.0

        before = _count()
        withdrawn.refuse_withdrawn(route, withdrawn.KEYWORD_MATCH_REASON)
        assert _count() == before + 1, "refusal must increment its caller counter"


class TestRetiredModulesAreGone:
    """DELETED capabilities: absent from the import surface, no dangling refs."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "src.api.deliberation",
            "src.api.decision_robustness",
            "src.services.decision_robustness_analyzer",
            "src.services.conditional_recommender",
        ],
    )
    def test_retired_module_is_not_importable(self, module_name):
        assert importlib.util.find_spec(module_name) is None, (
            f"{module_name} was retired under ROADMAP 2.704 but is still "
            f"importable — a wiring job could still mount it"
        )

    def test_positive_control_a_live_module_is_importable(self):
        """POSITIVE CONTROL for the scanner above.

        ``find_spec`` returning None proves absence only if it returns
        non-None for something that is present.
        """
        assert importlib.util.find_spec("src.api.robustness") is not None
        assert importlib.util.find_spec("src.services.sequential_decision") is not None

    def test_phase4_dark_router_object_is_gone_but_live_mount_remains(self):
        """The trio's router is deleted; the mounted sequential route is not.

        This is the binding that matters: a blanket deletion of phase4 would
        have taken the live A2 mount with it, and a blanket keep would have
        left the trio mountable.
        """
        phase4 = importlib.import_module("src.api.phase4")
        assert not hasattr(phase4, "router"), (
            "the dark phase4 `router` (conditional-recommend, policy-tree, "
            "stage-sensitivity) must be deleted"
        )
        assert hasattr(phase4, "sequential_router"), (
            "the LIVE mounted sequential router must survive the retirement"
        )

    def test_retired_engine_methods_are_gone(self):
        """The trio's engine entry points go with their routes.

        Left behind, they are exactly the "dormant code that looks usable"
        this lane exists to remove.
        """
        from src.services.sequential_decision import SequentialDecisionEngine

        for method in ("get_policy_tree", "stage_sensitivity"):
            assert not hasattr(SequentialDecisionEngine, method), (
                f"SequentialDecisionEngine.{method} was retired under 2.704"
            )
        # Positive control: the live entry point survives.
        assert hasattr(SequentialDecisionEngine, "analyze")

    def test_no_dangling_references_in_src(self):
        """No source file mentions a retired symbol.

        Scope: every ``*.py`` under ``src/`` except ``src/services/_archived/``
        (a deliberate archive, excluded by design and asserted non-empty below
        so this exclusion cannot silently swallow the whole tree).
        """
        retired_symbols = [
            "deliberation_orchestrator",
            "decision_robustness_analyzer",
            "conditional_recommender",
            "ConditionalRecommendationEngine",
            "decision_robustness_router",
            "deliberation_router",
            "phase4_router",
        ]
        scanned = [
            p
            for p in SRC_ROOT.rglob("*.py")
            if "_archived" not in p.parts
        ]
        assert len(scanned) > 100, (
            f"scanner found only {len(scanned)} files — scope is wrong, so a "
            f"clean result would prove nothing"
        )

        offenders = []
        for path in scanned:
            text = path.read_text(encoding="utf-8", errors="replace")
            for symbol in retired_symbols:
                if symbol in text:
                    offenders.append(f"{path.relative_to(SRC_ROOT)}: {symbol}")
        assert not offenders, "dangling references to retired symbols: " + "; ".join(
            offenders
        )

    def test_positive_control_scanner_finds_a_present_symbol(self):
        """POSITIVE CONTROL for the dangling-reference scan.

        The scan above reports absence. If its file list or its matching were
        broken it would report absence too, so prove it can find something.
        """
        scanned = [
            p for p in SRC_ROOT.rglob("*.py") if "_archived" not in p.parts
        ]
        hits = [
            p
            for p in scanned
            if "SequentialDecisionEngine" in p.read_text(encoding="utf-8", errors="replace")
        ]
        assert hits, "scanner cannot find a symbol that is definitely present"

    def test_main_does_not_reference_retired_routers(self):
        """Even the commented rollback lines must go.

        A commented ``include_router`` for a deleted module is an instruction
        to reintroduce a crash: ``deliberation`` imports a service that lives
        only under ``_archived/``, so uncommenting it fails at import.
        """
        main_src = (SRC_ROOT / "api" / "main.py").read_text(encoding="utf-8")
        for symbol in (
            "deliberation_router",
            "decision_robustness_router",
            "phase4_router",
        ):
            assert symbol not in main_src, (
                f"src/api/main.py still names {symbol}; the rollback comment "
                f"points at a deleted module"
            )
        # Positive control: the live mounts are still named.
        assert "phase4_sequential_router" in main_src
        assert "causal_counterfactual_router" in main_src
