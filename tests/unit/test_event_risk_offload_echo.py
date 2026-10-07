"""event_risk.v1 echo must survive the real worker-to-parent boundary."""

import json
from types import SimpleNamespace

import pytest
from fastapi.encoders import jsonable_encoder
from httpx import ASGITransport, AsyncClient

from src.models.robustness_v2 import RobustnessRequestV2, RobustnessResponseV2
from src.services.analysis_pool import create_analysis_pool, run_offloaded
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.services.robustness_worker import (
    decode_analysis_response,
    encode_analysis_response,
    run_robustness_v2,
)
from tests.unit.test_event_risk_v1 import supplier_request


@pytest.fixture(scope="module")
def event_request():
    return RobustnessRequestV2(**supplier_request(n_samples=500))


@pytest.fixture(scope="module")
def event_response(event_request):
    return RobustnessAnalyzerV2().analyze(event_request)


@pytest.fixture(scope="module")
def legacy_response():
    return RobustnessAnalyzerV2().analyze(
        RobustnessRequestV2(**supplier_request(event=False, n_samples=500))
    )


def echo_values(response):
    echo = getattr(response, "event_risks_applied", None)
    assert echo is not None, "event_risks_applied was lost at the worker boundary"
    return [(e.node_id, e.occurrence_used, e.p_low, e.p_high) for e in echo]


EXPECTED_ECHO = [("supplier_fails", 0.10, 0.05, 0.15)]


class TestEventRiskOffloadEcho:
    @pytest.mark.parametrize("transport", ["dump_validate", "worker_codec"])
    def test_dump_validate(self, event_response, transport):
        if transport == "dump_validate":
            restored = RobustnessResponseV2.model_validate_json(event_response.model_dump_json())
        else:
            restored = decode_analysis_response(encode_analysis_response(event_response))
        assert echo_values(restored) == EXPECTED_ECHO
        assert echo_values(restored) == echo_values(event_response)

    def test_worker_entry(self, event_request):
        restored = decode_analysis_response(run_robustness_v2(event_request.model_dump_json()))
        assert echo_values(restored) == EXPECTED_ECHO

    @pytest.mark.asyncio
    async def test_real_pool(self, event_request, monkeypatch):
        from src.services import analysis_pool

        def unexpected_fallback(*args, **kwargs):
            pytest.fail("real-pool test fell back to in-process analysis")

        monkeypatch.setattr(analysis_pool, "_run_in_process", unexpected_fallback)
        app = SimpleNamespace(state=SimpleNamespace(analysis_pool=create_analysis_pool(1)))
        try:
            restored = await run_offloaded(app, event_request, "rid")
            assert echo_values(restored) == EXPECTED_ECHO
        finally:
            app.state.analysis_pool.shutdown(wait=True, cancel_futures=True)

    def test_legacy_worker(self, legacy_response):
        assert legacy_response.event_risks_applied is None
        assert json.loads(encode_analysis_response(legacy_response)).get("event_risks_applied") is None

    @pytest.mark.asyncio
    async def test_route_with_real_pool(self, event_request, monkeypatch):
        from src.api.main import app
        from src.services import analysis_pool

        def unexpected_fallback(*args, **kwargs):
            pytest.fail("route test fell back to in-process analysis")

        monkeypatch.setattr(analysis_pool, "_run_in_process", unexpected_fallback)
        pool = create_analysis_pool(1)
        monkeypatch.setattr(app.state, "analysis_pool", pool, raising=False)
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/robustness/analyze/v2?response_version=2",
                    json=event_request.model_dump(mode="json", by_alias=True),
                )
            assert response.status_code == 200, response.text
            echo = response.json().get("event_risks_applied")
            assert echo is not None, "V2 envelope lost the offloaded event-risk echo"
            assert [(e["node_id"], e["occurrence_used"], e["p_low"], e["p_high"]) for e in echo] == EXPECTED_ECHO
        finally:
            app.state.analysis_pool.shutdown(wait=True, cancel_futures=True)

    @pytest.mark.asyncio
    async def test_legacy_v1_bytes(self, legacy_response, monkeypatch):
        from src.api import robustness
        from src.api.main import app

        async def fixed_analysis(*args, **kwargs):
            return legacy_response, None

        monkeypatch.setattr(robustness, "_admit_and_run", fixed_analysis)
        # The old PrivateAttr was absent; preserve every other key, including nulls.
        old_wire = robustness._non_finite_to_null(
            jsonable_encoder(legacy_response, exclude={"event_risks_applied"})
        )
        expected = json.dumps(old_wire, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/robustness/analyze/v2?response_version=1",
                json=supplier_request(event=False, n_samples=500),
            )
        assert response.status_code == 200, response.text
        assert response.content == expected

    @pytest.mark.asyncio
    async def test_legacy_v2_omits_echo(self, monkeypatch):
        from src.api.main import app

        monkeypatch.setattr(app.state, "analysis_pool", None, raising=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/robustness/analyze/v2?response_version=2",
                json=supplier_request(event=False, n_samples=500),
            )
        assert response.status_code == 200, response.text
        assert "event_risks_applied" not in response.json()
