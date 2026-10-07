"""
Sentry reporting contract (system S-H) — ISL half.

Rows run the REAL sentry-sdk against a real FastAPI app with an in-memory
transport, so "nothing user-authored is sent" is measured on the bytes the SDK
would put on the wire, not on a hand-built dict.

Each absence row has a precondition twin: the same request through the
pre-fix init options (FastAPI + Starlette integrations, PII off, a
before_send that only dropped auth headers) MUST carry the sentinel. An
absence that the harness could not have seen is not evidence.
"""

from __future__ import annotations

import json
import logging

from typing import Any, Dict, List

import pytest

sentry_sdk = pytest.importorskip("sentry_sdk")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sentry_sdk.transport import Transport  # noqa: E402

from src.utils.sentry_contract import (  # noqa: E402
    build_sentry_options,
    resolve_environment,
    resolve_release,
    scrub_breadcrumb,
    scrub_event,
)

SENTINEL = "SENTINEL-7c2e-acquire-northwind-for-40m"
SHA40 = "0123456789abcdef0123456789abcdef01234567"
DSN = "https://public@o0.ingest.sentry.io/0"


class _Collect(Transport):
    """Keeps every payload the SDK would send, serialised."""

    def __init__(self) -> None:
        super().__init__({"dsn": DSN})
        self.payloads: List[bytes] = []

    def capture_event(self, event: Dict[str, Any]) -> None:  # 1.x error path
        self.payloads.append(json.dumps(event, default=str).encode())

    def capture_envelope(self, envelope: Any) -> None:  # transactions / 2.x
        self.payloads.append(envelope.serialize())

    def joined(self) -> bytes:
        return b"\n".join(self.payloads)


def _app() -> FastAPI:
    app = FastAPI()

    @app.post("/v2/logs-a-label")
    async def logs_a_label(request: Request) -> Dict[str, Any]:
        graph = await request.json()
        # an f-string error log interpolating user content, as ISL does today
        # (e.g. structural_model_parser.py "Failed to evaluate equation '{equation}'")
        logging.getLogger("src.services.example").error(
            f"bad equation for {graph['nodes'][0]['label']}"
        )
        return {"ok": True}

    @app.post("/v2/boom")
    async def boom(request: Request) -> Dict[str, Any]:
        graph = await request.json()  # a frame local holding the user's graph
        raise RuntimeError(f"compute failed for {len(graph['nodes'])} nodes")

    return app


def _post_through_sdk(options: Dict[str, Any], path: str = "/v2/boom") -> _Collect:
    transport = _Collect()
    with sentry_sdk.init(transport=transport, **options):
        client = TestClient(_app(), raise_server_exceptions=False)
        client.post(
            path,
            json={"nodes": [{"id": "n1", "label": SENTINEL}], "brief": SENTINEL},
        )
        sentry_sdk.flush()
    return transport


def _pre_fix_options() -> Dict[str, Any]:
    """The exact options src/api/main.py passed before S-H (staging 09ea3000)."""
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    def before_send_filter(event: Any, hint: Any) -> Any:
        if "request" in event:
            headers = event["request"].get("headers", {})
            if isinstance(headers, dict):
                headers.pop("Authorization", None)
                headers.pop("X-API-Key", None)
                headers.pop("x-api-key", None)
        return event

    return {
        "dsn": DSN,
        "environment": "staging",
        "traces_sample_rate": 1.0,
        "integrations": [
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
        ],
        "before_send": before_send_filter,
        "release": "isl@0.1.0",
        "send_default_pii": False,
    }


def _contract_options() -> Dict[str, Any]:
    return build_sentry_options(
        dsn=DSN,
        sentry_environment="production",
        runtime_environment="staging",
        git_commit_sha=SHA40,
        traces_sample_rate=1.0,
        profiles_sample_rate=0.0,
    )


# ---------------------------------------------------------------------------
# End-to-end through the real SDK
# ---------------------------------------------------------------------------


def test_precondition_pre_fix_options_send_the_users_graph() -> None:
    sent = _post_through_sdk(_pre_fix_options())
    assert sent.payloads, "harness captured nothing"
    assert SENTINEL.encode() in sent.joined()


def test_contract_options_send_no_user_content_on_errors_or_transactions() -> None:
    sent = _post_through_sdk(_contract_options())
    blob = sent.joined()
    assert sent.payloads, "harness captured nothing"
    # control: the failure itself still reaches Sentry
    assert b"compute failed for 1 nodes" in blob
    assert SENTINEL.encode() not in blob


def test_precondition_pre_fix_options_turn_an_error_log_into_an_event_with_the_label() -> None:
    sent = _post_through_sdk(_pre_fix_options(), path="/v2/logs-a-label")
    assert SENTINEL.encode() in sent.joined()


def test_contract_options_send_no_log_text() -> None:
    sent = _post_through_sdk(_contract_options(), path="/v2/logs-a-label")
    assert SENTINEL.encode() not in sent.joined()
    # control: the request itself was traced (transaction sent), so the
    # harness was live for this path
    assert b"logs_a_label" in sent.joined() or b"logs-a-label" in sent.joined()


def test_contract_events_carry_service_environment_and_release() -> None:
    sent = _post_through_sdk(_contract_options())
    blob = sent.joined()
    assert b'"service": "isl"' in blob or b'"service":"isl"' in blob
    assert b'"environment": "production"' in blob or b'"environment":"production"' in blob
    assert SHA40.encode() in blob


# ---------------------------------------------------------------------------
# The filter, row by row (each claim mutable on its own)
# ---------------------------------------------------------------------------


def test_scrub_removes_request_body_query_cookies_and_credential_headers() -> None:
    event = {
        "request": {
            "url": f"http://isl/v2/robustness?label={SENTINEL}",
            "query_string": f"label={SENTINEL}",
            "data": {"nodes": [{"label": SENTINEL}]},
            "cookies": {"s": SENTINEL},
            "headers": {"X-API-Key": SENTINEL, "Authorization": SENTINEL, "X-Request-Id": "rid-1"},
        }
    }
    out = json.dumps(scrub_event(event))
    assert SENTINEL not in out
    assert "rid-1" in out and "http://isl/v2/robustness" in out


def test_scrub_removes_frame_local_variables() -> None:
    event = {
        "exception": {
            "values": [
                {
                    "type": "RuntimeError",
                    "stacktrace": {"frames": [{"function": "boom", "vars": {"graph": SENTINEL}}]},
                }
            ]
        }
    }
    out = json.dumps(scrub_event(event))
    assert SENTINEL not in out
    assert '"function": "boom"' in out


@pytest.mark.parametrize(
    "key",
    [
        "label",
        "node_labels",
        "goal_text",
        "brief",
        "equations",
        "query_params",
        "candidate",
        "result",
    ],
)
def test_extra_is_an_allowlist_and_custom_contexts_are_dropped(key: str) -> None:
    event = {
        "extra": {key: SENTINEL, "path": "/v2/CONTROL_PATH", "request_id": "CONTROL_RID"},
        "contexts": {
            "compute": {key: SENTINEL},
            "runtime": {"name": "CONTROL_RUNTIME"},
        },
    }
    out = json.dumps(scrub_event(event))
    assert SENTINEL not in out
    assert "CONTROL_PATH" in out and "CONTROL_RID" in out and "CONTROL_RUNTIME" in out


def test_scrub_drops_log_breadcrumbs_and_cuts_http_to_its_shape() -> None:
    event = {
        "breadcrumbs": {
            "values": [
                {"type": "log", "category": "src.services.x", "message": SENTINEL},
                {
                    "type": "http",
                    "category": "httplib",
                    "message": SENTINEL,
                    "data": {
                        "url": f"http://plot/v2/run?label={SENTINEL}",
                        "http.query": f"label={SENTINEL}",
                        "method": "POST",
                        "status_code": 503,
                    },
                },
            ]
        }
    }
    out = json.dumps(scrub_event(event))
    assert SENTINEL not in out
    assert "http://plot/v2/run" in out and "503" in out
    assert scrub_breadcrumb({"type": "log", "message": SENTINEL}) is None


def test_scrub_strips_queries_from_span_data_descriptions_and_transaction() -> None:
    event = {
        "type": "transaction",
        "transaction": f"/v2/robustness?label={SENTINEL}",
        "contexts": {"trace": {"data": {"url.full": f"http://isl/v2?label={SENTINEL}"}}},
        "spans": [
            {
                "description": f"GET http://plot/v2/run?label={SENTINEL}",
                "data": {
                    "url": f"http://plot/v2/run?label={SENTINEL}",
                    "http.query": f"label={SENTINEL}",
                    "http.fragment": SENTINEL,
                },
            }
        ],
    }
    out = json.dumps(scrub_event(event))
    assert SENTINEL not in out
    assert "http://plot/v2/run" in out and "/v2/robustness" in out


def test_release_never_falls_back_to_sdk_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_RELEASE", "isl@0.1.0")
    opts = build_sentry_options(
        dsn=DSN,
        sentry_environment=None,
        runtime_environment="staging",
        git_commit_sha="unknown",
        traces_sample_rate=0.0,
        profiles_sample_rate=0.0,
    )
    assert opts["release"] == "unidentified"


def test_scrub_redacts_transaction_span_data() -> None:
    event = {
        "type": "transaction",
        "contexts": {"trace": {"data": {"node_label": SENTINEL, "http.route": "CONTROL_ROUTE"}}},
        "spans": [{"data": {"goal_text": SENTINEL, "db.system": "CONTROL_DB"}}],
    }
    out = json.dumps(scrub_event(event))
    assert SENTINEL not in out
    assert "CONTROL_ROUTE" in out and "CONTROL_DB" in out


def test_scrub_tags_every_event_service_isl() -> None:
    assert scrub_event({})["tags"]["service"] == "isl"


# ---------------------------------------------------------------------------
# Environment and release
# ---------------------------------------------------------------------------


def test_sentry_environment_wins_over_runtime_label() -> None:
    assert resolve_environment("production", "staging") == "production"


def test_runtime_label_used_when_sentry_environment_unset_or_blank() -> None:
    assert resolve_environment(None, "staging") == "staging"
    assert resolve_environment("  ", "staging") == "staging"


def test_release_is_the_build_sha_or_unidentified() -> None:
    assert resolve_release(SHA40) == SHA40
    assert resolve_release("unknown") == "unidentified"


@pytest.mark.parametrize(
    "make",
    [
        lambda n: "GET http://plot/" + "a" * n,
        lambda n: "GET http://plot/x?" + "b" * n,
        lambda n: ("?a " * (n // 3 + 1))[:n],
    ],
)
def test_query_regex_scales_linearly(make: Any) -> None:
    import time

    def cost(n: int) -> float:
        text = make(n)
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(50):
                scrub_event({"transaction": text})
            best = min(best, time.perf_counter() - t0)
        return max(best, 5e-5)

    assert cost(20_000) / cost(5_000) < 8
