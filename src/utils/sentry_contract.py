"""
Sentry reporting contract for ISL (system S-H, shared with CEE, PLoT and the UI).

Every Olumi service reports to Sentry on the same terms:

1. ``environment`` = ``SENTRY_ENVIRONMENT`` when set (blank counts as unset),
   else the service's runtime label (``ENVIRONMENT``). Production ISL runs with
   ``ENVIRONMENT=staging``, so without the override every production event
   would be labelled "staging".
2. ``release`` = the full build SHA (``GIT_COMMIT_SHA`` from ``src.config``,
   which reads Render's ``RENDER_GIT_COMMIT``), never the package version.
3. every event carries the tag ``service=isl``.
4. NO user decision content leaves the process. ISL's request bodies are
   graphs whose nodes carry the user's labels, so:

   - the SDK never reads request bodies (``max_request_body_size="never"``);
   - stack frames carry no local variables (``include_local_variables=False``);
     a frame's locals hold the parsed graph;
   - log records are neither breadcrumbs nor events (``LoggingIntegration``
     off): ISL log messages interpolate equations and labels, and they stay
     in the service's own redacted logs;
   - one filter, :func:`scrub_event`, runs on errors AND transactions and
     removes any request body, query, cookies, credential headers, frame
     variables and log breadcrumbs; ``extra`` and ``contexts`` are ALLOWLISTS
     (ISL's own operational keys / the SDK's runtime contexts); span data
     drops queries and redacts decision-content keys.

Known residual (reported, not closed here): the text of an exception message
is sent as written. A ``raise ValueError(f"... {label}")`` would carry a label.
"""

from __future__ import annotations

import re

from typing import Any, Dict, Mapping, MutableMapping, Optional

SERVICE_TAG = "isl"

REDACTED = "[Redacted]"

#: Header names (lower case) that carry credentials or session state.
SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-isl-api-key",
        "x-olumi-assist-key",
        "x-admin-key",
        "cookie",
        "set-cookie",
        "proxy-authorization",
    }
)

#: Key names whose values are user decision content or credentials, wherever
#: they appear in ``extra`` / ``contexts``. Substring match on the lower-cased
#: key, so ``node_labels`` and ``goal_text`` are both covered.
SENSITIVE_KEY_SNIPPETS = (
    "label",
    "brief",
    "prompt",
    "message",
    "statement",
    "headline",
    "goal_text",
    "equation",
    "graph",
    "nodes",
    "edges",
    "payload",
    "body",
    "content",
    "query_params",
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
)

#: ``extra`` keys ISL itself sets (main.py global handler + request_id). Every
#: other extra key is replaced by REDACTED: an allowlist, not a key-name guess.
ALLOWED_EXTRA_KEYS = frozenset({"request_id", "path", "method"})

#: Contexts the SDK fills with runtime facts. Any other context (set by code)
#: is dropped. ``trace`` is kept but its ``data`` goes through the span rules.
ALLOWED_CONTEXTS = frozenset(
    {"runtime", "os", "device", "app", "culture", "cloud_resource", "trace", "response", "profile"}
)

#: Request headers kept on an event (lower case). Every other header is
#: dropped: an allowlist, because any header can carry caller-chosen text.
ALLOWED_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "content-length",
        "content-type",
        "host",
        "referer",
        "user-agent",
        "x-request-id",
        "x-trace-id",
        "x-correlation-id",
    }
)

#: Dynamic-sampling-context keys kept. sentry-sdk copies an inbound
#: ``baggage`` header into the envelope's trace header AFTER before_send, from
#: ``contexts.trace.dynamic_sampling_context``; anything else in it is
#: caller-chosen text (e.g. sentry-transaction, sentry-user_segment).
ALLOWED_DSC_KEYS = frozenset(
    {"trace_id", "public_key", "sample_rate", "sampled", "environment", "release"}
)

#: Span ops whose description is SDK-generated (method + URL, middleware or
#: query name). Any other span keeps only its op as its description.
SAFE_DESCRIPTION_OP_PREFIXES = ("http.", "middleware.", "db", "cache", "redis", "subprocess")

#: Span attributes that ARE a query or fragment: dropped.
QUERY_SPAN_KEYS = frozenset({"http.query", "http.fragment", "url.query", "url.fragment"})

#: Span attributes holding a URL: kept with the query stripped.
URL_SPAN_KEYS = frozenset({"url", "http.url", "url.full", "http.target"})

_QUERY_IN_TEXT = re.compile(r"[?#]\S*")

_LONG_STRING = 200
_MAX_DEPTH = 8
_SHA_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$", re.IGNORECASE)


def resolve_environment(sentry_environment: Optional[str], runtime_environment: str) -> str:
    """SENTRY_ENVIRONMENT wins; blank counts as unset; else the runtime label."""
    explicit = (sentry_environment or "").strip()
    return explicit or runtime_environment


UNIDENTIFIED_RELEASE = "unidentified"


def resolve_release(git_commit_sha: str) -> str:
    """The full build SHA, else ``"unidentified"``.

    Never ``None``: sentry-sdk reads ``None`` as "infer a release" and picks up
    SENTRY_RELEASE / git / other env, which name something other than this build.
    """
    return git_commit_sha if _SHA_RE.match(git_commit_sha or "") else UNIDENTIFIED_RELEASE


def _is_sensitive_key(key: str) -> bool:
    lower = str(key).lower()
    return any(snippet in lower for snippet in SENSITIVE_KEY_SNIPPETS)


def _redact(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return REDACTED
    if isinstance(value, Mapping):
        return {
            k: (REDACTED if _is_sensitive_key(k) else _redact(v, depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v, depth + 1) for v in value]
    if isinstance(value, str) and len(value) > _LONG_STRING:
        return REDACTED
    return value


def _scrub_span_data(data: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in _redact(data).items():
        if key in QUERY_SPAN_KEYS:
            continue
        out[key] = _strip_query(value) if key in URL_SPAN_KEYS else value
    return out


def _strip_frame_vars(container: Any) -> None:
    """Remove ``vars`` from every stack frame under exception/threads values."""
    if not isinstance(container, Mapping):
        return
    for value in container.get("values") or []:
        if not isinstance(value, MutableMapping):
            continue
        stacktrace = value.get("stacktrace")
        if isinstance(stacktrace, Mapping):
            for frame in stacktrace.get("frames") or []:
                if isinstance(frame, MutableMapping):
                    frame.pop("vars", None)


def _strip_query(url: Any) -> Any:
    if not isinstance(url, str):
        return url
    for sep in ("?", "#"):
        url = url.split(sep, 1)[0]
    return url


def scrub_breadcrumb(crumb: Mapping[str, Any], hint: Any = None) -> Optional[Dict[str, Any]]:
    """A breadcrumb keeps only its SHAPE, never free text.

    Log and console crumbs are dropped. Every other crumb loses ``message`` and
    keeps only the HTTP data shape ``{method, url (query stripped), status_code}``
    (sentry-python also puts ``http.query`` / ``http.fragment`` in ``data``).
    """
    if crumb.get("type") == "log" or crumb.get("category") in ("console", "logging"):
        return None
    out: Dict[str, Any] = {
        k: crumb[k] for k in ("type", "category", "level", "timestamp") if k in crumb
    }
    data = crumb.get("data")
    if isinstance(data, Mapping):
        kept: Dict[str, Any] = {}
        if isinstance(data.get("method"), str):
            kept["method"] = data["method"]
        if "url" in data:
            kept["url"] = _strip_query(data["url"])
        if isinstance(data.get("status_code"), int):
            kept["status_code"] = data["status_code"]
        out["data"] = kept
    return out


def scrub_event(event: Dict[str, Any], hint: Any = None) -> Dict[str, Any]:
    """The single privacy filter for every ISL event type. Mutates and returns ``event``."""
    tags = event.setdefault("tags", {})
    if isinstance(tags, MutableMapping):
        tags["service"] = SERVICE_TAG

    request = event.get("request")
    if isinstance(request, MutableMapping):
        request.pop("data", None)
        request.pop("cookies", None)
        request.pop("query_string", None)
        if "url" in request:
            request["url"] = _strip_query(request["url"])
        headers = request.get("headers")
        if isinstance(headers, MutableMapping):
            for name in list(headers.keys()):
                lower = str(name).lower()
                if lower in SENSITIVE_HEADERS or lower not in ALLOWED_HEADERS:
                    del headers[name]
                elif lower == "referer":
                    headers[name] = _strip_query(headers[name])
        request.pop("env", None)

    _strip_frame_vars(event.get("exception"))
    _strip_frame_vars(event.get("threads"))

    extra = event.get("extra")
    if isinstance(extra, Mapping):
        event["extra"] = {
            k: (_redact(v) if k in ALLOWED_EXTRA_KEYS else REDACTED) for k, v in extra.items()
        }

    contexts = event.get("contexts")
    if isinstance(contexts, MutableMapping):
        for key in list(contexts.keys()):
            if key not in ALLOWED_CONTEXTS or _is_sensitive_key(key):
                del contexts[key]
            elif isinstance(contexts[key], Mapping):
                contexts[key] = _redact(contexts[key])

    if isinstance(event.get("transaction"), str):
        event["transaction"] = _QUERY_IN_TEXT.sub("", event["transaction"])

    def _crumbs(values: list) -> list:
        kept = (scrub_breadcrumb(b) if isinstance(b, Mapping) else None for b in values)
        return [b for b in kept if b is not None]

    breadcrumbs = event.get("breadcrumbs")
    if isinstance(breadcrumbs, MutableMapping) and isinstance(breadcrumbs.get("values"), list):
        breadcrumbs["values"] = _crumbs(breadcrumbs["values"])
    elif isinstance(breadcrumbs, list):
        event["breadcrumbs"] = _crumbs(breadcrumbs)

    # Span data on transactions: same key-class redaction as extra.
    trace = contexts.get("trace") if isinstance(contexts, Mapping) else None
    if isinstance(trace, MutableMapping) and isinstance(trace.get("data"), Mapping):
        trace["data"] = _scrub_span_data(trace["data"])
    if isinstance(trace, MutableMapping) and isinstance(
        trace.get("dynamic_sampling_context"), Mapping
    ):
        trace["dynamic_sampling_context"] = {
            k: v for k, v in trace["dynamic_sampling_context"].items() if k in ALLOWED_DSC_KEYS
        }
    for span in event.get("spans") or []:
        if not isinstance(span, MutableMapping):
            continue
        if isinstance(span.get("data"), Mapping):
            span["data"] = _scrub_span_data(span["data"])
        if isinstance(span.get("description"), str):
            op = str(span.get("op") or "")
            if op.startswith(SAFE_DESCRIPTION_OP_PREFIXES):
                span["description"] = _QUERY_IN_TEXT.sub("", span["description"])
            else:
                span["description"] = op or "span"

    return event


def build_sentry_options(
    *,
    dsn: str,
    sentry_environment: Optional[str],
    runtime_environment: str,
    git_commit_sha: str,
    traces_sample_rate: float,
    profiles_sample_rate: float,
) -> Dict[str, Any]:
    """The keyword arguments ISL passes to ``sentry_sdk.init``."""
    # Imported here: sentry-sdk is only needed when Sentry is enabled.
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    return {
        "dsn": dsn,
        "environment": resolve_environment(sentry_environment, runtime_environment),
        "release": resolve_release(git_commit_sha),
        "traces_sample_rate": traces_sample_rate,
        "profiles_sample_rate": profiles_sample_rate,
        "send_default_pii": False,
        "max_request_body_size": "never",
        "include_local_variables": False,
        "integrations": [
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
            LoggingIntegration(level=None, event_level=None),
        ],
        "before_send": scrub_event,
        "before_send_transaction": scrub_event,
        "before_breadcrumb": scrub_breadcrumb,
    }
