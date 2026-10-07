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
     removes any request body, cookies, credential headers, frame variables,
     log breadcrumbs and decision-content keys that reach an event anyway.

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

_LONG_STRING = 200
_MAX_DEPTH = 8
_SHA_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$", re.IGNORECASE)


def resolve_environment(sentry_environment: Optional[str], runtime_environment: str) -> str:
    """SENTRY_ENVIRONMENT wins; blank counts as unset; else the runtime label."""
    explicit = (sentry_environment or "").strip()
    return explicit or runtime_environment


def resolve_release(git_commit_sha: str) -> Optional[str]:
    """The full build SHA, or None when the build identity is unknown."""
    return git_commit_sha if _SHA_RE.match(git_commit_sha or "") else None


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
                if str(name).lower() in SENSITIVE_HEADERS:
                    del headers[name]

    _strip_frame_vars(event.get("exception"))
    _strip_frame_vars(event.get("threads"))

    if isinstance(event.get("extra"), Mapping):
        event["extra"] = _redact(event["extra"])

    contexts = event.get("contexts")
    if isinstance(contexts, MutableMapping):
        for key in list(contexts.keys()):
            if _is_sensitive_key(key):
                del contexts[key]
            elif isinstance(contexts[key], Mapping):
                contexts[key] = _redact(contexts[key])

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
        trace["data"] = _redact(trace["data"])
    for span in event.get("spans") or []:
        if isinstance(span, MutableMapping) and isinstance(span.get("data"), Mapping):
            span["data"] = _redact(span["data"])

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
