"""C3 (altitude Q3 + hunter R1, 2026-07-23): every response model that aliases a
field to `_metadata` MUST set `populate_by_name: True`.

Why this is a class fix, not a one-off: the repo's worker-offload convention is
`response.model_dump_json()` on the worker -> `Model.model_validate_json(...)` on
the caller (analysis_pool.py). `model_dump_json()` defaults to `by_alias=False`,
so it emits the field by its PYTHON NAME (`"metadata"`), NOT the alias
(`"_metadata"`). A model WITHOUT `populate_by_name` refuses field-name input, so
`model_validate_json` silently drops the field to its `Optional` default `None`
(demonstrated for V1 RobustnessResponse in HUNT-SERIAL P6a). RobustnessResponseV2
survives production offload precisely because it already carries
`populate_by_name: True`; V1 did not.

This test is DERIVE-DON'T-MIRROR: it WALKS the `src.models` package and discovers
every `BaseModel` with an `_metadata`-aliased field at runtime — there is no
hand-maintained allowlist to drift. Adding a new aliased response model without
`populate_by_name` makes this test RED at collection of the model set, not in
production.
"""

import importlib
import inspect
import pkgutil

from pydantic import BaseModel

import src.models as models_pkg
from src.models.metadata import ResponseMetadata

# The model modules that MUST import cleanly for the walk to be meaningful. If any
# of these fails to import, the walk would under-count and the invariant would pass
# vacuously — so we assert they are all reachable (positive control on the walk).
_REQUIRED_MODEL_MODULES = [
    "src.models.responses",
    "src.models.robustness",
    "src.models.robustness_v2",
    "src.models.phase1_models",
    "src.models.decision_robustness",
    "src.models.deliberation",
]

# A floor on how many aliased models the walk must find. Not an allowlist (the walk
# is authoritative); a guard against a silently-empty discovery (trap #13 — the
# test must prove it can SEE the models it asserts over).
_MIN_ALIASED_MODELS = 20


def _has_populate_by_name(cls: type) -> bool:
    """True iff the model's EFFECTIVE Pydantic config enables populate_by_name.

    Pydantic v2 merges both the `model_config = {...}` dict idiom and the legacy
    `class Config` idiom into `cls.model_config`, so one lookup covers both.
    """
    mc = getattr(cls, "model_config", None)
    return isinstance(mc, dict) and mc.get("populate_by_name") is True


def _discover_metadata_aliased_models() -> list[tuple[type, str]]:
    """Walk src.models; return (class, field_name) for every BaseModel subclass
    DEFINED in src.models that aliases a field to `_metadata`."""
    # Assert the load-bearing modules import (guards against a vacuous walk).
    for name in _REQUIRED_MODEL_MODULES:
        importlib.import_module(name)

    found: list[tuple[type, str]] = []
    seen: set[type] = set()
    for mod_info in pkgutil.walk_packages(models_pkg.__path__, models_pkg.__name__ + "."):
        module = importlib.import_module(mod_info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, BaseModel) or cls is BaseModel:
                continue
            if not getattr(cls, "__module__", "").startswith("src.models"):
                continue  # skip re-exports; attribute to the defining module only
            if cls in seen:
                continue
            seen.add(cls)
            for field_name, field_info in cls.model_fields.items():
                if getattr(field_info, "alias", None) == "_metadata":
                    found.append((cls, field_name))
                    break
    return found


def test_all_metadata_aliased_models_declare_populate_by_name():
    """The invariant: every `_metadata`-aliased response model sets
    populate_by_name=True, so field-name construction (the offload convention and
    the routes' attribute-assign idiom) round-trips instead of silently dropping."""
    aliased = _discover_metadata_aliased_models()

    # Positive control: the walk actually found the model set (not vacuous).
    assert len(aliased) >= _MIN_ALIASED_MODELS, (
        f"only {len(aliased)} aliased models discovered (expected >= "
        f"{_MIN_ALIASED_MODELS}); the walk under-counted — a models module likely "
        f"failed to import."
    )

    offenders = [
        f"{cls.__module__}.{cls.__name__}.{field}"
        for cls, field in aliased
        if not _has_populate_by_name(cls)
    ]
    assert not offenders, (
        "response models alias a field to `_metadata` but do NOT set "
        "populate_by_name=True — field-name input (worker offload / route "
        "attribute-assign) will be silently dropped to None:\n  "
        + "\n  ".join(offenders)
    )


def test_v1_robustness_metadata_survives_offload_convention():
    """Functional round-trip on the DEMONSTRATED model (V1 RobustnessResponse,
    HUNT-SERIAL P6a): construct by FIELD NAME, then the exact worker convention
    (`model_dump_json()` -> `model_validate_json()`); the metadata survives."""
    from src.models.robustness import FACETRobustnessAnalysis, RobustnessResponse

    analysis = FACETRobustnessAnalysis(
        status="robust",
        robustness_score=0.5,
        region_count=1,
        total_volume=1.0,
        is_fragile=False,
        samples_tested=200,
        samples_successful=50,
        interpretation="X",
        recommendation="Y",
    )
    resp = RobustnessResponse(analysis=analysis)
    # Attribute-assign by field name — the live route's idiom.
    resp.metadata = ResponseMetadata(
        isl_version="v", config_fingerprint="fp", config_details={}, request_id="r"
    )

    # The worker offload convention: field-name dump, then reconstruct.
    dumped = resp.model_dump_json()
    restored = RobustnessResponse.model_validate_json(dumped)

    assert restored.metadata is not None, "V1 _metadata dropped under offload round-trip"
    assert restored.metadata.request_id == "r"


def test_offload_roundtrip_discriminates_on_populate_by_name():
    """Positive+negative control (trap #13): the offload round-trip can SEE the
    difference populate_by_name makes — WITHOUT it the field-name-assigned aliased
    field is DROPPED; WITH it the field SURVIVES. Proves the survival test above is
    not vacuous."""
    from typing import Optional

    from pydantic import Field

    class _WithoutPBN(BaseModel):
        metadata: Optional[ResponseMetadata] = Field(default=None, alias="_metadata")

    class _WithPBN(BaseModel):
        model_config = {"populate_by_name": True}
        metadata: Optional[ResponseMetadata] = Field(default=None, alias="_metadata")

    meta = ResponseMetadata(
        isl_version="v", config_fingerprint="fp", config_details={}, request_id="r"
    )

    without = _WithoutPBN()
    without.metadata = meta
    dropped = _WithoutPBN.model_validate_json(without.model_dump_json())
    assert dropped.metadata is None, "control failed: expected the DROP without populate_by_name"

    with_pbn = _WithPBN()
    with_pbn.metadata = meta
    kept = _WithPBN.model_validate_json(with_pbn.model_dump_json())
    assert kept.metadata is not None, "control failed: expected SURVIVAL with populate_by_name"
