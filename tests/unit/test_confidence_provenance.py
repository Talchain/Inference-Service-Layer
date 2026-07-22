"""
S2 — confidence-provenance disclosure marker: model serialisation + the
derive-don't-mirror fingerprint guard.

Two concerns:

1. ABSENT-NOT-NULL serialisation contract on FactorSensitivityV2:
   `confidence_provenance` is present EXACTLY when `confidence` is present, and
   OMITTED (never a JSON null) when `confidence` is absent — matching how ISL
   serialises every optional V2 field (by_alias=True, exclude_none=True).

2. THE GUARD (the point of the slice): a stable fingerprint over ALL the
   constants that define the provisional confidence mapping, pinned to
   CONFIDENCE_METHOD_VERSION. If any constant changes while method_version stays
   "stability-cv-blend-v1", this test fails loud and tells the editor to bump the
   version and update the pinned pair — mechanising the Neil-gate-1 rule that any
   recalibration is a DISCLOSED, versioned change (no silent reweighting).

   The pinned fingerprint below is a LITERAL. The test computes the CURRENT
   fingerprint from the live constants and compares it to that literal — it does
   NOT regenerate the pin from the same source it is checking, so it cannot
   self-heal.
"""

import hashlib
import json

import pytest

from src.config.stability_thresholds import (
    CONFIDENCE_CATEGORY_WEIGHT,
    CONFIDENCE_CV_ELASTICITY_FLOOR,
    CONFIDENCE_CV_WEIGHT,
    CONFIDENCE_METHOD_VERSION,
    DEFAULT_CV_HIGH_MODERATE_BOUNDARY,
    DEFAULT_CV_MODERATE_LOW_BOUNDARY,
    GRAPH_STRUCTURAL_CONFIDENCE_FLOOR,
    GRAPH_STRUCTURAL_CONFIDENCE_SCALE,
    GRAPH_STRUCTURAL_METHOD_VERSION,
    STABILITY_CONFIDENCE_DEFAULT,
    STABILITY_CONFIDENCE_MAP,
    get_confidence_method_version,
)
from src.models.response_v2 import ConfidenceProvenance, FactorSensitivityV2

# ---------------------------------------------------------------------------
# 1. ABSENT-NOT-NULL serialisation contract
# ---------------------------------------------------------------------------


class TestConfidenceProvenanceSerialisation:
    def test_absent_not_null_when_confidence_absent(self):
        """Factor WITHOUT confidence → no confidence_provenance key at all in the
        serialised wire response (absent, NOT null)."""
        fs = FactorSensitivityV2(
            node_id="x",
            sensitivity_score=0.5,
            direction="positive",
        )
        # Precondition: this factor carries no confidence.
        assert fs.confidence is None
        assert fs.confidence_provenance is None

        d = fs.model_dump(by_alias=True, exclude_none=True)  # exact wire semantics
        assert "confidence" not in d
        # The key must be ABSENT — not present as a JSON null.
        assert "confidence_provenance" not in d

        # Positive control: prove the assertion above can SEE a value if one is
        # emitted. Without exclude_none the field surfaces as an explicit null,
        # so a regression that stopped omitting it would be caught by the
        # exclude_none dump above.
        d_all = fs.model_dump(by_alias=True)
        assert "confidence_provenance" in d_all
        assert d_all["confidence_provenance"] is None

    def test_present_when_confidence_present(self):
        """Factor WITH confidence + a populated marker → both ride the wire, the
        marker carrying the version and calibrated=False (not null-injected)."""
        fs = FactorSensitivityV2(
            node_id="x",
            sensitivity_score=0.5,
            direction="positive",
            confidence=0.7,
            confidence_source="bootstrap_sampling",
            confidence_provenance=ConfidenceProvenance(
                method_version=CONFIDENCE_METHOD_VERSION,
                calibrated=False,
            ),
        )
        d = fs.model_dump(by_alias=True, exclude_none=True)
        assert d["confidence"] == 0.7
        assert "confidence_provenance" in d
        assert d["confidence_provenance"] == {
            "method_version": CONFIDENCE_METHOD_VERSION,
            "calibrated": False,
        }

    def test_marker_exactly_two_fields(self):
        """The disclosure marker has EXACTLY two fields (method_version, calibrated)."""
        prov = ConfidenceProvenance(method_version="whatever", calibrated=True)
        assert set(prov.model_dump().keys()) == {"method_version", "calibrated"}


# ---------------------------------------------------------------------------
# 2. THE GUARD — constants fingerprint <-> method_version
# ---------------------------------------------------------------------------

# Constants that DEFINE the provisional stability->confidence mapping. Any change
# to these is a recalibration and MUST bump CONFIDENCE_METHOD_VERSION.
#
# F-1: the DEFAULT CV boundaries (DEFAULT_CV_HIGH_MODERATE_BOUNDARY /
# DEFAULT_CV_MODERATE_LOW_BOUNDARY) are part of the STANDARD method — they decide
# `attribution_stability`, the INPUT to the mapping — so they are fingerprinted
# here too. (The STABILITY_CV_* env overrides that MOVE them at runtime are
# disclosed separately via get_confidence_method_version()'s "+env-override"
# suffix; this fingerprint guards the compile-time DEFAULTS.)
#
# PINNED PAIR (update BOTH together, and only as a deliberate, disclosed change):
#   fingerprint  <->  method_version
# When you intentionally change a mapping constant:
#   1. bump CONFIDENCE_METHOD_VERSION in src/config/stability_thresholds.py, and
#   2. update PINNED_CONSTANTS_FINGERPRINT + PINNED_METHOD_VERSION below to the
#      new values (the failure message prints the fresh fingerprint to paste).
# Do NOT regenerate the pin blindly to make this test pass — that would defeat
# the disclosure contract.
PINNED_CONSTANTS_FINGERPRINT = "9ed98e30b7208547ab0babd202442dac41d89f042fa5b8ad7b7a1e2ddb6b1219"
PINNED_METHOD_VERSION = "stability-cv-blend-v1"

# Graph-structural FALLBACK method (F-2) — a DIFFERENT confidence method with its
# OWN constants and OWN version string. Same pinned-pair pattern, separate literal.
PINNED_GRAPH_STRUCTURAL_FINGERPRINT = (
    "d1a6932a3e60c83ca0683d5a492603c5f799c2dd4b6d185e8a661564bb5b9a25"
)
PINNED_GRAPH_STRUCTURAL_METHOD_VERSION = "graph-structural-v1"


def _current_constants_fingerprint() -> str:
    """sha256 over a canonical repr of the mapping constants (computed LIVE from
    source — the pin is a hardcoded literal, so this never self-heals)."""
    canonical = json.dumps(
        {
            "STABILITY_CONFIDENCE_MAP": {
                k: STABILITY_CONFIDENCE_MAP[k] for k in sorted(STABILITY_CONFIDENCE_MAP)
            },
            "STABILITY_CONFIDENCE_DEFAULT": STABILITY_CONFIDENCE_DEFAULT,
            "CONFIDENCE_CATEGORY_WEIGHT": CONFIDENCE_CATEGORY_WEIGHT,
            "CONFIDENCE_CV_WEIGHT": CONFIDENCE_CV_WEIGHT,
            "CONFIDENCE_CV_ELASTICITY_FLOOR": CONFIDENCE_CV_ELASTICITY_FLOOR,
            "DEFAULT_CV_HIGH_MODERATE_BOUNDARY": DEFAULT_CV_HIGH_MODERATE_BOUNDARY,
            "DEFAULT_CV_MODERATE_LOW_BOUNDARY": DEFAULT_CV_MODERATE_LOW_BOUNDARY,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _current_graph_structural_fingerprint() -> str:
    """sha256 over the graph-structural fallback constants (F-2). Pinned to
    GRAPH_STRUCTURAL_METHOD_VERSION; computed LIVE, pin is a literal (no self-heal)."""
    canonical = json.dumps(
        {
            "GRAPH_STRUCTURAL_CONFIDENCE_FLOOR": GRAPH_STRUCTURAL_CONFIDENCE_FLOOR,
            "GRAPH_STRUCTURAL_CONFIDENCE_SCALE": GRAPH_STRUCTURAL_CONFIDENCE_SCALE,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TestConfidenceMethodVersionGuard:
    def test_method_version_matches_pin(self):
        """The served method_version must equal the pinned version (they move
        together)."""
        assert CONFIDENCE_METHOD_VERSION == PINNED_METHOD_VERSION

    def test_constants_fingerprint_pinned_to_version(self):
        """If any mapping constant changed while method_version stayed put, this
        fails loud and tells the editor exactly what to do."""
        current = _current_constants_fingerprint()
        assert current == PINNED_CONSTANTS_FINGERPRINT, (
            "The provisional confidence-mapping constants "
            "(STABILITY_CONFIDENCE_MAP / STABILITY_CONFIDENCE_DEFAULT / "
            "CONFIDENCE_CATEGORY_WEIGHT / CONFIDENCE_CV_WEIGHT / "
            "CONFIDENCE_CV_ELASTICITY_FLOOR / DEFAULT_CV_HIGH_MODERATE_BOUNDARY / "
            "DEFAULT_CV_MODERATE_LOW_BOUNDARY) changed but CONFIDENCE_METHOD_VERSION "
            f"is still '{CONFIDENCE_METHOD_VERSION}'. This is a recalibration and "
            "MUST be disclosed (Neil gate 1, no silent reweighting):\n"
            "  1. bump CONFIDENCE_METHOD_VERSION in src/config/stability_thresholds.py, and\n"
            "  2. update PINNED_CONSTANTS_FINGERPRINT + PINNED_METHOD_VERSION in this test.\n"
            f"Current fingerprint to pin: {current}\n"
            f"Old pinned fingerprint:     {PINNED_CONSTANTS_FINGERPRINT}"
        )


class TestGraphStructuralMethodVersionGuard:
    """F-2 — the fallback method's own constants pinned to its own version."""

    def test_graph_structural_method_version_matches_pin(self):
        """The fallback method_version must equal its pinned version and be
        DISTINCT from the bootstrap-blend version."""
        assert GRAPH_STRUCTURAL_METHOD_VERSION == PINNED_GRAPH_STRUCTURAL_METHOD_VERSION
        assert GRAPH_STRUCTURAL_METHOD_VERSION != CONFIDENCE_METHOD_VERSION

    def test_graph_structural_constants_fingerprint_pinned_to_version(self):
        """If a fallback constant changed while its version stayed put, fail loud."""
        current = _current_graph_structural_fingerprint()
        assert current == PINNED_GRAPH_STRUCTURAL_FINGERPRINT, (
            "The graph-structural fallback constants (GRAPH_STRUCTURAL_CONFIDENCE_FLOOR "
            "/ GRAPH_STRUCTURAL_CONFIDENCE_SCALE) changed but "
            f"GRAPH_STRUCTURAL_METHOD_VERSION is still '{GRAPH_STRUCTURAL_METHOD_VERSION}'. "
            "This is a recalibration of a DIFFERENT method and MUST be disclosed:\n"
            "  1. bump GRAPH_STRUCTURAL_METHOD_VERSION in src/config/stability_thresholds.py, and\n"
            "  2. update PINNED_GRAPH_STRUCTURAL_FINGERPRINT + "
            "PINNED_GRAPH_STRUCTURAL_METHOD_VERSION in this test.\n"
            f"Current fingerprint to pin: {current}\n"
            f"Old pinned fingerprint:     {PINNED_GRAPH_STRUCTURAL_FINGERPRINT}"
        )


# ---------------------------------------------------------------------------
# 3. F-1 — env-override recalibration is DISCLOSED in method_version
# ---------------------------------------------------------------------------

# The STABILITY_CV_* env overrides move the CV boundaries that decide
# attribution_stability (the INPUT to the confidence mapping), so an active
# override yields a DIFFERENT emitted confidence computed by a NON-standard
# method. get_confidence_method_version() must disclose this by suffixing
# "+env-override" — closing the back door where the number changes but the
# version does not. Detection is a SINGLE derivation (live boundaries vs the
# default literals), not a hand-maintained mirror.


class TestEnvOverrideMethodVersion:
    def test_no_suffix_when_no_override(self, monkeypatch):
        """No env vars set → the standard version, no suffix."""
        monkeypatch.delenv("STABILITY_CV_HIGH_MODERATE", raising=False)
        monkeypatch.delenv("STABILITY_CV_MODERATE_LOW", raising=False)
        assert get_confidence_method_version() == CONFIDENCE_METHOD_VERSION
        assert "+env-override" not in get_confidence_method_version()

    def test_no_suffix_when_override_equals_default(self, monkeypatch):
        """Setting the vars to EXACTLY the default boundary is not a
        recalibration — the method is unchanged, so no suffix. (This is why the
        detection compares VALUES, not mere env presence.)"""
        monkeypatch.setenv(
            "STABILITY_CV_HIGH_MODERATE", str(DEFAULT_CV_HIGH_MODERATE_BOUNDARY)
        )
        monkeypatch.setenv(
            "STABILITY_CV_MODERATE_LOW", str(DEFAULT_CV_MODERATE_LOW_BOUNDARY)
        )
        assert get_confidence_method_version() == CONFIDENCE_METHOD_VERSION

    def test_suffix_when_high_moderate_overridden(self, monkeypatch):
        """Only the high/moderate boundary moved → suffix present."""
        monkeypatch.setenv("STABILITY_CV_HIGH_MODERATE", "0.2")
        monkeypatch.delenv("STABILITY_CV_MODERATE_LOW", raising=False)
        assert (
            get_confidence_method_version()
            == CONFIDENCE_METHOD_VERSION + "+env-override"
        )

    def test_suffix_when_moderate_low_overridden(self, monkeypatch):
        """Only the moderate/low boundary moved → suffix present."""
        monkeypatch.delenv("STABILITY_CV_HIGH_MODERATE", raising=False)
        monkeypatch.setenv("STABILITY_CV_MODERATE_LOW", "0.4")
        assert (
            get_confidence_method_version()
            == CONFIDENCE_METHOD_VERSION + "+env-override"
        )

    def test_suffix_when_both_overridden(self, monkeypatch):
        """Both boundaries moved → suffix present."""
        monkeypatch.setenv("STABILITY_CV_HIGH_MODERATE", "0.15")
        monkeypatch.setenv("STABILITY_CV_MODERATE_LOW", "0.35")
        assert (
            get_confidence_method_version()
            == CONFIDENCE_METHOD_VERSION + "+env-override"
        )


# ---------------------------------------------------------------------------
# 4. F-3 — the iff-invariant is ENFORCED at the model level, both directions
# ---------------------------------------------------------------------------

# Without a model validator, a mutation that ALWAYS emits the marker (even when
# confidence is None) — or never emits it — passes the suite silently (trap-#13:
# an absence assertion with no discriminating enforcement). The validator makes
# BOTH violation directions fail loud at construction.


class TestConfidenceProvenanceIffInvariant:
    def test_both_none_is_valid(self):
        """confidence None + provenance None → valid (the absent case)."""
        fs = FactorSensitivityV2(
            node_id="x", sensitivity_score=0.5, direction="positive"
        )
        assert fs.confidence is None
        assert fs.confidence_provenance is None

    def test_both_present_is_valid(self):
        """confidence set + provenance set → valid (the present case)."""
        fs = FactorSensitivityV2(
            node_id="x",
            sensitivity_score=0.5,
            direction="positive",
            confidence=0.7,
            confidence_provenance=ConfidenceProvenance(
                method_version=CONFIDENCE_METHOD_VERSION, calibrated=False
            ),
        )
        assert fs.confidence == 0.7
        assert fs.confidence_provenance is not None

    def test_marker_without_confidence_raises(self):
        """provenance present while confidence is None → ValueError (the direction
        the marker-always mutation would exploit)."""
        with pytest.raises(ValueError, match="iff-invariant"):
            FactorSensitivityV2(
                node_id="x",
                sensitivity_score=0.5,
                direction="positive",
                confidence=None,
                confidence_provenance=ConfidenceProvenance(
                    method_version=CONFIDENCE_METHOD_VERSION, calibrated=False
                ),
            )

    def test_confidence_without_marker_raises(self):
        """confidence present while provenance is None → ValueError (the other
        direction — a dropped marker must not pass silently)."""
        with pytest.raises(ValueError, match="iff-invariant"):
            FactorSensitivityV2(
                node_id="x",
                sensitivity_score=0.5,
                direction="positive",
                confidence=0.7,
                confidence_provenance=None,
            )
