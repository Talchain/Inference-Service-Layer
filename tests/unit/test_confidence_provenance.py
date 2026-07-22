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

from src.config.stability_thresholds import (
    CONFIDENCE_CATEGORY_WEIGHT,
    CONFIDENCE_CV_ELASTICITY_FLOOR,
    CONFIDENCE_CV_WEIGHT,
    CONFIDENCE_METHOD_VERSION,
    STABILITY_CONFIDENCE_DEFAULT,
    STABILITY_CONFIDENCE_MAP,
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
# PINNED PAIR (update BOTH together, and only as a deliberate, disclosed change):
#   fingerprint  <->  method_version
# When you intentionally change a mapping constant:
#   1. bump CONFIDENCE_METHOD_VERSION in src/config/stability_thresholds.py, and
#   2. update PINNED_CONSTANTS_FINGERPRINT + PINNED_METHOD_VERSION below to the
#      new values (the failure message prints the fresh fingerprint to paste).
# Do NOT regenerate the pin blindly to make this test pass — that would defeat
# the disclosure contract.
PINNED_CONSTANTS_FINGERPRINT = "89749af1536d3b9ff02175cefd315b3d7ac3f1dd9ad70724257b50e3e87a9f69"
PINNED_METHOD_VERSION = "stability-cv-blend-v1"


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
            "CONFIDENCE_CV_ELASTICITY_FLOOR) changed but CONFIDENCE_METHOD_VERSION "
            f"is still '{CONFIDENCE_METHOD_VERSION}'. This is a recalibration and "
            "MUST be disclosed (Neil gate 1, no silent reweighting):\n"
            "  1. bump CONFIDENCE_METHOD_VERSION in src/config/stability_thresholds.py, and\n"
            "  2. update PINNED_CONSTANTS_FINGERPRINT + PINNED_METHOD_VERSION in this test.\n"
            f"Current fingerprint to pin: {current}\n"
            f"Old pinned fingerprint:     {PINNED_CONSTANTS_FINGERPRINT}"
        )
