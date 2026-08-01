"""
Track S Phase 1 — seed-sweep flip-threshold stability bands (DEFAULT-ON).

Why: the 2026-06-10 PLoT/ISL science-performance report found that flip
thresholds computed from a single seed are presented with false stability,
and recommended reporting flip thresholds with "a stability band from a
small seed sweep (e.g. 5 seeds)" with flip confidence based on band width
(report §5 "Flip-threshold drift", §7 recommendation, next-step #6).
The 2026-07-07 science-validation REPORT.md (§1, §5.1) reinforced that
single-seed flip evidence is weak.

History: bands shipped flag-gated behind ``ISL_FLIP_STABILITY_BANDS`` (+
``ISL_FLIP_STABILITY_SEEDS``) on 2026-07-11. Per Paul's ruling (2026-07-17:
core functionality, no flag unless genuinely needed) both env vars are
REMOVED and bands are computed whenever ``edge_e_values`` are.

Contract under test (default-on, ADDITIVE only):

- with NO environment configuration, each ``edge_e_values[]`` entry gains a
  ``stability`` object::

      {
        "n_seeds": int,             # seeds swept (constant 10)
        "n_seeds_flipped": int,     # seeds whose background admits a flip
        "band_min": float,          # min flip_mean across flipped seeds
        "band_median": float,       # median flip_mean across flipped seeds
        "band_max": float,          # max flip_mean across flipped seeds
        "band_width": float,        # band_max - band_min
        "seed_flip_means": [float|None, ...]  # per-child-seed flip means
      }

  The four band_* keys are OMITTED (not null) when n_seeds_flipped == 0,
  matching the v2 wire's exclude_none serialisation so the v1 (dict
  passthrough) and v2 (model) wires carry the same shape. None elements
  inside seed_flip_means are preserved on both wires (verified: pydantic
  exclude_none does not drop None list elements).

- the legacy env vars are DEAD: setting either has no effect.
- N is the code constant ``FLIP_STABILITY_N_SEEDS = 10`` (Paul ruling
  17 Jul raised the 06-10 report's 5 for a better stability basis); there
  is no runtime override.
- additivity vs base: stripping every ``stability`` key from the wire must
  recover the pre-bands base wire exactly, pinned by the golden fixture
  captured at origin/staging e029cae2d (comparison is modulo the four
  pre-existing volatile fields and the pre-existing set-ordering leak
  catalogued in docs/science-validation/REPORT.md §3 — those vary
  run-to-run/process-to-process on the BASE code already).
- budget degradation (mechanics unchanged from the flag era; value raised
  2000 -> 30000 ms by the same 17 Jul ruling): the sweep is capped by
  ``FLIP_STABILITY_BUDGET_MS``, ALL-OR-NOTHING on exceed — no entry carries
  a band (partial bands would bias readers toward whichever edges computed
  first), the base edge_e_values are untouched, and the degradation is
  disclosed via the ``flip_stability_budget_exceeded`` structured log
  event carrying ``elapsed_ms``.
- determinism: the sweep is seeded — child seeds are SHA-256-derived from
  the master (request) seed, so same request+seed -> byte-identical bands.

Regenerate the golden fixture (base wire modulo the additive stability key)
with:

    poetry run python -m tests.unit.test_flip_stability_bands
"""

import copy
import json
import logging
import statistics
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.response_v2 import FlipStabilityBandV2
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    FLIP_STABILITY_N_SEEDS,
    RobustnessAnalyzerV2,
)

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

# Removed env vars — referenced only to prove they are DEAD.
LEGACY_FLAG_ENV = "ISL_FLIP_STABILITY_BANDS"
LEGACY_N_SEEDS_ENV = "ISL_FLIP_STABILITY_SEEDS"
# Paul ruling 17 Jul (lenient-latency amendment): N raised from the 06-10
# report's 5 to 10 for a better stability basis, and the sweep budget raised
# 2000 -> 30000 ms. Both VALUE-pinned here so a silent revert goes RED.
N_SEEDS = 10
BUDGET_MS = 30000

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS_PATH = REPO_ROOT / "tests" / "benchmarks" / "sample_variants.json"
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "flip_stability" / "golden_base_v2.json"

STABILITY_REQUIRED_KEYS = {"n_seeds", "n_seeds_flipped", "seed_flip_means"}
STABILITY_BAND_KEYS = {"band_min", "band_median", "band_max", "band_width"}


def _assert_block_shape(block: dict, expected_n_seeds: int) -> None:
    """Shared shape assertion: exact key set depends on whether any seed flipped."""
    assert STABILITY_REQUIRED_KEYS <= set(block.keys())
    assert block["n_seeds"] == expected_n_seeds
    assert len(block["seed_flip_means"]) == expected_n_seeds
    assert 0 <= block["n_seeds_flipped"] <= expected_n_seeds
    if block["n_seeds_flipped"] > 0:
        assert set(block.keys()) == STABILITY_REQUIRED_KEYS | STABILITY_BAND_KEYS
    else:
        assert set(block.keys()) == STABILITY_REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _variant_request(idx: int, seed: int = 42) -> dict:
    """Build a wire-shaped request from the pinned sample_variants fixture."""
    variants = json.loads(VARIANTS_PATH.read_text())
    graph = variants["graphs"][idx]
    return {
        "request_id": "flip-stability-golden-001",
        "graph": graph,
        "options": variants["options"],
        "goal_node_id": variants["goal_node_id"],
        "seed": seed,
        "n_samples": variants["n_samples"],
        "include_e_values": True,
    }


def _analyzer_request(idx: int, seed: int = 42) -> RobustnessRequestV2:
    return RobustnessRequestV2(**_variant_request(idx, seed=seed))


# ---------------------------------------------------------------------------
# Wire normalisation (REPORT.md §3 volatile-field catalogue)
# ---------------------------------------------------------------------------


def normalize_v2_payload(payload: dict) -> dict:
    """Strip the pre-existing volatile fields and order leaks from a v2 wire payload.

    Everything removed/sorted here varies on the BASE code already
    (docs/science-validation/REPORT.md §3): wall clocks, uuid4 critique ids,
    and the ``list(set(...))`` ordering of the two v1-compat edge lists (plus
    the interpretation string derived from that ordering). All numeric
    content is preserved untouched.
    """
    data = copy.deepcopy(payload)
    # Envelope wall clocks
    data.pop("timestamp", None)
    data.pop("processing_time_ms", None)
    # Deploy-environment echo (RENDER_GIT_COMMIT or "dev") — not code behaviour
    data.pop("build", None)
    for meta_key in ("metadata", "_metadata"):
        meta = data.get(meta_key)
        if isinstance(meta, dict):
            meta.pop("execution_time_ms", None)
    # uuid4-per-run critique ids (critique content is deterministic)
    for critique in data.get("critiques") or []:
        if isinstance(critique, dict):
            critique.pop("id", None)
    # Per-process hash-salt ordering of the two v1-compat string lists,
    # and any interpretation text derived from that ordering.
    robustness = data.get("robustness")
    if isinstance(robustness, dict):
        for key in ("fragile_edges_v1", "robust_edges"):
            if isinstance(robustness.get(key), list):
                robustness[key] = sorted(robustness[key])
        robustness.pop("interpretation", None)
    return data


# Additive-only wire surfaces layered on top of the pre-bands base wire. The
# base golden (golden_base_v2.json) predates ALL of them, so each must be
# stripped before the modulo comparison. 'stability' = flip-stability bands;
# 'downside' = B2 tail-risk view (cvar_10/p05/expected_regret);
# 'decision_evpi' = S1 decision-level EVPI (A3 VOI, D-23.8; top-level scalar).
# Arch step 1 (2026-07-26) additions: 'confidence_basis' = machine-readable
# semantics marker beside robustness.confidence; 'sample_population_provenance' =
# per-metric noise provenance on the envelope. Both are purely ADDITIVE
# disclosure surfaces, which is exactly what this set is for.
_ADDITIVE_WIRE_SURFACES = frozenset(
    {
        "stability",
        "downside",
        "decision_evpi",
        "confidence_basis",
        "sample_population_provenance",
    }
)

# ROADMAP 2.228-F3 (2026-08-01): 'alternative_winner_id' and 'baseline_winner_id'
# on each robustness.edge_e_values entry. Purely additive — the winner on the
# flipped side of the bracket was already computed by the existing bisection and
# discarded, and the baseline winner is the one the search already ran against,
# so no existing value moves.
#
# ⚠ DELIBERATELY *NOT* ADDED TO _ADDITIVE_WIRE_SURFACES, which strips by key name
# RECURSIVELY: `alternative_winner_id` ALREADY EXISTS on FragileEdgeV2 and is part
# of the frozen golden, so a name-based strip would silently delete a pre-existing
# field from `current` and hide any regression in it. (Not hypothetical — the
# first attempt did exactly that and this test caught it.) Stripped by PATH
# instead, so the fragile-edge field stays compared.
_EDGE_E_VALUE_ADDITIVE_KEYS = ("alternative_winner_id", "baseline_winner_id")


def _strip_edge_e_value_additions(payload: dict) -> dict:
    """Remove the 2.228-F3 additive keys from robustness.edge_e_values entries only."""
    data = copy.deepcopy(payload)
    robustness = data.get("robustness")
    if isinstance(robustness, dict):
        for entry in robustness.get("edge_e_values") or []:
            if isinstance(entry, dict):
                for key in _EDGE_E_VALUE_ADDITIVE_KEYS:
                    entry.pop(key, None)
    return data

# Arch step 1 (2026-07-26): robustness.confidence is NOT additive — its VALUE
# changed, so it cannot be silently stripped. The golden holds the base value
# min(0.99, stability*(1 - 1/sqrt(n_samples))); the wire now carries the
# stability fraction itself, because the shrinkage term was a function of sample
# COUNT alone and moved the published number with how long the run took rather
# than with anything about the recommendation. The change is asserted explicitly
# in test_confidence_is_now_the_bare_stability_fraction below, so it is pinned
# rather than waved through, and only then excluded from the modulo comparison.
_CHANGED_WIRE_VALUES = ("robustness", "confidence")


def _strip_additive_surfaces(payload, keys=_ADDITIVE_WIRE_SURFACES):
    """Recursively remove additive-only surface keys so what remains is the
    pre-feature base wire (compared byte-for-byte against the frozen golden).

    ``keys`` is narrowable on purpose. The golden comparison strips EVERY
    additive surface, but a caller asking a narrower question ("did the band
    budget disturb the base e-values?") must strip only the surface it is
    controlling for — stripping more would also hide a real regression in the
    others.
    """
    if isinstance(payload, dict):
        return {
            k: _strip_additive_surfaces(v, keys) for k, v in payload.items() if k not in keys
        }
    if isinstance(payload, list):
        return [_strip_additive_surfaces(item, keys) for item in payload]
    return payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def no_env(monkeypatch):
    """Guarantee NEITHER legacy env var is set — the default-on contract is
    exactly 'bands present with no environment configuration at all'."""
    monkeypatch.delenv(LEGACY_FLAG_ENV, raising=False)
    monkeypatch.delenv(LEGACY_N_SEEDS_ENV, raising=False)


def _stability_blocks(response) -> list:
    assert response.edge_e_values, "expected edge_e_values to be computed"
    return [entry.get("stability") for entry in response.edge_e_values]


# ---------------------------------------------------------------------------
# DEFAULT-ON — bands present with no env var set (the gate-removal pin)
# ---------------------------------------------------------------------------


class TestDefaultOn:
    def test_bands_present_with_no_env_vars(self, no_env):
        """THE gate-removal pin: no environment configuration, bands attached."""
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        blocks = _stability_blocks(response)
        assert all(isinstance(b, dict) for b in blocks), (
            "every edge_e_values entry must carry a stability band by default " "(no env var set)"
        )
        for block in blocks:
            _assert_block_shape(block, N_SEEDS)

    def test_bands_on_v2_wire_with_no_env_vars(self, no_env, client):
        """Same pin at the analyze/v2 HTTP surface."""
        resp = client.post(ENDPOINT, json=_variant_request(2), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        edge_e_values = (resp.json().get("robustness") or {}).get("edge_e_values")
        assert edge_e_values, "expected edge_e_values on the v2 wire"
        for entry in edge_e_values:
            block = entry.get("stability")
            assert isinstance(block, dict), "default v2 wire must carry stability bands"
            _assert_block_shape(block, N_SEEDS)

    def test_legacy_env_vars_are_dead(self, monkeypatch):
        """The removed env vars must have NO effect: an old 'off' value cannot
        suppress bands, and an old seeds override cannot change N."""
        monkeypatch.setenv(LEGACY_FLAG_ENV, "0")
        monkeypatch.setenv(LEGACY_N_SEEDS_ENV, "3")
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        for block in _stability_blocks(response):
            _assert_block_shape(block, N_SEEDS)

    def test_n_seeds_constant_value_pinned_at_ten(self):
        """Paul ruling 17 Jul: n_seeds = 10. Value-pinned — a silent revert
        to the report-era 5 (or any other value) goes RED here."""
        assert FLIP_STABILITY_N_SEEDS == 10

    def test_budget_constant_value_pinned_at_30000(self):
        """Paul ruling 17 Jul: lenient sweep budget = 30000 ms (prioritise
        analysis quality; disclose slowness rather than silently cut).
        Value-pinned — a silent revert to the original 2000 goes RED here."""
        assert RobustnessAnalyzerV2.FLIP_STABILITY_BUDGET_MS == BUDGET_MS


# ---------------------------------------------------------------------------
# Additivity vs BASE — stripping stability recovers the pre-bands wire
# ---------------------------------------------------------------------------


class TestAdditiveVsBase:
    def test_v2_wire_modulo_stability_matches_base_golden(
        self, no_env, client, auto_noise_enabled
    ):
        """Pin: the ONLY wire deltas vs origin/staging base are the additive
        surfaces ``stability`` (flip bands) and ``downside`` (B2 tail-risk).

        The golden was captured from UNMODIFIED pre-bands base code
        (e029cae2d); this test failing after a src change means the base wire
        drifted (beyond the known additive surfaces).
        """
        assert GOLDEN_PATH.exists(), (
            f"golden fixture missing at {GOLDEN_PATH} — regenerate via "
            "`poetry run python -m tests.unit.test_flip_stability_bands`"
        )
        resp = client.post(ENDPOINT, json=_variant_request(0), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        golden = json.loads(GOLDEN_PATH.read_text())
        current = _strip_edge_e_value_additions(
            _strip_additive_surfaces(normalize_v2_payload(resp.json()))
        )
        # See _CHANGED_WIRE_VALUES: one value deliberately moved; drop it from BOTH
        # sides so this pin keeps guarding everything else byte-for-byte.
        section, key = _CHANGED_WIRE_VALUES
        current.get(section, {}).pop(key, None)
        golden.get(section, {}).pop(key, None)
        assert current == golden

    def test_confidence_is_now_the_bare_stability_fraction(self, no_env, client):
        """Pin the one value the golden comparison above excludes.

        Base wire: min(0.99, recommendation_stability * (1 - 1/sqrt(n_samples))).
        Now: recommendation_stability, unmodified — see
        RobustnessAnalyzerV2._stability_confidence_figure. Without this the
        exclusion above would be a hole in the pin."""
        resp = client.post(ENDPOINT, json=_variant_request(0), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        robustness = resp.json()["robustness"]
        assert robustness["confidence"] == robustness["recommendation_stability"]
        assert robustness["confidence_basis"] == "recommendation_stability_uncalibrated"

    def test_stability_present_before_stripping(self, no_env, client):
        """Positive control for the golden pin: the strip in the test above
        must actually be removing something, or the modulo comparison silently
        degenerates into 'wire unchanged'."""
        resp = client.post(ENDPOINT, json=_variant_request(0), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        assert _strip_additive_surfaces(resp.json()) != resp.json()


# ---------------------------------------------------------------------------
# Band shape and statistics
# ---------------------------------------------------------------------------


class TestBandShape:
    def test_bands_attached_to_every_edge_e_value_entry(self, no_env):
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        blocks = _stability_blocks(response)
        assert all(
            isinstance(b, dict) for b in blocks
        ), "every edge_e_values entry must carry a stability band"
        for block in blocks:
            _assert_block_shape(block, N_SEEDS)

    def test_band_statistics_consistent_with_seed_values(self, no_env):
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        for block in _stability_blocks(response):
            flipped = [v for v in block["seed_flip_means"] if v is not None]
            assert block["n_seeds_flipped"] == len(flipped)
            if not flipped:
                # band_* keys are omitted entirely when no seed flips
                assert not (STABILITY_BAND_KEYS & set(block.keys()))
                continue
            assert block["band_min"] == pytest.approx(min(flipped), abs=1e-6)
            assert block["band_max"] == pytest.approx(max(flipped), abs=1e-6)
            assert block["band_median"] == pytest.approx(statistics.median(flipped), abs=1e-6)
            assert block["band_width"] == pytest.approx(
                block["band_max"] - block["band_min"], abs=1e-6
            )
            assert block["band_min"] <= block["band_median"] <= block["band_max"]

    def test_at_least_one_band_has_nonzero_width(self, no_env):
        """The sweep must actually expose threshold uncertainty on this graph.

        sample_variants[2] has wide strength stds; a sweep whose bands all
        collapse to zero width would mean the child sweeps are not actually
        varying the background (the false-stability failure mode itself).
        """
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        widths = [
            b["band_width"] for b in _stability_blocks(response) if b.get("band_width") is not None
        ]
        assert widths, "expected at least one edge with a computable band"
        assert any(w > 0 for w in widths)


# ---------------------------------------------------------------------------
# Determinism — the sweep itself is seeded
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_request_same_seed_byte_identical_bands(self, no_env):
        first = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        second = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        assert json.dumps(_stability_blocks(first), sort_keys=True) == json.dumps(
            _stability_blocks(second), sort_keys=True
        )

    def test_different_master_seed_changes_bands(self, no_env):
        """Child seeds derive from the master seed: a different request seed
        must produce a different sweep (different sampled backgrounds)."""
        seed_a = RobustnessAnalyzerV2().analyze(_analyzer_request(2, seed=42))
        seed_b = RobustnessAnalyzerV2().analyze(_analyzer_request(2, seed=20260711))
        assert _stability_blocks(seed_a) != _stability_blocks(seed_b)


# ---------------------------------------------------------------------------
# Budget degradation — all-or-nothing, disclosed, base wire untouched
# ---------------------------------------------------------------------------


class TestBudgetDegradation:
    """The unchanged honest-degradation contract, now on the DEFAULT path.

    On budget exceed the sweep attaches NOTHING (partial bands would bias
    readers toward whichever edges computed first), never mutates the base
    edge_e_values, and discloses via the ``flip_stability_budget_exceeded``
    structured log event. Positive control first: the same request WITHOUT
    the exhausted budget DOES attach bands — so the absence assertions below
    are proven able to see a presence.
    """

    def test_budget_exhaustion_attaches_no_bands_and_discloses(self, no_env, monkeypatch, caplog):
        # Positive control: bands present under the real budget.
        control = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        assert all(
            isinstance(b, dict) for b in _stability_blocks(control)
        ), "positive control failed: bands absent even before exhausting the budget"

        # Exhaust the budget: any elapsed wall time exceeds -1 ms immediately.
        monkeypatch.setattr(RobustnessAnalyzerV2, "FLIP_STABILITY_BUDGET_MS", -1)
        with caplog.at_level(logging.INFO, logger="src.services.robustness_analyzer_v2"):
            degraded = RobustnessAnalyzerV2().analyze(_analyzer_request(2))

        # All-or-nothing: NO entry carries a band.
        assert degraded.edge_e_values, "base edge_e_values must survive budget exhaustion"
        for entry in degraded.edge_e_values:
            assert (
                "stability" not in entry
            ), "budget exhaustion must attach NO bands (all-or-nothing honest absence)"
        # Honest disclosure: the structured log event fired, WITH elapsed_ms
        # (Paul ruling 17 Jul: find out when something is slow — the event
        # must carry how slow).
        exceeded = [
            record
            for record in caplog.records
            if record.message == "flip_stability_budget_exceeded"
        ]
        assert exceeded, "budget degradation must be disclosed via flip_stability_budget_exceeded"
        assert hasattr(exceeded[0], "elapsed_ms"), "disclosure event must carry elapsed_ms"

    def test_budget_exhaustion_leaves_base_values_untouched(self, no_env, monkeypatch):
        control = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        monkeypatch.setattr(RobustnessAnalyzerV2, "FLIP_STABILITY_BUDGET_MS", -1)
        degraded = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        # Strip ONLY the band, the surface this test controls for: the winner-capture
        # fields (ROADMAP 2.228-F3) are part of the base e-values here and must be
        # compared, not hidden — a band budget trip has no business changing them.
        assert (
            _strip_additive_surfaces(control.edge_e_values, {"stability"})
            == degraded.edge_e_values
        )


# ---------------------------------------------------------------------------
# V2 wire serialisation
# ---------------------------------------------------------------------------


class TestV2Wire:
    def test_stability_serialised_on_v2_wire(self, no_env, client):
        resp = client.post(ENDPOINT, json=_variant_request(2), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        edge_e_values = (data.get("robustness") or {}).get("edge_e_values")
        assert edge_e_values, "expected edge_e_values on the v2 wire"
        for entry in edge_e_values:
            block = entry.get("stability")
            assert isinstance(block, dict), "default v2 wire must carry stability bands"
            _assert_block_shape(block, N_SEEDS)


# ---------------------------------------------------------------------------
# Zero-flip omission branch — forced, so the branch is GUARANTEED to execute
# ---------------------------------------------------------------------------


class TestZeroFlipOmission:
    """The n_seeds_flipped == 0 omission contract, exercised deterministically.

    No fixture graph guarantees a zero-flip band (the shape assertions above
    only cover the branch IF a fixture happens to produce one), so these
    tests force it: every background is made to admit no flip by patching
    ``_flip_mean_under_background`` to return None — the exact value the real
    search returns when no perturbation within [-1, 1] flips the winner.
    """

    @staticmethod
    def _force_no_flip(monkeypatch):
        monkeypatch.setattr(
            RobustnessAnalyzerV2,
            "_flip_mean_under_background",
            lambda self, request, evaluator, edge, background: None,
        )

    def test_forced_zero_flip_analyzer_branch(self, no_env, monkeypatch):
        """v1 dict-passthrough shape: band_* keys OMITTED (not null) at zero flips."""
        self._force_no_flip(monkeypatch)
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        blocks = _stability_blocks(response)
        assert blocks, "expected stability blocks on every edge_e_values entry"
        for block in blocks:
            assert block["n_seeds_flipped"] == 0
            assert block["seed_flip_means"] == [None] * N_SEEDS
            # The omission branch itself: exactly the 3 required keys, no band_*
            assert set(block.keys()) == STABILITY_REQUIRED_KEYS
            _assert_block_shape(block, N_SEEDS)

    def test_forced_zero_flip_v2_wire_parity(self, no_env, client, monkeypatch):
        """v2 model wire (exclude_none) carries the SAME zero-flip shape as v1.

        Pins the parity claim end-to-end: band_* omitted by exclude_none,
        while the None ELEMENTS inside seed_flip_means survive as JSON nulls.
        """
        self._force_no_flip(monkeypatch)
        resp = client.post(ENDPOINT, json=_variant_request(2), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        edge_e_values = (resp.json().get("robustness") or {}).get("edge_e_values")
        assert edge_e_values, "expected edge_e_values on the v2 wire"
        for entry in edge_e_values:
            block = entry.get("stability")
            assert isinstance(block, dict)
            assert set(block.keys()) == STABILITY_REQUIRED_KEYS
            assert block["n_seeds_flipped"] == 0
            assert block["seed_flip_means"] == [None] * N_SEEDS

    def test_flip_stability_band_v2_exclude_none_serialisation(self):
        """Direct model pin: the exact model_dump the v2 wire uses
        (src/api/robustness.py: by_alias=True, exclude_none=True) drops the
        four unset band_* fields but preserves None list elements."""
        band = FlipStabilityBandV2(
            n_seeds=N_SEEDS,
            n_seeds_flipped=0,
            seed_flip_means=[None] * N_SEEDS,
        )
        dumped = band.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {
            "n_seeds": N_SEEDS,
            "n_seeds_flipped": 0,
            "seed_flip_means": [None] * N_SEEDS,
        }


# ---------------------------------------------------------------------------
# Golden capture — base wire modulo the additive stability surface
# ---------------------------------------------------------------------------


def _capture_golden() -> None:  # pragma: no cover
    """Capture the golden as wire-modulo-stability.

    The original golden was captured from UNMODIFIED pre-bands base code with
    the (then) flag off. Now that bands are default-on, an equivalent golden
    is the current wire with every additive ``stability`` key stripped — if
    the modulo-stability wire has drifted from base, regenerating will make
    the drift visible in the fixture diff rather than hiding it.
    """
    local_client = TestClient(app)
    resp = local_client.post(ENDPOINT, json=_variant_request(0), headers=V2_HEADERS)
    if resp.status_code != 200:  # pragma: no cover
        raise SystemExit(f"capture failed: {resp.status_code} {resp.text}")
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(
            _strip_edge_e_value_additions(
                _strip_additive_surfaces(normalize_v2_payload(resp.json()))
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"golden written: {GOLDEN_PATH}")


if __name__ == "__main__":  # pragma: no cover
    _capture_golden()
