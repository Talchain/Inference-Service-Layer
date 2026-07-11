"""
Track S Phase 1 — seed-sweep flip-threshold stability bands (RED-first).

Why: the 2026-06-10 PLoT/ISL science-performance report found that flip
thresholds computed from a single seed are presented with false stability,
and recommended reporting flip thresholds with "a stability band from a
small seed sweep (e.g. 5 seeds)" with flip confidence based on band width
(report §5 "Flip-threshold drift", §7 recommendation, next-step #6).
The 2026-07-07 science-validation REPORT.md (§1, §5.1) reinforced that
single-seed flip evidence is weak.

Contract under test (flag-gated, ADDITIVE only):

- env ``ISL_FLIP_STABILITY_BANDS`` truthy ("1"/"true"/"yes"/"on") -> each
  ``edge_e_values[]`` entry gains a ``stability`` object::

      {
        "n_seeds": int,             # seeds swept (default 5)
        "n_seeds_flipped": int,     # seeds whose background admits a flip
        "band_min": float|None,     # min flip_mean across flipped seeds
        "band_median": float|None,  # median flip_mean across flipped seeds
        "band_max": float|None,     # max flip_mean across flipped seeds
        "band_width": float|None,   # band_max - band_min
        "seed_flip_means": [float|None, ...]  # per-child-seed flip means
      }

- env ``ISL_FLIP_STABILITY_SEEDS`` overrides N (default 5, per the report's
  recommendation); invalid values fall back to the default.
- flag OFF -> the wire is identical to the pre-change baseline, pinned by
  the golden fixture captured at origin/staging e029cae2d (comparison is
  modulo the four pre-existing volatile fields and the pre-existing
  set-ordering leak catalogued in docs/science-validation/REPORT.md §3 —
  those vary run-to-run/process-to-process on the BASE code already).
- determinism: the sweep is seeded — child seeds are SHA-256-derived from
  the master (request) seed, so same request+seed -> byte-identical bands.

Regenerate the golden fixture (flag OFF, unmodified base behaviour) with:

    poetry run python -m tests.unit.test_flip_stability_bands
"""

import copy
import json
import os
import statistics
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

FLAG_ENV = "ISL_FLIP_STABILITY_BANDS"
N_SEEDS_ENV = "ISL_FLIP_STABILITY_SEEDS"
DEFAULT_N_SEEDS = 5  # 06-10 report recommendation: "a small seed sweep (e.g. 5 seeds)"

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS_PATH = REPO_ROOT / "tests" / "benchmarks" / "sample_variants.json"
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "flip_stability" / "golden_flag_off_v2.json"

STABILITY_KEYS = {
    "n_seeds",
    "n_seeds_flipped",
    "band_min",
    "band_median",
    "band_max",
    "band_width",
    "seed_flip_means",
}


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


def _strip_stability(payload):
    """Recursively remove every 'stability' key (the additive surface)."""
    if isinstance(payload, dict):
        return {k: _strip_stability(v) for k, v in payload.items() if k != "stability"}
    if isinstance(payload, list):
        return [_strip_stability(item) for item in payload]
    return payload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG_ENV, raising=False)
    monkeypatch.delenv(N_SEEDS_ENV, raising=False)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "true")
    monkeypatch.delenv(N_SEEDS_ENV, raising=False)


def _stability_blocks(response) -> list:
    assert response.edge_e_values, "expected edge_e_values to be computed"
    return [entry.get("stability") for entry in response.edge_e_values]


# ---------------------------------------------------------------------------
# Flag OFF — the wire must be byte-identical to base (the pin)
# ---------------------------------------------------------------------------


class TestFlagOff:
    def test_no_stability_key_on_analyzer_output(self, flag_off):
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(0))
        assert response.edge_e_values, "expected edge_e_values to be computed"
        for entry in response.edge_e_values:
            assert "stability" not in entry, "flag-off must not attach stability bands"

    def test_v2_wire_matches_base_golden(self, flag_off, client):
        """Pin: flag-off wire identical to origin/staging base (golden fixture).

        The golden was captured from UNMODIFIED base code (e029cae2d); this
        test failing after a src change means the flag-off wire drifted.
        """
        assert GOLDEN_PATH.exists(), (
            f"golden fixture missing at {GOLDEN_PATH} — regenerate on BASE code via "
            "`poetry run python -m tests.unit.test_flip_stability_bands`"
        )
        resp = client.post(ENDPOINT, json=_variant_request(0), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        golden = json.loads(GOLDEN_PATH.read_text())
        assert normalize_v2_payload(resp.json()) == golden

    def test_v2_wire_has_no_stability_anywhere(self, flag_off, client):
        resp = client.post(ENDPOINT, json=_variant_request(2), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        assert _strip_stability(resp.json()) == resp.json()


# ---------------------------------------------------------------------------
# Flag ON — bands attached, additive-only, correct statistics
# ---------------------------------------------------------------------------


class TestFlagOnShape:
    def test_bands_attached_to_every_edge_e_value_entry(self, flag_on):
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        blocks = _stability_blocks(response)
        assert all(isinstance(b, dict) for b in blocks), (
            "every edge_e_values entry must carry a stability band when the flag is on"
        )
        for block in blocks:
            assert set(block.keys()) == STABILITY_KEYS
            assert block["n_seeds"] == DEFAULT_N_SEEDS
            assert len(block["seed_flip_means"]) == DEFAULT_N_SEEDS
            assert 0 <= block["n_seeds_flipped"] <= DEFAULT_N_SEEDS

    def test_band_statistics_consistent_with_seed_values(self, flag_on):
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        for block in _stability_blocks(response):
            flipped = [v for v in block["seed_flip_means"] if v is not None]
            assert block["n_seeds_flipped"] == len(flipped)
            if not flipped:
                assert block["band_min"] is None
                assert block["band_median"] is None
                assert block["band_max"] is None
                assert block["band_width"] is None
                continue
            assert block["band_min"] == pytest.approx(min(flipped), abs=1e-6)
            assert block["band_max"] == pytest.approx(max(flipped), abs=1e-6)
            assert block["band_median"] == pytest.approx(statistics.median(flipped), abs=1e-6)
            assert block["band_width"] == pytest.approx(
                block["band_max"] - block["band_min"], abs=1e-6
            )
            assert block["band_min"] <= block["band_median"] <= block["band_max"]

    def test_at_least_one_band_has_nonzero_width(self, flag_on):
        """The sweep must actually expose threshold uncertainty on this graph.

        sample_variants[2] has wide strength stds; a sweep whose bands all
        collapse to zero width would mean the child sweeps are not actually
        varying the background (the false-stability failure mode itself).
        """
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        widths = [
            b["band_width"]
            for b in _stability_blocks(response)
            if b["band_width"] is not None
        ]
        assert widths, "expected at least one edge with a computable band"
        assert any(w > 0 for w in widths)


class TestFlagOnAdditiveOnly:
    def test_flag_on_changes_nothing_but_stability(self, monkeypatch):
        monkeypatch.delenv(FLAG_ENV, raising=False)
        monkeypatch.delenv(N_SEEDS_ENV, raising=False)
        response_off = RobustnessAnalyzerV2().analyze(_analyzer_request(0))

        monkeypatch.setenv(FLAG_ENV, "1")
        response_on = RobustnessAnalyzerV2().analyze(_analyzer_request(0))

        assert any(
            "stability" in entry for entry in response_on.edge_e_values
        ), "flag-on must attach at least one stability band"

        def comparable(resp):
            dump = resp.model_dump()
            dump["metadata"].pop("execution_time_ms", None)
            for critique in dump.get("critiques") or []:
                critique.pop("id", None)
            return _strip_stability(dump)

        assert comparable(response_on) == comparable(response_off), (
            "flag-on must be additive-only: stripping 'stability' must recover "
            "the flag-off response exactly (same process, volatile fields masked)"
        )


# ---------------------------------------------------------------------------
# Determinism — the sweep itself is seeded
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_request_same_seed_byte_identical_bands(self, flag_on):
        first = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        second = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        assert json.dumps(_stability_blocks(first), sort_keys=True) == json.dumps(
            _stability_blocks(second), sort_keys=True
        )

    def test_different_master_seed_changes_bands(self, flag_on):
        """Child seeds derive from the master seed: a different request seed
        must produce a different sweep (different sampled backgrounds)."""
        seed_a = RobustnessAnalyzerV2().analyze(_analyzer_request(2, seed=42))
        seed_b = RobustnessAnalyzerV2().analyze(_analyzer_request(2, seed=20260711))
        assert _stability_blocks(seed_a) != _stability_blocks(seed_b)


# ---------------------------------------------------------------------------
# N configurability
# ---------------------------------------------------------------------------


class TestNSeedsConfig:
    def test_n_seeds_env_override(self, monkeypatch):
        monkeypatch.setenv(FLAG_ENV, "1")
        monkeypatch.setenv(N_SEEDS_ENV, "3")
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        for block in _stability_blocks(response):
            assert block["n_seeds"] == 3
            assert len(block["seed_flip_means"]) == 3

    def test_invalid_n_seeds_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(FLAG_ENV, "1")
        monkeypatch.setenv(N_SEEDS_ENV, "not-a-number")
        response = RobustnessAnalyzerV2().analyze(_analyzer_request(2))
        for block in _stability_blocks(response):
            assert block["n_seeds"] == DEFAULT_N_SEEDS


# ---------------------------------------------------------------------------
# V2 wire serialisation
# ---------------------------------------------------------------------------


class TestV2Wire:
    def test_stability_serialised_on_v2_wire_when_flag_on(self, flag_on, client):
        resp = client.post(ENDPOINT, json=_variant_request(2), headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        edge_e_values = (data.get("robustness") or {}).get("edge_e_values")
        assert edge_e_values, "expected edge_e_values on the v2 wire"
        for entry in edge_e_values:
            block = entry.get("stability")
            assert isinstance(block, dict), "flag-on v2 wire must carry stability bands"
            assert set(block.keys()) == STABILITY_KEYS


# ---------------------------------------------------------------------------
# Golden capture (run on BASE code only)
# ---------------------------------------------------------------------------


def _capture_golden() -> None:  # pragma: no cover
    if os.environ.get(FLAG_ENV):
        raise SystemExit(f"unset {FLAG_ENV} before capturing the flag-off golden")
    local_client = TestClient(app)
    resp = local_client.post(ENDPOINT, json=_variant_request(0), headers=V2_HEADERS)
    if resp.status_code != 200:  # pragma: no cover
        raise SystemExit(f"capture failed: {resp.status_code} {resp.text}")
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(normalize_v2_payload(resp.json()), indent=2, sort_keys=True) + "\n"
    )
    print(f"golden written: {GOLDEN_PATH}")


if __name__ == "__main__":  # pragma: no cover
    _capture_golden()
