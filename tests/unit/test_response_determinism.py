"""
Same-seed response determinism tests (science-validation report §3 / §5.7).

The merged science-validation report (docs/science-validation/REPORT.md)
catalogued two src-level determinism leaks on same-seed responses:

  a. ``critiques[].id`` was ``uuid.uuid4()`` per run (src/models/critique.py),
     so any response containing a critique could never be byte-identical to a
     same-seed re-run.
  b. ``robustness.fragile_edges`` / ``robustness.robust_edges`` were
     materialised from sets, so their order followed the per-process string
     hash salt — and ``robustness.interpretation`` cites the first three
     entries, i.e. a process-dependent arbitrary subset.

These tests pin the fixes: deterministic content-derived critique ids and
canonically sorted edge lists, byte-stable across same-seed runs in the same
process and across processes with different hash salts.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.models.critique import (
    DEGENERATE_OPTION_ZERO_VARIANCE,
    EDGE_STRENGTH_OUT_OF_RANGE,
    deterministic_critique_id,
)
from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Shared fixture: a request that produces >=3 fragile edges, >=4 robust edges,
# and at least one analysis critique (zero-variance option), fully seeded.
# =============================================================================


def make_determinism_request() -> RobustnessRequestV2:
    graph = GraphV2(
        nodes=[
            NodeV2(
                id="n_base",
                kind="factor",
                label="Baseline revenue",
                observed_state=ObservedState(value=2.0),
            ),
            NodeV2(id="n_mkt", kind="factor", label="Marketing"),
            NodeV2(id="n_price", kind="factor", label="Price"),
            NodeV2(id="n_qual", kind="factor", label="Quality"),
            NodeV2(id="n_dem", kind="chance", label="Demand"),
            NodeV2(id="n_churn", kind="chance", label="Churn"),
            NodeV2(id="n_rev", kind="outcome", label="Revenue"),
            NodeV2(id="n_flat", kind="outcome", label="Flat"),
        ],
        edges=[
            EdgeV2(
                **{"from": "n_base", "to": "n_rev"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=1.0, std=0.011),
            ),
            EdgeV2(
                **{"from": "n_mkt", "to": "n_dem"},
                exists_probability=0.6,
                strength=StrengthDistribution(mean=0.9, std=0.3),
            ),
            EdgeV2(
                **{"from": "n_price", "to": "n_dem"},
                exists_probability=0.55,
                strength=StrengthDistribution(mean=-0.8, std=0.3),
            ),
            EdgeV2(
                **{"from": "n_qual", "to": "n_churn"},
                exists_probability=0.5,
                strength=StrengthDistribution(mean=-0.9, std=0.35),
            ),
            EdgeV2(
                **{"from": "n_dem", "to": "n_rev"},
                exists_probability=0.65,
                strength=StrengthDistribution(mean=1.0, std=0.3),
            ),
            EdgeV2(
                **{"from": "n_churn", "to": "n_rev"},
                exists_probability=0.6,
                strength=StrengthDistribution(mean=-0.9, std=0.3),
            ),
            EdgeV2(
                **{"from": "n_mkt", "to": "n_rev"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.01, std=0.01),
            ),
            EdgeV2(
                **{"from": "n_qual", "to": "n_flat"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.02, std=0.01),
            ),
        ],
    )
    options = [
        InterventionOption(id="opt_a", label="Push marketing", interventions={"n_mkt": 1.0}),
        InterventionOption(id="opt_b", label="Cut price", interventions={"n_price": -1.0}),
        # Intervenes directly on the goal -> zero variance -> critique fires.
        InterventionOption(id="opt_goal", label="Fix goal", interventions={"n_rev": 0.5}),
    ]
    return RobustnessRequestV2(
        request_id="determinism-test-001",
        graph=graph,
        options=options,
        goal_node_id="n_rev",
        n_samples=200,
        seed=4242,
        parameter_uncertainties=[
            ParameterUncertainty(node_id="n_base", distribution="point_mass"),
        ],
    )


# Script run in child interpreters with different PYTHONHASHSEED values.
# Reuses make_determinism_request() from this module so the request cannot
# drift from the in-process tests.
_CHILD_SCRIPT = """
import json
import sys

sys.path.insert(0, {repo_root!r})
sys.path.insert(0, {test_dir!r})

from test_response_determinism import make_determinism_request
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

response = RobustnessAnalyzerV2().analyze(make_determinism_request())
payload = response.model_dump(mode="json", by_alias=True)
# The only sanctioned volatile field on the in-process response: wall clock.
# (by_alias serialises the metadata field under its "_metadata" alias.)
payload["_metadata"]["execution_time_ms"] = 0
sys.stdout.write(json.dumps(payload, sort_keys=False))
"""


def _run_child(hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["ISL_AUTH_DISABLED"] = "true"
    script = _CHILD_SCRIPT.format(
        repo_root=str(REPO_ROOT),
        test_dir=str(REPO_ROOT / "tests" / "unit"),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert result.returncode == 0, f"child interpreter failed:\n{result.stderr}"
    return result.stdout


# =============================================================================
# Deterministic critique ids (report §5.7a)
# =============================================================================


class TestDeterministicCritiqueId:
    def test_same_inputs_same_id(self):
        """Identical critique content must always yield the identical id."""
        a = DEGENERATE_OPTION_ZERO_VARIANCE.build(
            option_label="Option A", affected_option_ids=["opt_a"]
        )
        b = DEGENERATE_OPTION_ZERO_VARIANCE.build(
            option_label="Option A", affected_option_ids=["opt_a"]
        )
        assert a.id == b.id
        assert a.id.startswith("critique_")

    def test_different_content_different_id(self):
        """Template vars distinguish critiques sharing code + affected ids."""
        a = EDGE_STRENGTH_OUT_OF_RANGE.build(from_node="x", to_node="y", value=4.2)
        b = EDGE_STRENGTH_OUT_OF_RANGE.build(from_node="x", to_node="z", value=4.2)
        assert a.id != b.id

    def test_seed_participates_in_id(self):
        """Report §5.7a: the id derives from (seed, code, affected ids...)."""
        a = DEGENERATE_OPTION_ZERO_VARIANCE.build(
            option_label="Option A", affected_option_ids=["opt_a"], seed=1
        )
        b = DEGENERATE_OPTION_ZERO_VARIANCE.build(
            option_label="Option A", affected_option_ids=["opt_a"], seed=2
        )
        assert a.id != b.id

    def test_helper_is_pure(self):
        assert deterministic_critique_id(
            "CODE", "msg", ["o1"], ["n1"], seed=7
        ) == deterministic_critique_id("CODE", "msg", ["o1"], ["n1"], seed=7)
        assert deterministic_critique_id("CODE", "msg") != deterministic_critique_id(
            "CODE", "other msg"
        )


# =============================================================================
# Same-seed stability of the touched fields (in-process)
# =============================================================================


class TestSameSeedFieldStability:
    def test_two_runs_identical_critique_ids_and_edge_lists(self):
        """Two fresh analyzers, same seed/request -> identical volatile fields."""
        request = make_determinism_request()
        first = RobustnessAnalyzerV2().analyze(request)
        second = RobustnessAnalyzerV2().analyze(request)

        assert first.critiques, "fixture must produce at least one critique"
        assert [c.id for c in first.critiques] == [c.id for c in second.critiques]
        assert first.robustness.fragile_edges == second.robustness.fragile_edges
        assert first.robustness.robust_edges == second.robustness.robust_edges
        assert first.robustness.interpretation == second.robustness.interpretation

    def test_edge_lists_canonically_sorted(self):
        """fragile/robust edges must be in sorted order, not set order (§5.7b)."""
        response = RobustnessAnalyzerV2().analyze(make_determinism_request())
        fragile = response.robustness.fragile_edges
        robust = response.robustness.robust_edges
        assert len(fragile) >= 3, "fixture must produce several fragile edges"
        assert len(robust) >= 3, "fixture must produce several robust edges"
        assert fragile == sorted(fragile)
        assert robust == sorted(robust)

    def test_interpretation_cites_canonical_fragile_edges(self):
        """The 'sensitive to:' list must cite the sorted head of fragile_edges."""
        response = RobustnessAnalyzerV2().analyze(make_determinism_request())
        fragile = response.robustness.fragile_edges
        interpretation = response.robustness.interpretation
        if "sensitive to:" in interpretation:
            cited = interpretation.split("sensitive to:")[1].strip().split(", ")
            assert cited == sorted(fragile)[:3]


# =============================================================================
# Cross-process byte stability (report §3 cross-process finding)
# =============================================================================


class TestCrossProcessByteStability:
    def test_same_seed_byte_identical_across_hash_salts(self):
        """Same-seed responses from interpreters with different hash salts must
        be byte-identical once the wall-clock field is zeroed.

        Pre-fix this failed two ways: uuid4 critique ids differed on every
        run, and fragile/robust edge order (plus the interpretation string
        derived from it) followed PYTHONHASHSEED.
        """
        out_a = _run_child("0")
        out_b = _run_child("1")
        assert out_a == out_b

        payload = json.loads(out_a)
        assert payload["critiques"], "fixture must produce at least one critique"
        assert len(payload["robustness"]["fragile_edges"]) >= 3
        assert len(payload["robustness"]["robust_edges"]) >= 3
