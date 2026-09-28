"""Read-only adapter around the frozen R3-B research evaluator.

No graph, mapping, production service or model function is modified here.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from regions import ROOT, Refusal

PINNED_COMMIT = "fdb51e84a673993de7cddb85d5b963741941b80a"
GRAPH_ID = "pj-20260927T180910Z-A"
MODEL_IDS = ("X_net_reading", "X_gross_reading")
SOURCE_ROOT = ROOT.parent / "r3b_sim"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim.corpus import load_graph, sha256_file, verify_corpus  # noqa: E402
from sim.dynamic import run_model  # noqa: E402
from sim.frames import canon_unit, node_unit  # noqa: E402
from sim.run import model_evals, simulate  # noqa: E402
from sim.spec import load_valid_spec  # noqa: E402
from sim.static import point_draws  # noqa: E402


def source_hash() -> str:
    h = hashlib.sha256()
    for name in ("sim/corpus.py", "sim/dynamic.py", "sim/edges.py", "sim/frames.py", "sim/run.py", "sim/spec.py", "sim/static.py"):
        h.update(name.encode() + b"\0" + (SOURCE_ROOT / name).read_bytes())
    return h.hexdigest()


def verify_source() -> dict[str, Any]:
    frozen = {}
    for line in (SOURCE_ROOT / "FROZEN.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        frozen[name] = sha256_file(SOURCE_ROOT / name.strip()) == digest
    corpus = verify_corpus()
    if not all(frozen.values()) or not all(corpus.values()):
        raise Refusal("MODEL_IDENTITY_MISMATCH", "R3-B frozen mapping or corpus hash failed")
    return {"commit": PINNED_COMMIT, "frozen_files": frozen, "corpus_files": corpus, "evaluator_sha256": source_hash(), "graph_sha256": sha256_file(SOURCE_ROOT / "corpus" / f"{GRAPH_ID}.json"), "mapping_sha256": sha256_file(SOURCE_ROOT / "mapping" / f"{GRAPH_ID}.json")}


def _grid(lo: str, hi: str, n: int) -> list[float]:
    from fractions import Fraction
    a, b = Fraction(lo), Fraction(hi)
    return [float(a + (b - a) * Fraction(i, n - 1)) for i in range(n)]


class PricingAdapter:
    def __init__(self, model_id: str):
        if model_id not in MODEL_IDS:
            raise Refusal("UNSUPPORTED_SEMANTICS", model_id)
        self.identity = verify_source()
        self.graph = load_graph(GRAPH_ID)
        self.spec = load_valid_spec(self.graph)
        self.model = next(m for m in self.spec["dynamic_models"] if m["id"] == model_id)
        self.draws = point_draws(self.graph)
        self.model_id = model_id
        self.binding_units = {"X_price_churn_direct": "percentage points per +£10", "X_feature_competitive_mrr": "GBP per month"}
        x_effects = {e["id"]: e for e in self.spec["x_effects"]}
        if not set(self.binding_units).issubset(x_effects):
            raise Refusal("UNRESOLVED_BINDING", "frozen mapping lacks required effects")
        if x_effects["X_price_churn_direct"]["amount_unit"] != "percentage points" or x_effects["X_price_churn_direct"]["per_source_change"] != 10:
            raise Refusal("UNRESOLVED_BINDING", "churn effect units changed")
        if x_effects["X_feature_competitive_mrr"]["amount_unit"] != "GBP per month":
            raise Refusal("UNRESOLVED_BINDING", "competitive effect units changed")
        if self.spec["goal"]["temporal_semantics"] != "attain_by_H" or self.spec["horizon"]["months"] != 12:
            raise Refusal("UNSUPPORTED_SEMANTICS", "goal or horizon differs from frozen case")
        constraint = next(c for c in self.graph.constraints if c["node_id"] == "monthly_churn")
        if constraint["operator"] != "<=" or float(constraint["value"]) != 4 or canon_unit(constraint["unit"]) != node_unit(self.graph, "monthly_churn"):
            raise Refusal("UNSUPPORTED_SEMANTICS", "churn constraint binding changed")
        self.constraint = constraint

    def request(self, n: int) -> dict[str, Any]:
        xs, ys = _grid("0", "17/4", n), _grid("-31250", "0", n)
        graph, spec = self.graph, self.spec
        return {
            "schema_version": "RegionRequestV1", "request_id": f"r3b-{self.model_id}-{n}",
            "source": {"repository": "Talchain/Inference-Service-Layer", "commit": PINNED_COMMIT, "graph_sha256": self.identity["graph_sha256"], "mapping_sha256": self.identity["mapping_sha256"], "tier": "X", "model_id": self.model_id, "evaluator_sha256": self.identity["evaluator_sha256"], "adapter_version": "r3b-adapter-v1"},
            "case": GRAPH_ID, "option_ids": graph.option_ids, "option_labels": {oid: graph.nodes[oid]["label"] for oid in graph.option_ids}, "baseline_id": graph.baseline_option,
            "comparison_set": list(spec["brief_decision_pair"]) + ["49_with_feature_release"],
            "axes": [
                {"id": "churn_response", "label": "Churn response to £10 rise", "unit": self.binding_units["X_price_churn_direct"], "role": "SCENARIO_ASSUMPTION", "operation": "FIX_SCENARIO_PARAMETER", "binding": "X_price_churn_direct", "grid_values": xs, "domain_source": "Exploratory interpolation between frozen X sweep and break-even bracket; not evidence-backed plausibility", "valid_min": 0, "valid_max": 4.25, "joint_support": None},
                {"id": "competitive_response", "label": "Feature competitive response", "unit": self.binding_units["X_feature_competitive_mrr"], "role": "SCENARIO_ASSUMPTION", "operation": "FIX_SCENARIO_PARAMETER", "binding": "X_feature_competitive_mrr", "grid_values": ys, "domain_source": "Exploratory frozen X sweep endpoints; no joint probability interpretation", "valid_min": -31250, "valid_max": 0, "joint_support": None}
            ],
            "fixed_assumptions": [{"id": "X_price_churn_via_sensitivity", "value": 0, "source": "frozen break-even reading"}] + [{"id": name, "value": True, "source": "frozen model " + self.model_id} for name in sorted(self.model["defaults"] + self.model["assumptions"])],
            "goal": {"id": "mrr_goal", "metric": "mrr", "unit": "GBP per month", "operator": ">=", "limit": float(graph.resolve(spec["goal"]["threshold_ptr"])), "temporal_rule": "FIRST_PASSAGE_BY_H"},
            "objective": {"metric": "mrr", "unit": "GBP per month", "direction": "MAXIMISE", "functional": "POINT_VALUE", "temporal_rule": "TERMINAL", "delta": 0, "delta_source": "research mathematical separation only"},
            "constraints": [{"id": self.constraint["constraint_id"], "metric": "monthly_churn", "unit": "percent per month", "operator": "<=", "limit": 4, "temporal_rule": "EACH_MONTH", "enforcement": "DETERMINISTIC_HARD"}],
            "execution": {"seed": 20260928, "numeric_policy": "FLOAT64_RECORDED", "max_seconds": 600, "max_memory_mib": 1024, "max_evaluations": 4000, "draw_count": 1, "confidence_scope": "NOT_APPLICABLE"},
            "comparison": {"baseline_status": "PINNED_LOCAL", "reference_id": sha256_file(SOURCE_ROOT / "breakeven.json"), "differences": ["No captured deployed-current comparator", "Terminal MRR objective explicit to this research request"]}
        }

    def validate_request_identity(self, request: dict[str, Any]) -> None:
        expected = self.request(len(request["axes"][0]["grid_values"]))
        if request["source"] != expected["source"] or request["option_ids"] != expected["option_ids"] or request["option_labels"] != expected["option_labels"] or request["baseline_id"] != expected["baseline_id"] or request["comparison_set"] != expected["comparison_set"]:
            raise Refusal("MODEL_IDENTITY_MISMATCH", "pinned source or declared options differ")
        if request["axes"] != expected["axes"] or request["goal"] != expected["goal"] or request["constraints"] != expected["constraints"] or request["fixed_assumptions"] != expected["fixed_assumptions"]:
            raise Refusal("MODEL_IDENTITY_MISMATCH", "frozen goal, constraints or assumptions differ")

    def _scenario(self, x: float, y: float) -> dict[str, float]:
        return {"X_price_churn_direct": x, "X_feature_competitive_mrr": y, "X_price_churn_via_sensitivity": 0.0}

    def measurements(self, x: float, y: float) -> dict[str, dict[str, Any]]:
        scenario = self._scenario(x, y)
        evals = model_evals(self.graph, self.spec, self.model, scenario, self.draws)
        sq = evals[self.graph.baseline_option].nodes
        out = {}
        for oid in self.graph.option_ids:
            ev = evals[oid]
            trajectory = run_model(self.graph, self.model, oid, ev.nodes if ev.computable else None, sq, 12, 3, self.draws.n)
            churn_node = ev.nodes.get("monthly_churn")
            churn = None if churn_node is None or churn_node.value is None else float(churn_node.value[0])
            path = None if trajectory.y is None else trajectory.y[0]
            if path is not None and (len(path) != 13 or not np.all(np.isfinite(path))):
                raise Refusal("EVALUATION_FAILED", f"bad trajectory {oid}")
            out[oid] = {
                "status": "COMPUTED" if path is not None else "PARTIAL",
                "goal_value": None if path is None else float(max(path)),
                "objective_value": None if path is None else float(path[-1]),
                "constraints": {} if churn is None else {self.constraint["constraint_id"]: churn},
                "tier": "X", "reason": None if path is not None else ";".join(sorted(set(ev.option_gaps + trajectory.gaps))) or "UNKNOWN_GAP",
            }
        return out

    def assert_composition(self, x: float, y: float) -> None:
        assembled = self.measurements(x, y)
        original = simulate(self.graph, self.spec, self.model, self._scenario(x, y), self.draws)
        for oid in self.graph.option_ids:
            path = original[oid].y
            terminal = None if path is None else float(path[0, -1])
            if assembled[oid]["objective_value"] != terminal:
                raise Refusal("MODEL_IDENTITY_MISMATCH", f"composed path differs from simulate for {oid}")


def reproduce_stored_thresholds() -> dict[str, Any]:
    from sim.breakeven import compute
    stored = json.loads((SOURCE_ROOT / "breakeven.json").read_text())
    actual = compute()
    if actual != stored:
        raise Refusal("MODEL_IDENTITY_MISMATCH", "stored R3-B break-even output differs from pinned code")
    return {"status": "MATCH", "stored_sha256": sha256_file(SOURCE_ROOT / "breakeven.json"), "models": stored["models"]}
