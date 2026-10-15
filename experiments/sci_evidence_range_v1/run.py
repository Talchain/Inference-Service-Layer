#!/usr/bin/env python3
"""SCI-EVIDENCE range elicitation v1: before/after one HYPOTHETICAL user-supplied range, existing computation only.

Brief: olumi-programme-docs#75 5909728656. Corrected scope: Science plan review 5911560060.

Three parts, each on code that already exists and is consumed read-only from a pinned checkout:

  r3b      The frozen R3-B evaluator (ISL `fdb51e84`, consumed via SCI-REGIONS' `r3b_adapter.PricingAdapter` at
           `68e8c887`). Base monthly churn is swept across the hypothetical 2-4 % range by changing only the churn
           node's held level on an in-memory copy of the graph. Results are PER OPTION (constraint and goal). There is
           no option ranking, no leader and no EVPPI (Science corrections 1 and 2). No distribution is assumed.
  control  Paul's current MRR graph (A-graph) on served ISL `f7f19e3` (this checkout's `src/`), in-process through the
           v2 route, with the exact ISL request PLoT sent (seed 1254899477). Before = as sent; after = a user spread on
           churn. The production engine needs a sigma, so this is the ONLY place a distributional assumption is made.
  cards    The frozen SCI-EVIDENCE Lab adapter (olumi-programme-docs `ef108360`, `lab/w1_card.card_from_served`)
           on the real served MRR reload (the Lab's pinned PRIMARY input) and on the control's before/after bodies.

Run from the ISL repo root (after `poetry install`):

    poetry run python experiments/sci_evidence_range_v1/run.py \
        --r3b-root <ISL @ 68e8c887>/experiments \
        --control-root <ISL @ 6f7f9b43>/acceptance-evidence/structural-robustness-20260930 \
        --evidence-root <olumi-programme-docs @ ef108360>/research/sci-evidence-v1

Writes results/*.json, manifest.json, payload.json and coaching.md in this directory. Deterministic: no timestamps,
no wall-clock values. No network, no provider calls.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ISL_ROOT = HERE.parents[1]

# ---- the experimental input (fixed before any run; never tuned) -------------------------------------------------
RANGE_LOW = 2.0
RANGE_HIGH = 4.0
RANGE_UNIT = "percent per month"
RANGE_LABEL = (
    "HYPOTHETICAL USER-SUPPLIED range, experimental control only: not evidence, not a claimed real-world "
    "distribution, not Olumi's estimate"
)
CENTRE_NOTE = "The centre stays Olumi's 3 % estimate; the user was asked only for a low and a high."

# ---- the case ---------------------------------------------------------------------------------------------------
CHURN = "monthly_churn"
MODELS = (
    "X_net_reading",
    "X_gross_reading",
)  # N is SCI-REGIONS' model; G is the frozen alternative reading
REFERENCE = (
    0.5,
    0.0,
)  # SCI-REGIONS' evaluated reference: churn response 0.5 pp per +£10, competitive response £0
TIE = 1e-9  # AIQ tie band (ISL #212): within 1e-9 * max(1, |t|) is ON the threshold, and "<=" / ">=" hold there
SUBJECT_OPTION = (
    "59_with_feature_release"  # SCI-REGIONS' threshold subject; every computable option is reported
)


def _grid(lo: str, hi: str, n: int) -> List[float]:
    a, b = Fraction(lo), Fraction(hi)
    return [float(a + (b - a) * Fraction(i, n - 1)) for i in range(n)]


BASE_GRID = _grid("2", "4", 41)  # 0.05 pp steps across the hypothetical range
ANCHORS = (2.0, 3.0, 4.0)  # low, Olumi's point, high
X_GRID = _grid("0", "17/4", 41)  # SCI-REGIONS' own churn-response axis


def r6(x: float) -> float:
    return float(round(float(x), 6))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def git_head(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - the manifest records absence rather than failing
        return "unavailable"


def git_tree(path: Path, sub: str, rev: str = "HEAD") -> str:
    """The tree hash of `sub` at `rev`; for HEAD, only if the working tree has no change under `sub`."""
    try:
        if (
            rev == "HEAD"
            and subprocess.run(
                ["git", "status", "--porcelain", "--", sub],
                cwd=path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        ):
            return "dirty"
        return subprocess.run(
            ["git", "rev-parse", f"{rev}:{sub}"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def verdict(value: float, threshold: float, operator: str) -> str:
    """Inclusive limits: ON_THRESHOLD still satisfies '<=' / '>='; it is named so no margin is implied."""
    if abs(value - threshold) <= TIE * max(1.0, abs(threshold)):
        return "ON_THRESHOLD"
    if operator == "<=":
        return "MET" if value < threshold else "BREACHED"
    return "MET" if value > threshold else "MISSED"


def satisfied(v: str) -> bool:
    return v in ("MET", "ON_THRESHOLD")


# =================================================================================================================
# Part 1: R3-B (frozen evaluator), base churn swept across the hypothetical range
# =================================================================================================================


class R3B:
    """One frozen PricingAdapter whose graph can be re-held at a different base churn (nothing else changes)."""

    def __init__(self, r3b_root: Path, model_id: str):
        regions = r3b_root / "sci_regions_v1"
        if str(regions) not in sys.path:
            sys.path.insert(0, str(regions))
        import r3b_adapter  # noqa: E402  (frozen, read-only)
        from sim.corpus import Graph  # noqa: E402
        from sim.frames import held_level  # noqa: E402

        self.adapter = r3b_adapter.PricingAdapter(model_id)
        self.model_id = model_id
        self.Graph = Graph
        self.held_level = held_level
        self.doc0 = copy.deepcopy(self.adapter.graph.doc)
        self.request = self.adapter.request(41)
        self.goal_limit = float(self.request["goal"]["limit"])
        self.goal_op = self.request["goal"]["operator"]
        c = self.request["constraints"][0]
        self.cid, self.limit, self.limit_op = c["id"], float(c["limit"]), c["operator"]
        node = self.adapter.graph.nodes[CHURN]
        self.churn_node = copy.deepcopy(node)
        if node.get("scale_frame") != 100 or node["observed_state"].get("raw_value") != 3:
            raise SystemExit("frozen churn node changed: expected raw_value 3 on scale_frame 100")

    def hold(self, base: float) -> None:
        doc = copy.deepcopy(self.doc0)
        node = next(n for n in doc["draft_graph"]["nodes"] if n["id"] == CHURN)
        node["observed_state"]["raw_value"] = base
        node["observed_state"]["value"] = base / 100.0
        self.adapter.graph = self.Graph(self.adapter.graph.id, doc)
        level = self.held_level(self.adapter.graph, CHURN)
        if level is None or level.value != base:
            raise SystemExit(f"held level not applied: {base} -> {level}")

    def measure(self, base: float, x: float, y: float) -> Dict[str, Dict[str, Any]]:
        self.hold(base)
        return self.adapter.measurements(x, y)

    def row(self, m: Dict[str, Any]) -> Dict[str, Any]:
        if m["status"] != "COMPUTED":
            return {"status": m["status"], "withheld_reason": m["reason"]}
        churn = float(m["constraints"][self.cid])
        goal = float(m["goal_value"])
        return {
            "status": "COMPUTED",
            "churn_pct": r6(churn),
            "limit_verdict": verdict(churn, self.limit, self.limit_op),
            "limit_margin_pp": r6(self.limit - churn),
            "goal_first_passage_max_mrr_gbp": r6(goal),
            "goal_verdict": verdict(goal, self.goal_limit, self.goal_op),
            "goal_margin_gbp": r6(goal - self.goal_limit),
        }

    def churn_of(self, option: str) -> Callable[[float, float], float]:
        def f(base: float, x: float) -> float:
            return float(self.measure(base, x, REFERENCE[1])[option]["constraints"][self.cid])

        return f

    def line_crossing(self, g: Callable[[float], float], a: float, b: float) -> Dict[str, Any]:
        """SCI-REGIONS' own method (lab_handoff.py): a two-point line through the evaluator, then the crossing is
        CHECKED on the evaluator itself (on the limit at t, breached just above). Nothing is fitted.
        """
        ga, gb = g(a), g(b)
        slope = (gb - ga) / (b - a)
        if abs(slope) <= TIE:
            return {"status": "NO_DEPENDENCE", "slope": r6(slope)}
        t = a + (self.limit - ga) / slope
        on = g(t)
        above = g(t + 1e-6)
        checked = abs(on - self.limit) <= 1e-9 and above > self.limit
        return {
            "status": "COMPUTED" if checked else "EVALUATOR_CHECK_FAILED",
            "at": r6(t),
            "slope": r6(slope),
            "evaluator_at_crossing": r6(on),
            "evaluator_just_above": r6(above),
        }

    def goal_crossing(self, option: str, xs: List[float], base: float) -> Dict[str, Any]:
        rows = [(x, self.row(self.measure(base, x, REFERENCE[1])[option])) for x in xs]
        flips = [
            (rows[i][0], rows[i + 1][0])
            for i in range(len(rows) - 1)
            if satisfied(rows[i][1]["goal_verdict"]) != satisfied(rows[i + 1][1]["goal_verdict"])
        ]
        if not flips:
            return {
                "status": "NO_CROSSING_ON_AXIS",
                "goal_met_everywhere": satisfied(rows[0][1]["goal_verdict"]),
                "min_margin_gbp": r6(min(r["goal_margin_gbp"] for _, r in rows)),
            }
        lo, hi = flips[0]

        def met(x: float) -> bool:
            return satisfied(self.row(self.measure(base, x, REFERENCE[1])[option])["goal_verdict"])

        m_lo = met(lo)
        for _ in range(60):  # bisection on the evaluator itself, inside the grid bracket
            mid = (lo + hi) / 2
            if met(mid) == m_lo:
                lo = mid
            else:
                hi = mid
        return {
            "status": "BRACKETED",
            "n_crossings_on_grid": len(flips),
            "bracket": [r6(flips[0][0]), r6(flips[0][1])],
            "bisected_at": r6((lo + hi) / 2),
            "met_below": m_lo,
        }


def run_r3b(r3b_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "range": {"low": RANGE_LOW, "high": RANGE_HIGH, "unit": RANGE_UNIT, "label": RANGE_LABEL},
        "reference_point": {
            "churn_response_pp_per_10gbp": REFERENCE[0],
            "competitive_response_gbp_per_month": REFERENCE[1],
        },
        "models": {},
    }
    for model_id in MODELS:
        r = R3B(r3b_root, model_id)
        if "quantities" not in out:
            g = r.adapter.graph
            out["quantities"] = [
                {
                    "id": nid,
                    "label": n.get("label"),
                    "kind": n.get("kind"),
                    "category": n.get("category"),
                    "value": (n.get("observed_state") or {}).get("raw_value"),
                    "unit": (n.get("observed_state") or {}).get("unit"),
                    "source": (n.get("observed_state") or {}).get("source"),
                    "has_own_std_or_range": any(
                        k in (n.get("observed_state") or {})
                        for k in ("std", "range_min", "range_max")
                    )
                    or "prior" in n,
                }
                for nid, n in sorted(g.nodes.items())
                if not g.is_structural(nid)
            ]
            out["constraints_as_stated"] = g.constraints
        ident = {
            k: v for k, v in r.adapter.identity.items() if k not in ("frozen_files", "corpus_files")
        }
        ident["frozen_files_all_match"] = all(r.adapter.identity["frozen_files"].values())
        ident["corpus_files_all_match"] = all(r.adapter.identity["corpus_files"].values())
        # Control: the unmodified adapter and the re-held graph at Olumi's own 3 % give identical measurements.
        pristine = copy.deepcopy(r.adapter.measurements(*REFERENCE))
        reheld = r.measure(3.0, *REFERENCE)
        options = list(r.adapter.graph.option_ids)
        labels = {o: r.adapter.graph.nodes[o]["label"] for o in options}
        sweep = []
        for b in BASE_GRID:
            m = r.measure(b, *REFERENCE)
            sweep.append({"base_churn_pct": r6(b), "options": {o: r.row(m[o]) for o in options}})
        computed = [o for o in options if sweep[0]["options"][o]["status"] == "COMPUTED"]
        per_option: Dict[str, Any] = {}
        for o in options:
            if o not in computed:
                per_option[o] = {
                    "label": labels[o],
                    "status": "WITHHELD",
                    "withheld_reason": sweep[0]["options"][o]["withheld_reason"],
                    "note": "no levels: nothing about this option is claimed, before or after the range",
                }
                continue
            rows = [s["options"][o] for s in sweep]
            point = next(s["options"][o] for s in sweep if s["base_churn_pct"] == 3.0)
            churn_f = r.churn_of(o)
            base_cross = r.line_crossing(lambda b, f=churn_f: f(b, REFERENCE[0]), 2.0, 3.0)
            band = {}
            for b in ANCHORS:
                band[str(b)] = r.line_crossing(lambda x, f=churn_f, b=b: f(b, x), 0.0, 1.0)
            goal_values = [row["goal_first_passage_max_mrr_gbp"] for row in rows]
            per_option[o] = {
                "label": labels[o],
                "at_olumi_point_3pct": point,
                "across_range_at_reference": {
                    "limit_satisfied_everywhere": all(
                        satisfied(row["limit_verdict"]) for row in rows
                    ),
                    "limit_verdicts_seen": sorted({row["limit_verdict"] for row in rows}),
                    "min_limit_margin_pp": r6(min(row["limit_margin_pp"] for row in rows)),
                    "goal_satisfied_everywhere": all(
                        satisfied(row["goal_verdict"]) for row in rows
                    ),
                    "goal_missed_everywhere": not any(
                        satisfied(row["goal_verdict"]) for row in rows
                    ),
                    "goal_verdicts_seen": sorted({row["goal_verdict"] for row in rows}),
                    "goal_first_passage_gbp_min": r6(min(goal_values)),
                    "goal_first_passage_gbp_max": r6(max(goal_values)),
                    "goal_invariant_to_base_churn": max(goal_values) - min(goal_values) == 0.0,
                },
                "base_churn_limit_crossing_at_reference": base_cross,
                "churn_response_limit_threshold_by_base_churn": band,
                "goal_crossing_on_churn_response_axis_by_base_churn": {
                    str(b): r.goal_crossing(o, X_GRID, b) for b in ANCHORS
                },
            }
        out["models"][model_id] = {
            "identity": ident,
            "structure_note": {
                "X_net_reading": "flows: net additions persist; churn_change_out = (option churn - status-quo churn) x "
                "stock (sim/dynamic.py rate_delta_x_stock). The base churn level cancels out of the "
                "MRR path; it enters only the churn limit.",
                "X_gross_reading": "flows: gross additions = held net + held churn x held subscribers; churn_out = "
                "option churn x current stock (derived_gross, rate_x_stock). Base churn enters both "
                "the MRR path and the churn limit.",
            }[model_id],
            "control_reheld_at_3pct_equals_frozen": reheld == pristine,
            "control_regions_threshold_reproduced": (
                model_id != "X_net_reading"
                or per_option[SUBJECT_OPTION]["churn_response_limit_threshold_by_base_churn"][
                    "3.0"
                ]["at"]
                == 1.5
            ),
            "limit": {"id": r.cid, "operator": r.limit_op, "value_pct": r.limit},
            "goal": {
                "operator": r.goal_op,
                "value_gbp": r.goal_limit,
                "rule": "FIRST_PASSAGE_BY_MONTH_12",
            },
            "computable_options": computed,
            "per_option": per_option,
            "sweep": sweep,
        }
        r.hold(3.0)
    return out


# =================================================================================================================
# Part 2: production-path control (served ISL src/, in-process), before vs after a user spread on churn
# =================================================================================================================

VOLATILE = {"processing_time_ms", "timestamp", "request_id", "request_echo", "build", "ms"}


def run_control(control_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    os.environ.setdefault("ISL_AUTH_DISABLED", "true")
    if str(ISL_ROOT) not in sys.path:
        sys.path.insert(0, str(ISL_ROOT))
    from fastapi.testclient import TestClient  # noqa: E402

    import src.middleware.request_limits as request_limits  # noqa: E402
    import src.services.analysis_pool as analysis_pool  # noqa: E402
    from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

    # Same deviation as the structural-robustness lane's runner: wall-clock budgets only (they gate whether a phase
    # completes, never its numbers).
    for attr in (
        "OVERALL_REQUEST_BUDGET_MS",
        "EVPI_BUDGET_MS",
        "PATH_DECOMPOSITION_BUDGET_MS",
        "E_VALUE_BUDGET_MS",
        "FLIP_STABILITY_BUDGET_MS",
        "FACTOR_FLIP_BUDGET_MS",
    ):
        setattr(RobustnessAnalyzerV2, attr, 3_600_000)
    for k in list(request_limits.ENDPOINT_TIMEOUTS):
        request_limits.ENDPOINT_TIMEOUTS[k] = 3600
    analysis_pool.ANALYSIS_HARD_DEADLINE_S = 3600.0
    from src.api.main import app  # noqa: E402

    pinned_path = control_root / "runs-direct" / "A-cur-s1254899477.isl-response.json"
    graph_path = control_root / "models" / "A-graph.json"
    pinned = json.loads(pinned_path.read_text())
    graph = json.loads(graph_path.read_text())
    req0 = pinned["request"]
    width_normalised = (RANGE_HIGH - RANGE_LOW) / 100.0  # churn is held as a fraction (3 % = 0.03)
    variants = {
        "before_as_sent": None,
        "before_same_default_labelled_template": {"std": None, "spread_source": "template"},
        "after_user_range_sd_width_over_4": {"std": width_normalised / 4, "spread_source": "user"},
        "after_user_range_sd_width_over_1_349": {
            "std": width_normalised / 1.349,
            "spread_source": "user",
        },
    }
    client = TestClient(app)
    responses: Dict[str, Dict[str, Any]] = {}
    requests: Dict[str, Dict[str, Any]] = {}
    for name, change in variants.items():
        req = copy.deepcopy(req0)
        if change is not None:
            pu = next(p for p in req["parameter_uncertainties"] if p["node_id"] == CHURN)
            if change["std"] is not None:
                pu["std"] = change["std"]
            pu["spread_source"] = change["spread_source"]
        req["request_id"] = f"sci-evidence-range-v1-{name}"
        r = client.post("/api/v1/robustness/analyze/v2?response_version=2", json=req)
        if r.status_code != 200:
            raise SystemExit(f"control {name}: HTTP {r.status_code}")
        responses[name] = r.json()
        requests[name] = req

    def science(resp: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in resp.items() if k not in VOLATILE}

    before = science(responses["before_as_sent"])
    reproduction = {k: science(pinned["response"]).get(k) == before.get(k) for k in sorted(before)}

    def summary(resp: Dict[str, Any]) -> Dict[str, Any]:
        cap = graph_goal_cap(graph)
        opts = {}
        for o in resp["options"]:
            ca = (o.get("constraint_analysis") or {}).get("constraints") or []
            opts[o["id"]] = {
                "probability_of_goal": o.get("probability_of_goal"),
                "churn_limit_prob_satisfied": ca[0]["prob_satisfied"] if ca else None,
                "mrr_p10_p50_p90_gbp": [r6(o["outcome"][q] * cap) for q in ("p10", "p50", "p90")],
            }
        churn_evppi = next(
            (f for f in resp.get("factor_evppi", []) if f["factor_id"] == CHURN), None
        )
        churn_sens = next(
            (f for f in resp.get("factor_sensitivity", []) if f["node_id"] == CHURN), None
        )
        return {
            "per_option": opts,
            "churn_factor_evppi": None
            if churn_evppi is None
            else {
                k: churn_evppi.get(k)
                for k in ("status", "status_reason", "spread_source", "evppi", "noise_floor")
            },
            "churn_factor_sensitivity": None
            if churn_sens is None
            else {
                k: churn_sens.get(k)
                for k in (
                    "elasticity",
                    "sensitivity_score",
                    "interpretation",
                    "rank_flip_rate",
                    "importance_rank",
                )
            },
            "factor_importance_order": [f["node_id"] for f in resp.get("factor_sensitivity", [])],
            "abs_elasticity_by_factor": {
                f["node_id"]: float(f"{abs(f['elasticity']):.15g}")
                for f in resp.get("factor_sensitivity", [])
            },
            "churn_flip": next(
                (f for f in resp.get("factor_flip_values", []) if f["factor_id"] == CHURN),
                {"status": "no row for monthly_churn"},
            ),
            "inference_warning_codes": sorted(
                w["code"] for w in resp.get("inference_warnings", [])
            ),
        }

    out: Dict[str, Any] = {
        "graph": "A-graph (Paul's current MRR brief, served m0 graph)",
        "engine": "this checkout's src/ (ISL f7f19e3, the served head), route /api/v1/robustness/analyze/v2",
        "pinned_request_sha256": sha256_json(req0),
        "churn_uncertainty_as_sent": next(
            p for p in req0["parameter_uncertainties"] if p["node_id"] == CHURN
        ),
        "distribution_assumption": (
            "ONLY here: the production engine samples Normal(3 %, sigma). sigma = width/4 "
            "(CEE's existing range rule) and width/1.349 (ISL range-fit IQR reading) are both "
            "run. Neither is the meaning of the user's range."
        ),
        "reproduction_of_pinned_served_response": {
            "all_science_fields_equal": all(reproduction.values()),
            "fields": reproduction,
        },
        "variants": {},
    }
    for name in variants:
        s = science(responses[name])
        changed = sorted(k for k in before if before.get(k) != s.get(k))
        out["variants"][name] = {
            "churn_uncertainty_sent": next(
                p for p in requests[name]["parameter_uncertainties"] if p["node_id"] == CHURN
            ),
            "summary": summary(responses[name]),
            "top_level_blocks_changed_vs_before": changed,
            "largest_numeric_change_by_block": {
                k: max_numeric_change(before.get(k), s.get(k)) for k in changed
            },
            "per_option_summary_identical_to_before": (
                summary(responses[name])["per_option"]
                == summary(responses["before_as_sent"])["per_option"]
            ),
            "science_sha256": sha256_json(s),
        }
    return out, {"graph": graph, "responses": {k: science(v) for k, v in responses.items()}}


def max_numeric_change(x: Any, y: Any) -> Dict[str, Any]:
    """Largest absolute and relative change across numeric leaves; non-numeric differences are counted, not hidden."""
    worst = {"abs": 0.0, "rel": 0.0, "non_numeric_differences": 0}

    def walk(a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                walk(a.get(k), b.get(k))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for p, q in zip(a, b):
                walk(p, q)
        elif (
            isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
        ):
            d = abs(float(a) - float(b))
            worst["abs"] = max(worst["abs"], d)
            if d:
                worst["rel"] = max(worst["rel"], d / max(abs(float(a)), abs(float(b))))
        elif a != b:
            worst["non_numeric_differences"] += 1

    walk(x, y)
    return {
        "abs": float(f"{worst['abs']:.3g}"),
        "rel": float(f"{worst['rel']:.3g}"),
        "non_numeric_differences": worst["non_numeric_differences"],
    }


def graph_goal_cap(graph: Dict[str, Any]) -> float:
    goal = next(n for n in graph["nodes"] if n["kind"] == "goal")
    return float(goal["observed_state"]["cap"])


# =================================================================================================================
# Part 3: the frozen SCI-EVIDENCE Lab adapter, unchanged
# =================================================================================================================


def run_cards(evidence_root: Path, control_raw: Dict[str, Any]) -> Dict[str, Any]:
    if str(evidence_root) not in sys.path:
        sys.path.insert(0, str(evidence_root))
    import lab  # noqa: F401,E402  (makes the frozen src/ importable)
    from lab.w1_card import _at, card_from_served  # noqa: E402

    fixture = evidence_root / "lab" / "fixtures" / "served-reload-c3f76e4f.json"
    served = _at(json.loads(fixture.read_text()), "read")
    real = card_from_served(served)
    out: Dict[str, Any] = {
        "adapter": "olumi-programme-docs lab/w1_card.py card_from_served (frozen, unchanged)",
        "real_served_before": {
            "source": "lab/fixtures/served-reload-c3f76e4f.json 'read' (the Lab join's PRIMARY input: a real served "
            "MRR reload, UI c3f76e4f, 28 Sep 2026)",
            "fixture_sha256": sha256_file(fixture),
            "lines": real["lines"],
            "rows": [
                {k: r[k] for k in ("factor_id", "status", "spread_source", "shown_as")}
                for r in real["internal"]["rows"]
            ],
        },
        "control_bodies": {},
        "body_note": (
            "DERIVED served-body shape around each control response: analysis_state complete_current, "
            "leader withheld (this experiment makes no ranking), computed_against_hash = graph_hash, and "
            "ISL factor_sensitivity node_id/label mapped to factor_id/factor_label as PLoT serves them."
        ),
    }
    graph = control_raw["graph"]
    ghash = sha256_json(graph)[:16]
    for name, resp in control_raw["responses"].items():
        sens = [
            {**row, "factor_id": row["node_id"], "factor_label": row["label"]}
            for row in resp.get("factor_sensitivity", [])
        ]
        body = {
            "analysis_state": {
                "run_state": {"kind": "complete_current", "computed_at": f"DERIVED:{name}"},
                "leader_claim": {"permitted": False, "withheld_reason": "experiment_no_ranking"},
            },
            "analysis_result": {
                "enrichment": {
                    "factor_evppi": resp.get("factor_evppi", []),
                    "factor_sensitivity": sens,
                    "inference_warnings": resp.get("inference_warnings", []),
                },
                "computed_against_hash": ghash,
            },
            "graph_hash": ghash,
            "graph": graph,
        }
        card = card_from_served(body)
        out["control_bodies"][name] = {
            "lines": card["lines"],
            "withheld": card["internal"]["withheld"],
            "rows": [
                {k: r[k] for k in ("factor_id", "status", "spread_source", "shown_as", "notes")}
                for r in card["internal"]["rows"]
            ],
        }
    return out


# =================================================================================================================


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--r3b-root", type=Path, required=True)
    ap.add_argument("--control-root", type=Path, required=True)
    ap.add_argument("--evidence-root", type=Path, required=True)
    args = ap.parse_args(argv)
    results = HERE / "results"
    results.mkdir(exist_ok=True)

    r3b = run_r3b(args.r3b_root.resolve())
    (results / "r3b.json").write_text(dump(r3b))
    control, control_raw = run_control(args.control_root.resolve())
    (results / "control.json").write_text(dump(control))
    cards = run_cards(args.evidence_root.resolve(), control_raw)
    (results / "cards.json").write_text(dump(cards))

    import build_payload  # noqa: E402  (sibling module: payload.json + coaching.md from the results only)

    payload, coaching, classification = build_payload.build(r3b, control, cards)
    (HERE / "payload.json").write_text(dump(payload))
    (HERE / "coaching.md").write_text(coaching)
    (HERE / "classification.json").write_text(dump(classification))

    inputs = {
        "r3b_graph": args.r3b_root / "r3b_sim" / "corpus" / "pj-20260927T180910Z-A.json",
        "r3b_mapping": args.r3b_root / "r3b_sim" / "mapping" / "pj-20260927T180910Z-A.json",
        "regions_adapter": args.r3b_root / "sci_regions_v1" / "r3b_adapter.py",
        "control_graph": args.control_root / "models" / "A-graph.json",
        "control_pinned_response": args.control_root
        / "runs-direct"
        / "A-cur-s1254899477.isl-response.json",
        "evidence_adapter": args.evidence_root / "lab" / "w1_card.py",
        "evidence_reference_card": args.evidence_root / "src" / "sci_evidence" / "card.py",
        "evidence_fixture": args.evidence_root / "lab" / "fixtures" / "served-reload-c3f76e4f.json",
    }
    code = {p.name: p for p in (HERE / "run.py", HERE / "build_payload.py")}
    code["robustness_analyzer_v2.py"] = ISL_ROOT / "src" / "services" / "robustness_analyzer_v2.py"
    manifest = {
        "experiment": "SCI-EVIDENCE range elicitation v1",
        "brief": "olumi-programme-docs#75 5909728656; corrected scope 5911560060",
        "heads": {
            "isl_src_tree": git_tree(ISL_ROOT, "src"),
            "isl_src_tree_at_served_f7f19e3": git_tree(
                ISL_ROOT, "src", "f7f19e3125ac54707aec38c1dca4bfbc80f03a8d"
            ),
            "r3b_and_regions_checkout": git_head(args.r3b_root),
            "r3b_pinned_evaluator_commit": r3b["models"]["X_net_reading"]["identity"]["commit"],
            "control_checkout": git_head(args.control_root),
            "evidence_checkout": git_head(args.evidence_root),
        },
        "inputs_sha256": {k: sha256_file(v) for k, v in inputs.items()},
        "code_sha256": {k: sha256_file(v) for k, v in code.items()},
        "outputs_sha256": {
            p: sha256_file(HERE / p)
            for p in (
                "results/r3b.json",
                "results/control.json",
                "results/cards.json",
                "payload.json",
                "coaching.md",
                "classification.json",
            )
        },
        "python": platform.python_version(),
        "provider_runs": "NOT_RUN",
        "network": "none used",
        "hypothetical_range": {
            "low": RANGE_LOW,
            "high": RANGE_HIGH,
            "unit": RANGE_UNIT,
            "label": RANGE_LABEL,
        },
        "verdict": payload["verdict"]["decision"],
    }
    (HERE / "manifest.json").write_text(dump(manifest))
    print(
        f"verdict {payload['verdict']['decision']}; wrote results/, payload.json, coaching.md, manifest.json"
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
