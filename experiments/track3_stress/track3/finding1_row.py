"""Finding 1 real-path row: a STATED Normal law must reach ISL as stated (PLoT's epistemic floor widened it).

  PLOT_DIR=<plot checkout> python -m track3.finding1_row --out <fresh dir>
RED on PLoT acf24764 (the floor lifts the stated sd); GREEN on the patch (wip/track3-floor-declared-sd).
"""
import argparse, json, os, sys
from statistics import NormalDist

from .evaluator_adapter import Engine
from .m2 import _tol

S = 200.0


def core_normal_link():
    """The core mapper's shape for a person's Normal link (mapper.ts:301-302): beta ~ N(0.1, 0.02) on the goal's frame."""
    goal_obs = {"value": 80 / S, "baseline": 80 / S, "raw_value": 80, "cap": S, "unit": "£000"}
    body = {"request_id": "track3-finding1:core-normal-link", "graph": {"nodes": [
                {"id": "y", "kind": "goal", "label": "Outcome (£000)", "observed_state": goal_obs, "goal_threshold": 98 / S,
                 "goal_threshold_raw": 98, "goal_threshold_unit": "£000", "goal_threshold_cap": S, "goal_threshold_frame": "level"},
                {"id": "x", "kind": "factor", "label": "Programme in place", "category": "controllable",
                 "observed_state": {"value": 0, "raw_value": 0, "cap": 1, "unit": ""}}],
            "edges": [{"id": "x_y", "from": "x", "to": "y", "effect_direction": "positive", "exists_probability": 1,
                       "strength": {"mean": 0.1, "std": 0.02},
                       "provenance": {"source": "brief_extraction", "magnitude": "user_stated",
                                      "natural_effect": {"amount": 20, "amount_unit": "£000", "per_source_change": 1,
                                                         "per_source_change_unit": "", "strength_mean": 0.1, "strength_mean_frame": "edge_strength"}}}]},
            "options": [{"id": "carry_on", "label": "Carry on", "interventions": {"x": 0.0}},
                        {"id": "programme", "label": "Run the programme", "interventions": {"x": 1.0}}],
            "goal_node_id": "y", "goal_threshold": 98 / S, "goal_direction": "maximise", "seed": 20261021, "n_samples": 10000}
    exact = 1 - NormalDist(0.1, 0.02).cdf(0.09)          # level 0.4 + beta >= 0.49
    return body, "programme", exact, 0.02 * S


def g4_shape():
    """The original G4 Channel A row, with the exact link marked as the person's (std at ISL's minimum)."""
    body = {"request_id": "track3-finding1:g4-channel-a", "graph": {"nodes": [
                {"id": "y", "kind": "goal", "label": "Outcome Y", "intercept": 0, "goal_threshold": 100 / S,
                 "goal_threshold_frame": "delta", "goal_threshold_cap": S},
                {"id": "y_level", "kind": "factor", "label": "Outcome driver", "category": "observable",
                 "observed_state": {"value": 90 / S, "std": 10 / S}},
                {"id": "lever", "kind": "factor", "label": "Lever", "observed_state": {"value": 0, "std": 1e-4}}],
            "edges": [{"from": "y_level", "to": "y", "exists_probability": 1, "strength": {"mean": 1, "std": 0.0011},
                       "provenance": {"source": "user_specified", "magnitude": "user_stated"}},
                      {"from": "lever", "to": "y", "exists_probability": 1, "strength": {"mean": 1, "std": 0.0011},
                       "provenance": {"source": "user_specified", "magnitude": "user_stated"}}]},
            "options": [{"id": "mu90", "label": "mu 90", "interventions": {"lever": 0.0}},
                        {"id": "mu100", "label": "mu 100", "interventions": {"lever": 10 / S}}],
            "goal_node_id": "y", "goal_threshold": 100 / S, "goal_direction": "maximise", "seed": 20261021, "n_samples": 10000}
    return body, "mu90", 1 - NormalDist(90, 10).cdf(100), 10.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "capture"), exist_ok=True)
    plot_dir = os.environ.get("PLOT_DIR", "/private/tmp/track3-engine/plot")
    rows = []
    with Engine(os.path.join(a.out, "capture")) as engine:
        for name, (body, option, exact, declared_sd) in {"core_normal_link": core_normal_link(), "g4_channel_a": g4_shape()}.items():
            res = engine.evaluate(body, name)
            o = next((x for x in res["plot"].get("option_comparison") or [] if x.get("option_id") == option), {})
            p, n = o.get("probability_of_goal"), (o.get("probability_of_goal_precision") or {}).get("n_informative") or 0
            sent = [e["strength"] for x in res["isl_exchanges"] for e in x["request"]["graph"]["edges"]]
            sd = (o.get("outcome") or {}).get("std")
            ok = p is not None and n > 0 and abs(p - exact) <= _tol(exact, n)
            rows.append({"row": name, "exact_p_goal": round(exact, 6), "engine_p_goal": p, "n": n,
                         "tolerance": round(_tol(exact, n), 5) if n else None, "declared_sd": declared_sd,
                         "engine_outcome_sd": None if sd is None else round(sd * S, 3), "strengths_sent_to_isl": sent,
                         "verdict": "GREEN" if ok else "RED"})
        identity = engine.identity
    out = {"plot_dir": plot_dir, "plot_head": os.popen(f"git -C {plot_dir} rev-parse HEAD").read().strip(),
           "plot_dirty": os.popen(f"git -C {plot_dir} status --porcelain --untracked-files=no").read().strip() != "",
           "isl_src_identical_to_pin": identity.get("isl_src_identical_to_pin"), "rows": rows}
    json.dump(out, open(os.path.join(a.out, "finding1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(0 if all(r["verdict"] == "GREEN" for r in rows) else 1)


if __name__ == "__main__":
    main()
