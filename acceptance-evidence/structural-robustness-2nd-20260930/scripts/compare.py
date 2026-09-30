#!/usr/bin/env python3
"""SCI-STRUCTURAL-ROBUSTNESS (2nd run): build `results/comparison.json` from `runs-direct/` (like-for-like fields only).

Usage (evidence dir): python scripts/compare.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOAL_FRAME = "106250"  # MRR's execution frame (cap): £85,000 = 0.8
SEEDS = ["1254899477", "1", "20260930"]
MODELS = {"A": "today's subscribers (m2 + card Yes)", "A2": "frame control of A (subscriber cap 20,000)",
          "B": "subscribers at month 12 (Olumi's monthly effects accumulated over the brief's 12 months)",
          "C": "A with Olumi's price -> churn sign reversed (SCIENCE 5912253182 / R3 5912313721)"}
OPTSETS = {"cur": "today's CEE submission (Raise to £59, Status quo)", "served3": "+ Olumi's 'Raise to £54'"}


def headline(path: str) -> dict:
    out = subprocess.check_output([sys.executable, f"{HERE}/extract.py", "headline", path, GOAL_FRAME])
    return json.loads(out)


def main() -> None:
    runs = {}
    for optset in OPTSETS:
        for model in MODELS:
            for seed in SEEDS:
                p = f"{ROOT}/runs-direct/{model}-{optset}-s{seed}.isl-response.json"
                runs[f"{model}|{optset}|{seed}"] = headline(p)

    def span(model: str, optset: str, f) -> list:
        vals = [f(runs[f"{model}|{optset}|{s}"]) for s in SEEDS]
        return [min(vals), max(vals)] if all(isinstance(v, (int, float)) for v in vals) else vals

    o59 = "raise_pro_price_to_59"
    table = {}
    for optset in OPTSETS:
        rows = {}
        for model in MODELS:
            rows[model] = {
                "leader": sorted({runs[f"{model}|{optset}|{s}"]["leader"] for s in SEEDS}),
                "win_probability_59": span(model, optset, lambda h: h["options"][o59]["win_probability"]),
                "p_mrr_above_85k_59": span(model, optset, lambda h: h["options"][o59]["probability_of_goal"]),
                "mrr_mean_59_gbp": span(model, optset, lambda h: h["options"][o59]["mrr_mean_gbp"]),
                "mrr_p10_59_gbp": span(model, optset, lambda h: h["options"][o59]["mrr_p10_gbp"]),
                "p_below_today_59": span(model, optset, lambda h: h["options"]["status_quo"]["win_probability"]),
                "p_churn_below_5pct_59": span(model, optset, lambda h: h["options"][o59]["churn_below_limit"][0]),
                "robustness_level": sorted({runs[f"{model}|{optset}|{s}"]["robustness_level"] for s in SEEDS}),
                "fragile_edges": sorted({e for s in SEEDS for e, _ in runs[f"{model}|{optset}|{s}"]["fragile_edges"]}),
                "fragile_switch_probability": {
                    e: span(model, optset, lambda h, e=e: dict(h["fragile_edges"]).get(e, 0.0))
                    for e in sorted({e for s in SEEDS for e, _ in runs[f"{model}|{optset}|{s}"]["fragile_edges"]})
                },
                "identity_evaluated": sorted({str(i["evaluated"]) for s in SEEDS
                                              for i in runs[f"{model}|{optset}|{s}"]["identity_evaluations"]}),
            }
            if optset == "served3":
                rows[model]["win_probability_54"] = span(
                    model, optset, lambda h: h["options"]["raise_pro_price_to_54"]["win_probability"])
                rows[model]["p_mrr_above_85k_54"] = span(
                    model, optset, lambda h: h["options"]["raise_pro_price_to_54"]["probability_of_goal"])
        table[optset] = rows

    json.dump({
        "experiment": "SCI-STRUCTURAL-ROBUSTNESS, independent 2nd run (30 Sep 2026)",
        "seeds": SEEDS, "n_samples": 10000, "models": MODELS, "option_sets": OPTSETS,
        "ranges": "[min, max] over the three seeds",
        "table": table,
        "runs": runs,
    }, open(f"{ROOT}/results/comparison.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
