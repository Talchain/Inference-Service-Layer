"""Generate the per-graph evaluation tables inside EVALUATION.md from results.json.

Only the block between the GENERATED markers is rewritten; the narrative is hand-written.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from .corpus import ROOT, load_graph
from .frames import held_level
from .spec import load_valid_spec

BEGIN = "<!-- BEGIN GENERATED: per-graph tables (sim/report.py) -->"
END = "<!-- END GENERATED -->"

CURRENT = {
    "baseline": "NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current "
    "£14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577)",
    "identities": 'No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); '
    "propagation is additive in normalised space (CODE, ISL `3717e36`)",
    "deadline": "No horizon concept in the engine (CODE)",
    "time": "No (CODE)",
    "constraints": "NOT IN CORPUS; limits are scored per draw on normalised levels (CODE)",
    "withheld": "NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE)",
    "separation": "NOT IN CORPUS",
    "evppi": "NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution "
    "finding, AIC 5858555342)",
    "gaps": "Not reported by the engine",
    "sensitivity": "NOT IN CORPUS",
    "assumptions": "N/A",
    "runtime": "NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5)",
}


def _cells(blocks: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    total = computed = 0
    verdicts: list[str] = []
    for b in blocks:
        for opt, row in b["options"].items():
            total += 1
            if row["status"] == "computed":
                computed += 1
                held = (
                    "holds"
                    if row["p_held"] == 1.0
                    else ("breached" if row["p_held"] == 0.0 else f"P={row['p_held']}")
                )
                verdicts.append(f"{b['node']} {opt}: {row['value_mean']:g} ({held}, {row['tier']})")
    return computed, total, verdicts


def _gap_top(blocks: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for b in blocks:
        for row in b["options"].values():
            for g in {x.split(":", 1)[0] for x in row.get("gaps", [])}:
                if g in ("TAINTED_BY", "OPERAND_WITHHELD"):
                    continue
                counts[g] = counts.get(g, 0) + 1
    return [f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]


def _identities(blocks: list[dict[str, Any]]) -> str:
    parts = []
    for b in blocks:
        rows = b["options"]
        comp = [
            f"{o}: £{r['value_mean']:,.0f}"
            + (
                " (at unchanged " + ", ".join(r["conditional_on_unchanged"]) + ")"
                if r["status"] == "conditional"
                else ""
            )
            for o, r in rows.items()
            if r["status"] in ("computed", "conditional")
        ]
        if comp:
            parts.append(f"`{b['target']}` = " + "; ".join(comp))
        else:
            gaps = sorted(
                {g.split(":", 1)[0] for r in rows.values() for g in r.get("gaps", [])}
                - {"TAINTED_BY", "OPERAND_WITHHELD"}
            )
            parts.append(f"`{b['target']}` withheld ({', '.join(gaps)})")
    return "<br>".join(parts) if parts else "none declared"


def _models_line(models: list[dict[str, Any]], semantics: str) -> str:
    lines = []
    for m in models:
        if m["template_draws"]:
            continue
        comp = []
        for o, r in m["options"].items():
            if r["status"] != "computed":
                continue
            p = (
                r["p_by_H"]
                if (semantics == "attain_by_H" and r.get("p_by_H") is not None)
                else r["p_at_H"]
            )
            comp.append(
                f"{o} £{r['outcome_at_H_mean']:,.0f} (goal {'met' if p == 1.0 else 'not met'})"
            )
        sc = ", ".join(f"{k}={v:g}" for k, v in sorted(m["scenario"].items())) or "no X effects"
        lines.append(
            f"{m['model']} [{sc}]: " + ("; ".join(comp) if comp else "no option computable")
        )
    return "<br>".join(lines)


def sensitivity_rank(
    models: list[dict[str, Any]], top: int = 3
) -> dict[str, list[tuple[str, float, int]]]:
    """Per template-draw model: parameters ranked by the median, across scenarios, of the
    largest |Spearman rho| (parameter vs outcome at H) over the model's decision options."""
    per: dict[str, dict[str, list[float]]] = {}
    for m in models:
        if not m["template_draws"] or m["evppi"].get("status") != "computed":
            continue
        for pid, v in m["evppi"]["parameters"].items():
            rho = [abs(x) for x in v["spearman_vs_outcome_at_H"].values() if x is not None]
            if rho:
                per.setdefault(m["model"], {}).setdefault(pid, []).append(max(rho))
    out = {}
    for model, params in sorted(per.items()):
        ranked = sorted(
            ((pid, statistics.median(v), len(v)) for pid, v in params.items()),
            key=lambda t: (-t[1], t[0]),
        )
        out[model] = [(pid, round(med, 2), n) for pid, med, n in ranked[:top]]
    return out


def _sensitivity(models: list[dict[str, Any]]) -> str:
    ranks = sensitivity_rank(models)
    parts = [
        f"{model}: "
        + ", ".join(f"`{pid}` {med:.2f}" for pid, med, _ in rows)
        + f" (median over {rows[0][2]} scenario{'s' if rows[0][2] > 1 else ''})"
        for model, rows in ranks.items()
        if rows
    ]
    skipped = sorted(
        {
            f"{m['model']}: {m['evppi'].get('status')} ({m['evppi'].get('reason', '')})"
            for m in models
            if m["template_draws"] and m["evppi"].get("status") != "computed"
        }
    )
    return "<br>".join(parts + skipped)


def graph_table(gid: str, g: dict[str, Any], runtime: dict[str, float]) -> list[str]:
    graph = load_graph(gid)
    spec = load_valid_spec(graph)
    t0, t1, t2, tx = (g["tiers"][k] for k in ("T0", "T1", "T2", "X"))
    held0 = held1 = 0
    for node in graph.quantity_ids:
        lvl = held_level(graph, node)
        if lvl is not None and lvl.value is not None:
            held1 += 1
            held0 += int(lvl.tier == 0)
    c0, n0, _ = _cells(t0["constraints"])
    c1, n1, v1 = _cells(t1["constraints"])
    id_total = sum(len(b["options"]) for b in t1["identities"])
    id_withheld = sum(
        1 for b in t1["identities"] for r in b["options"].values() if r["status"] == "withheld"
    )
    s = g["summary"]
    semantics = g["goal"]["temporal_semantics"]
    t2_models = t2.get("goal_models", [])
    t2_line = (
        _models_line(t2_models, semantics)
        if t2_models
        else "withheld: " + ", ".join(t2.get("goal", {}).get("gaps", []))
    )
    x_line = (
        _models_line(tx["models"], semantics)
        or "no Mode X model (nothing to explore without inventing quantities)"
    )
    defaults = sorted(
        {d for m in t2_models for r in m["options"].values() for d in r.get("defaults", [])}
    )
    x_defaults = sorted(
        {d for m in tx["models"] for r in m["options"].values() for d in r.get("defaults", [])}
    )
    assumptions = sorted(
        {a for m in tx["models"] for r in m["options"].values() for a in r.get("assumptions", [])}
    )
    rt_strict = (runtime.get(f"{gid}:T0", 0) + runtime.get(f"{gid}:T1", 0)) * 1000
    rt_t2 = runtime.get(f"{gid}:T2", 0) * 1000
    rt_x = runtime.get(f"{gid}:X", 0)
    pair = g["brief_decision_pair"]
    hz = g["horizon"]
    rows = [
        (
            "Baseline reproduced correctly",
            CURRENT["baseline"],
            f"Yes: status quo equals every admissible held level ({held0} user/brief-stated at T0; {held1} incl. Olumi estimates at T1); tested",
            "Held levels read through the R3-8 frame reader",
        ),
        (
            "Exact identities represented",
            CURRENT["identities"],
            _identities(t1["identities"]),
            "Declared identities only; label-derived sums are Mode X",
        ),
        (
            "Deadline represented",
            CURRENT["deadline"],
            f"Withheld at T0/T1 (no admissible time path). Horizon: {hz['status']}{(' ' + str(hz['months']) + ' months') if hz['months'] else ''}",
            f"T2: {t2_line}",
        ),
        (
            "Time accumulation represented",
            CURRENT["time"],
            "No (needs persistence and onset defaults)",
            f"T2 time accumulation: {'yes' if s['T2']['time_accumulation'] else 'no'}; X models: {', '.join(s['X']['models_run']) or 'none'}",
        ),
        (
            "Supported constraints evaluated",
            CURRENT["constraints"],
            f"T0: {c0}/{n0} option-limit cells; T1: {c1}/{n1}. "
            + ("; ".join(v1[:6]) if v1 else ""),
            "Static (timeless) verdicts: the same the H=0 control gives",
        ),
        (
            "Unsupported claims withheld",
            CURRENT["withheld"],
            f"T1 withheld: {n1 - c1}/{n1} limit cells, {id_withheld}/{id_total} identity cells; top causes: {', '.join(_gap_top(t1['constraints'] + t1['identities'])) or '—'}",
            "Taint: an unquantified link whose source moves withholds its target",
        ),
        (
            "Decision-relevant option separation",
            CURRENT["separation"],
            "Not computable at T0/T1 (goal withheld)",
            f"Whole ranking: T2 {'yes' if s['T2']['ranking_available'] else 'withheld'}, X {'yes' if s['X']['ranking_available'] else 'withheld'}; brief pair {pair}: X {'computed' if s['X']['brief_pair_available'] else 'withheld'}"
            + (
                f"; X goal verdict flips across assumptions for {', '.join(s['X']['goal_verdict_flips'])}"
                if s["X"]["goal_verdict_flips"]
                else ""
            ),
        ),
        (
            "EVPPI / information-value result",
            CURRENT["evppi"],
            "NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1)",
            f"T2 resolved: ISL-status {s['T2']['evppi_resolved_isl']}, strict {s['T2']['evppi_resolved_strict']}; X resolved: ISL-status {s['X']['evppi_resolved_isl']}, strict {s['X']['evppi_resolved_strict']}",
        ),
        (
            "Sensitivity rank (\\|Spearman ρ\\| vs outcome at H)",
            CURRENT["sensitivity"],
            "NOT_COMPUTABLE (no admissible uncertainty at T0/T1)",
            "T2: "
            + (_sensitivity(t2_models) or "no template-draw model")
            + "; X: "
            + (_sensitivity(tx["models"]) or "no template-draw model")
            + (
                ". `amount:` = effect size, `exists:` = link existence. Spreads are templates"
                " (31/31), so the ranks reflect template uncertainty, not elicited uncertainty"
                if sensitivity_rank(t2_models) or sensitivity_rank(tx["models"])
                else ""
            ),
        ),
        (
            "Required semantic gaps",
            CURRENT["gaps"],
            ", ".join(s["gap_classes"]),
            "From the frozen mapping plus engine withholding",
        ),
        (
            "Prototype-only assumptions",
            CURRENT["assumptions"],
            "None (strict)",
            f"T2 goal-model defaults: {', '.join(defaults) or 'none (no T2 goal model)'}; X defaults: {', '.join(x_defaults) or 'none'}; X assumptions: {', '.join(assumptions) or 'none'}",
        ),
        (
            "Runtime",
            CURRENT["runtime"],
            f"{rt_strict:.1f} ms (T0 + T1, all options)",
            f"T2 {rt_t2:.1f} ms; X {rt_x:.2f} s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns)",
        ),
    ]
    out = [f"### {gid} (journey {graph.journey})", "", f"> {graph.brief}", ""]
    out.append(
        "| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |"
    )
    out.append("|---|---|---|---|")
    for name, cur, strict, notes in rows:
        out.append(f"| {name} | {cur} | {strict} | {notes} |")
    if x_line and tx["models"]:
        out.append("")
        out.append(f"Mode X deadline outcomes (point, links exist): {x_line}")
    out.append("")
    _ = spec
    return out


def render_block() -> str:
    results = json.loads((ROOT / "results.json").read_text())
    runtime = json.loads((ROOT / "runtime.json").read_text())
    lines = [BEGIN, ""]
    for gid, g in results["graphs"].items():
        lines.extend(graph_table(gid, g, runtime))
    lines.append(END)
    return "\n".join(lines)


def main() -> None:
    path = ROOT / "EVALUATION.md"
    text = path.read_text()
    start, end = text.index(BEGIN), text.index(END) + len(END)
    path.write_text(text[:start] + render_block() + text[end:])


if __name__ == "__main__":
    main()
