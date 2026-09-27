"""Render MAPPING.md from the frozen specs and the corpus (Task 1 deliverable)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .corpus import ROOT, Graph, graph_ids, load_graph
from .edges import classify_edge
from .frames import held_level
from .spec import load_valid_spec


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}"


def _evidence(evs: list[dict[str, Any]]) -> str:
    out = []
    for ev in evs:
        kind = ev["kind"]
        if kind == "brief":
            out.append(f'brief: "{ev["quote"]}"')
        elif kind in ("typed_field", "declared_identity"):
            out.append(f"{kind}: `{ev['ptr']}`")
        elif kind == "accounting_identity":
            out.append(f"accounting_identity: {ev['basis']}")
        else:
            out.append(kind)
    return "<br>".join(out) if out else "none"


def _graph_section(graph: Graph, spec: dict[str, Any]) -> list[str]:
    lines = [f"## {graph.id} (journey {graph.journey})", ""]
    lines.append(f"> {graph.brief}")
    lines.append("")
    hz = spec["horizon"]
    months = f"{hz['months']} months" if hz["months"] else "—"
    lines.append(
        f"- **Horizon:** {hz['status']} ({months}); evidence: brief \"{hz['evidence']['quote']}\"."
    )
    goal = spec["goal"]
    thr = graph.resolve(goal["threshold_ptr"]) if goal["threshold_ptr"] else None
    lines.append(
        f"- **Goal:** `{goal['node']}` {goal['operator'] or ''} {_fmt(thr)}; temporal semantics "
        f"`{goal['temporal_semantics']}`; gaps: {', '.join(goal['gaps']) or 'none'}."
    )
    for tid, d in spec["declared_identities"].items():
        ident = graph.identity(tid)
        assert ident is not None
        lines.append(
            f"- **Declared identity:** `{tid}` = {' x '.join(ident['factor_ids'])} → "
            f"**{d['status']}** {('(' + ', '.join(d['gaps']) + ')') if d['gaps'] else ''}. {d['note']}"
        )
    for d in spec["derived_identities"]:
        lines.append(
            f"- **Derived identity (Mode X only):** `{d['target']}` = {' + '.join(d['operands'])}. "
            f"{d['basis']}"
        )
    t2 = spec["goal_model_t2"]
    lines.append(
        f"- **T2 goal model:** {t2['status']}"
        f"{(' (' + t2['model'] + ')') if t2['model'] else ''}; gaps: {', '.join(t2['gaps']) or 'none'}. "
        f"{t2['note']}"
    )
    lines.append("")
    lines.append(
        "| Quantity | Kind | Class | Held level (user units, source) | Evidence | GAPs | Note |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for qid, q in spec["quantities"].items():
        lvl = held_level(graph, qid)
        held = "—" if lvl is None else f"{_fmt(lvl.value)} ({lvl.source})"
        lines.append(
            f"| `{qid}` | {graph.kind(qid)} | {q['class']} | {held} | {_evidence(q['evidence'])} | "
            f"{', '.join(q['gaps']) or '—'} | {q['note']} |"
        )
    lines.append("")
    lines.append(
        "| Edge | Class | Evidence | Natural effect (user units) | Tier | sd/mean | GAPs |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for e in graph.behavioural_edges:
        ec = classify_edge(graph, spec, e)
        ne = e.natural_effect
        ne_s = (
            f"{ne['amount']:g} {ne['amount_unit']} per {ne['per_source_change']:g} "
            f"{ne['per_source_change_unit']} ({ec.magnitude or 'no magnitude tag'})"
            if ne
            else "—"
        )
        tier = "—" if ec.tier is None else f"T{ec.tier}"
        ratio = "—" if ec.sd_over_mean is None else f"{ec.sd_over_mean:.2f}"
        lines.append(
            f"| `{e.key}` | {ec.cls} | {ec.evidence} | {ne_s} | {tier} | {ratio} | "
            f"{', '.join(ec.gaps) or '—'} |"
        )
    if spec["x_effects"]:
        lines.append("")
        lines.append("Mode X prototype assumptions (sensitivity-tested, never used as evidence):")
        lines.append("")
        for xe in spec["x_effects"]:
            vals = "; ".join(xe["basis"])
            lines.append(
                f"- `{xe['id']}` on {' → '.join(xe['path'])}: sweep {xe['sweep']} ({vals})."
            )
    for m in spec["dynamic_models"]:
        lines.append(
            f"- Model `{m['id']}` ({m['tier']}, {m['type']}): defaults {m['defaults']}; "
            f"assumptions {m['assumptions']}. {m['note']}"
        )
    if spec["notes"]:
        lines.append("")
        for n in spec["notes"]:
            lines.append(f"- Note: {n}")
    lines.append("")
    return lines


def render() -> str:
    graphs = [load_graph(g) for g in graph_ids()]
    specs = {g.id: load_valid_spec(g) for g in graphs}
    classes: Counter[str] = Counter()
    ev_kinds: Counter[str] = Counter()
    gap_graphs: Counter[str] = Counter()
    level_sources: Counter[str] = Counter()
    edge_classes: Counter[str] = Counter()
    ne_template = 0
    ne_total = 0
    for g in graphs:
        spec = specs[g.id]
        for qid, q in spec["quantities"].items():
            classes[q["class"]] += 1
            for ev in q["evidence"]:
                ev_kinds[ev["kind"]] += 1
            lvl = held_level(g, qid)
            if lvl is not None:
                level_sources[str(lvl.source)] += 1
        for code in (
            set(spec["graph_gaps"]) | set(spec["goal"]["gaps"]) | set(spec["horizon"]["gaps"])
        ):
            gap_graphs[code] += 1
        for e in g.behavioural_edges:
            ec = classify_edge(g, spec, e)
            edge_classes[ec.cls] += 1
            if e.natural_effect is not None:
                ne_total += 1
                if ec.sd_over_mean is not None and round(ec.sd_over_mean, 6) in (0.25, 0.5):
                    ne_template += 1
    out = [
        "# R3-B MAPPING: temporal semantics audit of the 12-graph corpus",
        "",
        "Generated by `sim/mapping_report.py` from the frozen `mapping/*.json` specs and the "
        "verbatim corpus copy. Every classification records its evidence. **A goal node is not "
        "treated as a stock by default, and an edge is not treated as a monthly rate by default.** "
        "Where temporal semantics are absent, the entry is a GAP.",
        "",
        "Evidence tiers are defined in `ANALYSIS-PLAN.md`. Held levels are read through the R3-8 "
        "frame reader (`raw_value`, else value x cap, else value x scale_frame); a missing frame "
        "means the level is withheld.",
        "",
        "## Corpus-wide summary",
        "",
        "| Quantity class | Count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(classes.items())],
        "",
        "| Behavioural-edge class (mechanical) | Count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(edge_classes.items())],
        "",
        f"Natural effects whose sd/mean is exactly 0.25 or 0.5 (template spread): "
        f"{ne_template} of {ne_total}.",
        "",
        "| Held-level source | Count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(level_sources.items())],
        "",
        "| Quantity evidence kind | Count |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(ev_kinds.items())],
        "",
        "| GAP class | Graphs affected (of 12) |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(gap_graphs.items(), key=lambda kv: (-kv[1], kv[0]))],
        "",
    ]
    for g in graphs:
        out.extend(_graph_section(g, specs[g.id]))
    return "\n".join(out)


def main() -> None:
    (ROOT / "MAPPING.md").write_text(render())


if __name__ == "__main__":
    main()
