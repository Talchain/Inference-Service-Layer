"""Shared helpers for the R3-B tests."""

from __future__ import annotations

from typing import Any

from sim.corpus import Graph, load_graph
from sim.dynamic import Trajectory
from sim.run import simulate
from sim.spec import load_valid_spec
from sim.static import Draws, OptionEval, evaluate_all, point_draws

A1 = "pj-20260927T180910Z-A"
A2 = "pj-20260927T181846Z-A"
A3 = "pj-20260927T182848Z-A"
A4 = "pj-20260927T183807Z-A"
C2 = "pj-20260927T181846Z-C"
C3 = "pj-20260927T182848Z-C"
C4 = "pj-20260927T183807Z-C"
E3 = "pj-20260927T182848Z-E"


def load(gid: str) -> tuple[Graph, dict[str, Any]]:
    g = load_graph(gid)
    return g, load_valid_spec(g)


def static(gid: str, tier: int, **kw: Any) -> dict[str, OptionEval]:
    g, spec = load(gid)
    return evaluate_all(g, spec, tier, point_draws(g), **kw)


def model(spec: dict[str, Any], model_id: str) -> dict[str, Any]:
    return next(m for m in spec["dynamic_models"] if m["id"] == model_id)


def run(
    gid: str, model_id: str, scenario: dict[str, float], draws: Draws | None = None
) -> dict[str, Trajectory]:
    g, spec = load(gid)
    return simulate(g, spec, model(spec, model_id), scenario, draws or point_draws(g))
