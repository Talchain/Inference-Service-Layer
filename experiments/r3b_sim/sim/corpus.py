"""Read-only access to the verbatim R3-B corpus copy (``experiments/r3b_sim/corpus``)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
MAPPING_DIR = ROOT / "mapping"

STRUCTURAL_KINDS = frozenset({"decision", "option"})
USER_SOURCES = frozenset(
    {"brief_extraction", "user_override", "user_assumption", "user_specified", "user_set"}
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((CORPUS_DIR / "MANIFEST.json").read_text())
    return manifest


def verify_corpus() -> dict[str, bool]:
    """Every corpus file's sha256 against the manifest written at copy time."""
    files: dict[str, str] = load_manifest()["files"]
    return {name: sha256_file(CORPUS_DIR / name) == digest for name, digest in files.items()}


def graph_ids() -> list[str]:
    index: list[dict[str, Any]] = json.loads((CORPUS_DIR / "INDEX.json").read_text())
    return sorted(entry["file"][: -len(".json")] for entry in index)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    raw: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.src}->{self.dst}"

    @property
    def natural_effect(self) -> dict[str, Any] | None:
        ne = (self.raw.get("provenance") or {}).get("natural_effect")
        return ne if isinstance(ne, dict) else None

    @property
    def source(self) -> str | None:
        src = (self.raw.get("provenance") or {}).get("source")
        return src if isinstance(src, str) else None

    @property
    def magnitude(self) -> str | None:
        mag = (self.raw.get("provenance") or {}).get("magnitude")
        return mag if isinstance(mag, str) else None

    @property
    def strength_mean(self) -> float:
        return float(self.raw["strength"]["mean"])

    @property
    def strength_std(self) -> float:
        return float(self.raw["strength"]["std"])

    @property
    def exists_probability(self) -> float:
        return float(self.raw.get("exists_probability", 1.0))


class Graph:
    """One corpus draft_graph plus its brief. Values are only ever read, never re-typed."""

    def __init__(self, graph_id: str, doc: dict[str, Any]) -> None:
        self.id = graph_id
        self.doc = doc
        self.journey: str = doc["journey"]
        self.brief: str = doc["brief_first_message"]
        dg = doc["draft_graph"]
        self.nodes: dict[str, dict[str, Any]] = {n["id"]: n for n in dg["nodes"]}
        self.edges: list[Edge] = [Edge(e["from"], e["to"], e) for e in dg["edges"]]
        self.constraints: list[dict[str, Any]] = list(dg.get("goal_constraints") or [])

    def kind(self, node_id: str) -> str:
        return str(self.nodes[node_id]["kind"])

    def is_structural(self, node_id: str) -> bool:
        return self.kind(node_id) in STRUCTURAL_KINDS

    @property
    def quantity_ids(self) -> list[str]:
        return [n for n in self.nodes if not self.is_structural(n)]

    @property
    def option_ids(self) -> list[str]:
        return [n for n in self.nodes if self.kind(n) == "option"]

    @property
    def baseline_option(self) -> str:
        base = [o for o in self.option_ids if self.nodes[o].get("is_baseline")]
        if len(base) != 1:
            raise ValueError(f"{self.id}: expected exactly one baseline option, got {base}")
        return base[0]

    @property
    def behavioural_edges(self) -> list[Edge]:
        """Every edge whose source is a quantity (not a decision or option)."""
        return [e for e in self.edges if not self.is_structural(e.src)]

    def edge(self, key: str) -> Edge:
        for e in self.edges:
            if e.key == key:
                return e
        raise KeyError(f"{self.id}: no edge {key}")

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.behavioural_edges if e.dst == node_id]

    def interventions(self, option_id: str) -> dict[str, dict[str, Any]]:
        raw = self.nodes[option_id].get("interventions") or {}
        return dict(raw)

    def identity(self, node_id: str) -> dict[str, Any] | None:
        ident = self.nodes[node_id].get("nonlinear_identity")
        return ident if isinstance(ident, dict) else None

    @property
    def identity_targets(self) -> list[str]:
        return [n for n in self.nodes if self.identity(n) is not None]

    def resolve(self, ptr: str) -> Any:
        """Resolve a mapping pointer: ``nodes/<id>/<field>/...``, ``edges/<a>-><b>/...``,
        ``goal_constraints/<node_id>/...`` or ``brief``. Raises KeyError when absent."""
        parts = ptr.split("/")
        head = parts[0]
        cur: Any
        if head == "brief":
            return self.brief
        if head == "nodes":
            cur = self.nodes[parts[1]]
        elif head == "edges":
            cur = self.edge(parts[1]).raw
        elif head == "goal_constraints":
            matches = [c for c in self.constraints if c["node_id"] == parts[1]]
            if len(matches) != 1:
                raise KeyError(ptr)
            cur = matches[0]
        else:
            raise KeyError(ptr)
        for part in parts[2:]:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(ptr)
            cur = cur[part]
        return cur


def load_graph(graph_id: str) -> Graph:
    doc = json.loads((CORPUS_DIR / f"{graph_id}.json").read_text())
    return Graph(graph_id, doc)
