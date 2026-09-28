"""Create a small, deterministic archive and readable top-level study result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from r3b_adapter import PricingAdapter

ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    output = ROOT / "output"
    manifest = json.loads((output / "run-manifest.json").read_text())
    raw = sorted(p for p in output.glob("*.json") if p.name != "results.json")
    archive = output / "results.zip"
    members = {}
    with ZipFile(archive, "w") as z:
        for path in raw:
            data = path.read_bytes()
            entry = ZipInfo(path.name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            z.writestr(entry, data, compress_type=ZIP_DEFLATED, compresslevel=9)
            members[path.name] = digest(data)
    net = json.loads((output / "r3b-X_net_reading-41.json").read_text())
    gross = json.loads((output / "r3b-X_gross_reading-41.json").read_text())
    def competitive(result: dict) -> dict:
        p = next(p for p in result["points"] if p["x"] == 0 and p["y"] == -31250)
        o = p["options"]["59_with_feature_release"]
        return {"goal": o["goal"], "month_12_mrr_gbp": o["objective_value"], "churn_percent": o["constraints"][0]["value"], "churn_constraint": o["constraints"][0]["state"], "named_preference": p["named_preference"], "overall_preference": p["overall_preference"]}
    summary = {
        "schema": "SCI-REGIONS-study-results-v1", "status": manifest["status"],
        "source_commit": manifest["source_commit"], "source_identity": PricingAdapter("X_net_reading").identity,
        "synthetic": {"fixture_grids": len(manifest["synthetic"]), "evaluated_coordinates": sum(r["evaluated"] for r in manifest["synthetic"]), "false_feasible": sum(r["false_feasible"] for r in manifest["synthetic"]), "false_confident_preference": sum(r["false_confident_preference"] for r in manifest["synthetic"]), "f11_coarse_missed_island": any(r["case"] == "F11" and r["grid"] == 41 and r["topology_failures"] for r in manifest["synthetic"])},
        "r3b": {"evaluated_coordinates": sum(r["evaluated"] for r in manifest["r3b"]), "full_comparison": "INCOMPLETE_COMPARISON", "net_competitive_zero_churn": competitive(net), "gross_competitive_zero_churn": competitive(gross), "false_feasible": None, "false_confident_preference": None},
        "baseline_reproduction": manifest["baseline"]["status"],
        "runtime_manifest": "run-manifest.json", "offline_visual": "comparison.html", "raw_archive": {"file": archive.name, "sha256": digest(archive.read_bytes()), "members": members},
        "not_run": ["Monte Carlo intervals/coverage", "3-D", "Human comparison", "Deployed-current comparator"],
    }
    (output / "results.json").write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "raw_archive_bytes": archive.stat().st_size, "members": len(members)}, indent=2))


if __name__ == "__main__":
    main()
