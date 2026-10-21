"""Finding 2 probe: replay captured PLoT→ISL sum-route requests in-process against an ISL tree.
usage: python f2_probe.py <isl_dir> <capture.json>..."""
import json
import os
import sys

isl = os.path.abspath(sys.argv[1])
sys.path.insert(0, isl)
os.chdir(isl)
from src.models.robustness_v2 import RobustnessRequestV2  # noqa: E402
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

for path in sys.argv[2:]:
    body = json.load(open(path))["request"]
    resp = RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**body)).model_dump(mode="json", by_alias=True)
    ident = [(i["node_id"], i["evaluated"], i.get("level_source")) for i in resp.get("identity_evaluations") or []]
    print(os.path.basename(path), body.get("request_id"), "identity", ident)
    for o in resp["results"]:
        d = o["outcome_distribution"]
        print("  ", o["option_id"], repr(d["mean"]), "sd", d["std"], "P", o.get("probability_of_goal"))
