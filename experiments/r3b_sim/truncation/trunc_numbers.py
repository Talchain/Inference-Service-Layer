import json, sys, numpy as np
sys.path.insert(0, ".")
from tests.unit.test_anchored_delta_levels import served_wire, analyse, effect_gbp, effect_se_gbp, with_the_served_truncated_sampler, P59, P54, CONVERSION, RETENTION, KEEP
from tests.unit.test_edge_strength_unbounded import _double_the_frame
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
legacy = lambda d: with_the_served_truncated_sampler(analyse, d)
print("== A (a6ed1bff served wire), GBP effects vs keep-current")
for name, fn in (("truncated (served)", legacy), ("unbounded (this PR)", analyse)):
    r = fn(served_wire()); r2 = fn(_double_the_frame(served_wire(), "pro_plan_price"))
    wins = {o.option_id: round(o.win_probability, 4) for o in r.results}
    print(f"{name:20s} P59 {effect_gbp(r,P59):9.2f} ±{effect_se_gbp(r,P59):.2f} | price frame x2: {effect_gbp(r2,P59):9.2f} | P54 {effect_gbp(r,P54):8.2f} conv {effect_gbp(r,CONVERSION):7.2f} ret {effect_gbp(r,RETENTION):6.2f}")
    print(f"{'':20s} win shares {wins}")
    fin = all(np.isfinite(o.outcome_distribution.mean) and np.isfinite(o.outcome_distribution.std) for o in r.results)
    print(f"{'':20s} outcomes finite: {fin}; means range {min(o.outcome_distribution.mean for o in r.results):.4f}..{max(o.outcome_distribution.mean for o in r.results):.4f}")
print("== E (eh4, PLoT ccd602c -> ISL request)")
e = json.load(open(sys.argv[1]))
for name, fn in (("truncated (served)", lambda d: with_the_served_truncated_sampler(lambda x: RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(x)), d)),
                 ("unbounded (this PR)", lambda d: RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d)))):
    r = fn(e)
    wins = {o.option_id: round(o.win_probability, 4) for o in r.results}
    fin = all(np.isfinite(o.outcome_distribution.mean) and np.isfinite(o.outcome_distribution.std) for o in r.results)
    print(f"{name:20s} win {wins} finite {fin} means {[round(o.outcome_distribution.mean,4) for o in r.results]}")
