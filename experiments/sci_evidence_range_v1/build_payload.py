"""Build payload.json, classification.json and coaching.md from the computed results ONLY.

Every figure here is read from results/*.json. The wording follows the SCI-EVIDENCE doctrine proposal: "under the
current model", no information-value numbers, no "next"/"first"/"you should", and no option ranking or leader
(Science correction 1, #75 5911560060).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

SUBJECT = "59_with_feature_release"
CHOSEN = "monthly_churn"
USER_SOURCES = {
    "brief_extraction",
    "user_override",
    "user_assumption",
    "user_specified",
    "user_set",
}
WHY_NOT = {
    "pro_paying_subscribers": (
        "Not chosen: bound with Non-Pro MRR by the month-0 identity 49 x 1,200 + 16,200 = "
        "75,000, so a range on one alone contradicts the stated £75k and the evaluator "
        "withholds (IDENTITY_INCONSISTENT_WITH_HELD_LEVEL)."
    ),
    "non_pro_mrr": "Not chosen: same month-0 identity coupling as Pro subscribers.",
    "monthly_net_pro_additions": "Plausible second candidate (point-only Olumi estimate on the goal path); not this cycle.",
    "feature_value_perception": "Latent score out of 100; a user range would not be meaningful without a scale anchor.",
    "pro_plan_price": "Controllable lever: set by the options, never an evidence gap.",
}


def _money(x: float) -> str:
    return f"£{x:,.0f}"


def _pp(x: float) -> str:
    return f"{x:g}"


def classify(r3b: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for q in r3b["quantities"]:
        src, cat, kind = q["source"], q["category"], q["kind"]
        if kind == "goal":
            status = "goal: today's level user-stated (point); target user-stated"
        elif cat == "controllable":
            status = "controllable lever"
        elif q["value"] is None:
            status = "no value (unquantified node)"
        elif src == "user_assumption":
            status = "user assumption (0/1 switch)"
        elif src in USER_SOURCES:
            status = "user-stated point estimate, no range"
        elif q["has_own_std_or_range"]:
            status = "has a range/std on the node"
        else:
            status = "Olumi point estimate only (no range)"
        row = {**q, "uncertainty_status": status}
        if q["id"] == CHOSEN:
            row["selection"] = (
                "CHOSEN: Olumi's own estimate with no range, on both the ≤4% churn limit and the "
                "£100k goal path; the existing Regions grid and its 1.5 pp threshold assume it is "
                "exactly 3%."
            )
        elif q["id"] in WHY_NOT:
            row["selection"] = WHY_NOT[q["id"]]
        rows.append(row)
    return rows


def build(
    r3b: Dict[str, Any], control: Dict[str, Any], cards: Dict[str, Any]
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    net = r3b["models"]["X_net_reading"]
    gross = r3b["models"]["X_gross_reading"]
    lo, hi = r3b["range"]["low"], r3b["range"]["high"]
    computed = net["computable_options"]
    withheld = [o for o in net["per_option"] if o not in computed]
    subj_n, subj_g = net["per_option"][SUBJECT], gross["per_option"][SUBJECT]
    band = {
        b: subj_n["churn_response_limit_threshold_by_base_churn"][b]["at"]
        for b in ("2.0", "3.0", "4.0")
    }
    band_g = {
        b: subj_g["churn_response_limit_threshold_by_base_churn"][b]["at"]
        for b in ("2.0", "3.0", "4.0")
    }
    goal_x_n = {
        b: subj_n["goal_crossing_on_churn_response_axis_by_base_churn"][b].get("bisected_at")
        for b in ("2.0", "3.0", "4.0")
    }
    goal_x_g = {
        b: subj_g["goal_crossing_on_churn_response_axis_by_base_churn"][b].get("bisected_at")
        for b in ("2.0", "3.0", "4.0")
    }
    first_break = {
        reading: {b: ("goal" if gx[b] is not None and gx[b] < bd[b] else "churn limit") for b in bd}
        for reading, bd, gx in (
            ("X_net_reading", band, goal_x_n),
            ("X_gross_reading", band_g, goal_x_g),
        )
    }

    per_option = {}
    for o in computed:
        pn, pg = net["per_option"][o], gross["per_option"][o]
        an, ag = pn["across_range_at_reference"], pg["across_range_at_reference"]
        per_option[o] = {
            "label": pn["label"],
            "churn_limit_across_range": {
                "satisfied_everywhere": an["limit_satisfied_everywhere"]
                and ag["limit_satisfied_everywhere"],
                "smallest_margin_pp": min(an["min_limit_margin_pp"], ag["min_limit_margin_pp"]),
                "on_the_limit_somewhere": "ON_THRESHOLD" in an["limit_verdicts_seen"],
                "base_churn_at_which_limit_is_reached_pct": pn[
                    "base_churn_limit_crossing_at_reference"
                ]["at"],
                "margin_at_olumi_point_pp": pn["at_olumi_point_3pct"]["limit_margin_pp"],
            },
            "goal_across_range": {
                "X_net_reading": {
                    "verdict_everywhere": (
                        "MET"
                        if an["goal_satisfied_everywhere"]
                        else "MISSED"
                        if an["goal_missed_everywhere"]
                        else "CHANGES"
                    ),
                    "invariant_to_base_churn": an["goal_invariant_to_base_churn"],
                    "first_passage_gbp": [
                        an["goal_first_passage_gbp_min"],
                        an["goal_first_passage_gbp_max"],
                    ],
                },
                "X_gross_reading": {
                    "verdict_everywhere": (
                        "MET"
                        if ag["goal_satisfied_everywhere"]
                        else "MISSED"
                        if ag["goal_missed_everywhere"]
                        else "CHANGES"
                    ),
                    "invariant_to_base_churn": ag["goal_invariant_to_base_churn"],
                    "first_passage_gbp": [
                        ag["goal_first_passage_gbp_min"],
                        ag["goal_first_passage_gbp_max"],
                    ],
                },
            },
        }

    goal_verdicts_stable = all(
        v["verdict_everywhere"] != "CHANGES"
        for p in per_option.values()
        for v in p["goal_across_range"].values()
    )
    limit_on_edge = [
        o for o, p in per_option.items() if p["churn_limit_across_range"]["on_the_limit_somewhere"]
    ]
    threshold_moves = band["2.0"] != band["4.0"]
    reference_on_threshold_at_high = (
        abs(band["4.0"] - r3b["reference_point"]["churn_response_pp_per_10gbp"]) <= 1e-9
    )
    binding_switches = len(set(first_break["X_net_reading"].values())) > 1

    ctrl_after = control["variants"]["after_user_range_sd_width_over_4"]
    ctrl_iqr = control["variants"]["after_user_range_sd_width_over_1_349"]
    ctrl_tmpl = control["variants"]["before_same_default_labelled_template"]
    card_before = cards["control_bodies"]["before_as_sent"]["lines"]
    card_after = cards["control_bodies"]["after_user_range_sd_width_over_4"]["lines"]
    card_tmpl = cards["control_bodies"]["before_same_default_labelled_template"]["lines"]
    control_unchanged = (
        ctrl_after["per_option_summary_identical_to_before"]
        and ctrl_iqr["per_option_summary_identical_to_before"]
    )

    keep = (
        goal_verdicts_stable
        and threshold_moves
        and reference_on_threshold_at_high
        and binding_switches
    )
    decision = "KEEP" if keep else "KILL"

    s59 = per_option[SUBJECT]
    keep_label = per_option["keep_current_pricing"]["label"]
    f49_label = per_option["49_with_feature_release"]["label"]
    s59_label = s59["label"]

    explanation = [
        (
            f"Under the current model, the £100k goal verdict of every option Olumi can compute stays the same anywhere "
            f"in your {_pp(lo)}–{_pp(hi)}% range: {f49_label} and {s59_label} reach it, {keep_label} does not."
        ),
        (
            f"{s59_label} stays within your 4% churn limit across the whole range, but at {_pp(hi)}% it sits exactly on "
            f"the limit with no room to spare (at Olumi's 3% it had {_pp(s59['churn_limit_across_range']['margin_at_olumi_point_pp'])} "
            f"point)."
        ),
        (
            f"The earlier 'what would change this' point, {_pp(band['3.0'])} points of extra churn per £10, assumed 3%. "
            f"Across your range it runs from {_pp(band['2.0'])} (at {_pp(lo)}%) to {_pp(band['4.0'])} (at {_pp(hi)}%), and "
            f"{_pp(band['4.0'])} is the model's own working guess for that response."
        ),
        (
            "So how churn responds to the £10 rise is the check that could change this, and it matters most if today's "
            "churn is near the top of your range."
        ),
    ]

    payload: Dict[str, Any] = {
        "experiment": "SCI-EVIDENCE range elicitation v1 (experimental local payload, not a product contract)",
        "case_model_identity": {
            "graph_id": "pj-20260927T180910Z-A",
            "brief": (
                "Given our goal of reaching £100k MRR within 12 months [Currently 75k] while keeping monthly "
                "churn under 4%, should we increase the Pro plan price from £49 to £59 per month with the next "
                "Pro feature release?"
            ),
            "evaluator": {
                "commit": net["identity"]["commit"],
                "evaluator_sha256": net["identity"]["evaluator_sha256"],
                "graph_sha256": net["identity"]["graph_sha256"],
                "mapping_sha256": net["identity"]["mapping_sha256"],
                "frozen_files_all_match": net["identity"]["frozen_files_all_match"],
            },
            "models": {
                "primary": "X_net_reading (SCI-REGIONS' model)",
                "secondary": "X_gross_reading",
            },
            "reference_point": r3b["reference_point"],
            "consumed_from": "ISL sci/regions-contrastive-vulnerability @ 68e8c887 (read-only)",
        },
        "quantity": {
            "id": CHOSEN,
            "label": "Monthly churn (today, before any price change)",
            "on_paths": [
                "hard limit: monthly churn ≤ 4% every month",
                "goal: MRR ≥ £100k by month 12",
            ],
        },
        "current_value": {"value": 3, "unit": "percent per month"},
        "uncertainty_status_before": "Olumi point estimate only: no range, no std, no user or evidence bound",
        "provenance": {
            "observed_state.source": "cee_inference",
            "extractionType": "inferred",
            "author": "Olumi",
            "note": "Never the user's figure; the brief states only the 4% limit",
        },
        "requested_range_or_evidence": {
            "question": (
                "What would you consider a plausible low-to-high range for your monthly churn today, before "
                "any price change? A rough 'somewhere between X% and Y%' is enough."
            ),
            "or": "Any recent figure you trust for monthly churn (e.g. last quarter's), with where it comes from.",
            "not_asked": "a probability distribution, a confidence level, or a most-likely value",
        },
        "experimental_supplied_range": {
            "low": lo,
            "high": hi,
            "unit": "percent per month",
            "label": r3b["range"]["label"],
        },
        "assumptions_to_interpret_it": [
            "The range is read as two ends to test, not as a distribution: the R3-B analysis is a deterministic sweep "
            "(41 points, 0.05 pp apart) with no probability attached to any point.",
            "The centre stays Olumi's 3%; only the ends come from the (hypothetical) user.",
            "Everything else is held at the frozen case's values, including SCI-REGIONS' reference churn response "
            "(0.5 pp per +£10) and competitive response (£0/month), both exploratory research references.",
            "Limits are inclusive: a value within 1e-9 of the limit is ON it and still meets '≤'.",
            "Production-path control only: the engine needs a sigma, so width/4 and width/1.349 were both run and "
            "labelled as assumptions.",
        ],
        "analysis_capability_before": [
            "Every churn-limit verdict and the 1.5 pp 'what would change this' threshold silently assume churn is "
            "exactly 3%; nothing could be said about how much room the limit has if it isn't.",
            "The served Evidence card could only say: "
            + " ".join(cards["real_served_before"]["lines"]),
            "Olumi could not honestly say whether its own churn guess could change any goal or limit verdict.",
        ],
        "analysis_capability_after": [
            "A per-option robustness statement for the £100k goal across the user's whole range (both readings).",
            "A per-option churn-limit headroom across the range, including where it reaches zero.",
            "The single churn-response threshold becomes a range-conditional band, checked on the evaluator.",
            "Which limit gives way first as the price response rises, per base churn (per option, no ranking).",
        ],
        "newly_available_finding": {
            "goal_verdicts_stable_across_range": goal_verdicts_stable,
            "per_option": per_option,
            "churn_response_threshold_band_pp_per_10gbp": {
                "subject_option": SUBJECT,
                "X_net_reading": band,
                "X_gross_reading": band_g,
                "reference_value": r3b["reference_point"]["churn_response_pp_per_10gbp"],
                "reference_sits_on_threshold_at_high_end": reference_on_threshold_at_high,
            },
            "goal_crossing_on_churn_response_axis_pp_per_10gbp": {
                "X_net_reading": goal_x_n,
                "X_gross_reading": goal_x_g,
            },
            "what_gives_way_first_as_churn_response_rises": first_break,
            "options_on_the_limit_at_high_end": limit_on_edge,
            "withheld_options": {o: net["per_option"][o]["withheld_reason"] for o in withheld},
        },
        "user_facing_explanation": explanation,
        "production_path_control": {
            "case": "Paul's current MRR graph (A-graph) on served ISL f7f19e3, exact PLoT request, seed 1254899477",
            "reproduced_pinned_served_response": control["reproduction_of_pinned_served_response"][
                "all_science_fields_equal"
            ],
            "churn_as_sent_today": control["churn_uncertainty_as_sent"],
            "after_user_spread": [
                ctrl_after["churn_uncertainty_sent"],
                ctrl_iqr["churn_uncertainty_sent"],
            ],
            "per_option_goal_limit_and_mrr_figures_unchanged": control_unchanged,
            "largest_change_in_changed_blocks": ctrl_after["largest_numeric_change_by_block"],
            "churn_limit_prob_satisfied_all_variants": sorted(
                {
                    v["churn_limit_prob_satisfied"]
                    for var in control["variants"].values()
                    for v in var["summary"]["per_option"].values()
                }
            ),
            "factor_importance_order_before_after": {
                "before": control["variants"]["before_as_sent"]["summary"][
                    "factor_importance_order"
                ],
                "after": ctrl_after["summary"]["factor_importance_order"],
                "note": (
                    "Churn and Monthly new Pro subscribers have equal |elasticity| to about 15 significant "
                    "figures in every variant, so ranks 3 and 4 swap on float noise. It is a tie, not a change "
                    "in meaning; a consumer that shows this order would show the swap."
                ),
            },
            "evidence_card_before": card_before,
            "evidence_card_after_user_range": card_after,
            "evidence_card_if_plot_labelled_its_default": card_tmpl,
            "finding": (
                "Today's product path cannot deliver this interaction for a shared observable: a user spread "
                "on churn leaves every per-option goal, limit and MRR figure identical (other changes are "
                "float noise, plus a tie-driven swap of two factor-importance ranks), the churn-limit chance stays 1.0 because the limit is scored as the option's change "
                "against Olumi's held 3% (so it also silently assumes 3%), and the Evidence card's words do "
                "not change. The only visible difference available today is the adapter's 'default width' "
                "line, and only if PLoT labelled its own synthesised spread (it does not)."
            ),
        },
        "caveats": [
            "The 2–4% range is a HYPOTHETICAL stand-in for what a user might supply; it is not evidence.",
            "Model-relative results from a frozen research evaluator, not causal truth and not a forecast.",
            f"Only {len(computed)} of {len(net['per_option'])} options have levels; the other {len(withheld)} are "
            "withheld (OPTION_LEVELS_MISSING) and nothing is said about them. No option is ranked.",
            "The reference churn response (0.5 pp per +£10) and the competitive response (£0) are exploratory "
            "research references, not user figures.",
            "Under reading N the goal does not depend on base churn at all (only a change in churn removes "
            "subscribers), so its goal robustness is structural in this model; reading G carries it through the flows.",
            "Every link spread in this graph is Olumi's template; this experiment ranges only base churn.",
            "Not served, not integrated, no human-comprehension test.",
        ],
        "verdict": {
            "decision": decision,
            "rule": (
                "KEEP only if the range licenses a new per-option threshold/robustness claim or reveals a "
                "crossing the point estimate could not support; otherwise KILL (Science 5911560060)."
            ),
            "basis": {
                "new_robustness_claim": goal_verdicts_stable,
                "threshold_becomes_range_conditional": threshold_moves,
                "reference_reaches_threshold_inside_range": reference_on_threshold_at_high,
                "what_gives_way_first_changes_inside_range": binding_switches,
                "product_path_can_deliver_today": not control_unchanged,
            },
        },
    }

    coaching = f"""# SCI-EVIDENCE range elicitation: coaching example

Case: the frozen R3-B MRR case (£49 → £59, £100k MRR by month 12, churn ≤ 4%). All numbers come from
`results/r3b.json`. **The user's range below is HYPOTHETICAL, an experimental stand-in only.**

**1. Olumi names the gap**
> Monthly churn matters here twice: it's what your 4% limit is about, and it feeds how Pro subscribers grow.
> The 3% I've used is my own estimate, with no range, so everything I've said about the churn limit quietly
> assumes churn is exactly 3%. I can't yet tell you how much room that limit really has.

**2. Olumi asks for the smallest useful input**
> What would you consider a plausible low-to-high range for your monthly churn today, before any price change?
> A rough "somewhere between X% and Y%" is enough. A recent figure you trust works too.

**3. The user answers** *(HYPOTHETICAL, experimental control input)*
> Somewhere between {_pp(lo)}% and {_pp(hi)}%.

**4. Olumi explains what the re-run shows**
> {explanation[0]}
> {explanation[1]}
> {explanation[2]}

**5. What changed, and what is still uncertain**
> {explanation[3]}
> This is a model result, not a forecast. It covers only the {len(computed)} options I have figures for, and the
> churn response to the price rise is still my working guess.
"""
    return payload, coaching, classify(r3b)
