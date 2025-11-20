"""
Explanation generator for creating human-readable explanations.

Generates plain English explanations of technical analysis results
that are accessible to non-technical decision makers.
"""

from typing import List

from src.models.shared import ExplanationMetadata


class ExplanationGenerator:
    """Generates explanations for analysis results."""

    @staticmethod
    def generate_causal_validation_explanation(
        status: str,
        treatment: str,
        outcome: str,
        adjustment_sets: List[List[str]] = None,
        minimal_set: List[str] = None,
        issues: List[dict] = None,
    ) -> ExplanationMetadata:
        """
        Generate explanation for causal validation results.

        Args:
            status: Validation status (identifiable, uncertain, cannot_identify, degraded)
            treatment: Treatment variable name
            outcome: Outcome variable name
            adjustment_sets: Valid adjustment sets (if identifiable)
            minimal_set: Minimal adjustment set (if identifiable)
            issues: Validation issues (if not identifiable)

        Returns:
            ExplanationMetadata: Structured explanation
        """
        if status == "identifiable":
            if minimal_set:
                control_vars = ", ".join(minimal_set) if minimal_set else "no variables"
                summary = f"Effect is identifiable by controlling for {control_vars}"
                reasoning = (
                    f"To isolate the causal effect of {treatment} on {outcome}, "
                    f"you need to control for {control_vars}. "
                    f"This blocks all backdoor paths that create spurious correlation."
                )
                technical_basis = f"Backdoor criterion satisfied with adjustment set {{{control_vars}}}"
            else:
                summary = "Effect is identifiable (no controls needed)"
                reasoning = (
                    f"The causal effect of {treatment} on {outcome} can be identified "
                    f"without controlling for additional variables. There are no backdoor paths."
                )
                technical_basis = "No backdoor paths exist; causal effect is directly identifiable"

            assumptions = [
                "No unmeasured confounding",
                "Correct causal structure specified",
                "Causal sufficiency (all relevant variables included)",
            ]

        elif status == "uncertain":
            summary = "Causal identification uncertain due to potential issues"
            reasoning = (
                f"Without additional information about the causal structure, "
                f"we cannot confidently identify the effect of {treatment} on {outcome}. "
                f"The issues below need to be resolved."
            )
            technical_basis = "Backdoor criterion cannot be verified with current structure"
            assumptions = [
                "Current structure may be incomplete",
                "Additional edges or variables may exist",
            ]

        elif status == "degraded":
            summary = "Advanced causal analysis unavailable; using fallback structural assessment"
            reasoning = (
                f"The advanced causal identification engine encountered an error while analyzing "
                f"the effect of {treatment} on {outcome}. A basic structural assessment has been "
                f"provided instead. This may indicate an issue with the graph structure complexity "
                f"or an internal analysis error. Manual review by a causal inference expert is recommended."
            )
            technical_basis = "Y₀ causal identification failed; fallback to structural path analysis"
            assumptions = [
                "Fallback structural analysis may be incomplete",
                "Manual expert review recommended before making decisions",
                "Graph may need simplification or restructuring",
            ]

        else:  # cannot_identify
            summary = "Causal effect cannot be identified from this structure"
            reasoning = (
                f"The causal effect of {treatment} on {outcome} is not identifiable "
                f"from the given graph structure. Fundamental issues prevent identification."
            )
            technical_basis = "Backdoor criterion fails; effect is not identifiable"
            assumptions = [
                "Graph structure accurately represents causal relationships",
                "No measurement or intervention available to resolve issues",
            ]

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis=technical_basis,
            assumptions=assumptions,
        )

    @staticmethod
    def generate_counterfactual_explanation(
        outcome: str,
        intervention: dict,
        point_estimate: float,
        ci_lower: float,
        ci_upper: float,
        uncertainty_level: str,
        robustness_level: str,
    ) -> ExplanationMetadata:
        """
        Generate explanation for counterfactual analysis.

        Args:
            outcome: Outcome variable
            intervention: Intervention dictionary
            point_estimate: Point estimate value
            ci_lower: Lower confidence interval
            ci_upper: Upper confidence interval
            uncertainty_level: Overall uncertainty (low/medium/high)
            robustness_level: Robustness score (robust/moderate/fragile)

        Returns:
            ExplanationMetadata: Structured explanation
        """
        # Format intervention for display
        intervention_str = ", ".join(
            f"{var}={val}" for var, val in intervention.items()
        )

        # Create summary
        if ci_lower >= 0 and ci_upper >= 0:
            direction = "increases"
        elif ci_lower <= 0 and ci_upper <= 0:
            direction = "decreases"
        else:
            direction = "changes"

        summary = (
            f"{outcome} likely {direction} to {point_estimate:.0f} "
            f"(range: {ci_lower:.0f} to {ci_upper:.0f})"
        )

        # Create reasoning based on uncertainty
        if uncertainty_level == "low":
            uncertainty_note = "The prediction is relatively certain"
        elif uncertainty_level == "medium":
            uncertainty_note = "There is moderate uncertainty in the prediction"
        else:
            uncertainty_note = "There is substantial uncertainty in the prediction"

        reasoning = (
            f"Under the intervention {intervention_str}, {outcome} is predicted to be "
            f"{point_estimate:.0f}. {uncertainty_note}, as reflected in the confidence "
            f"interval range. The uncertainty comes from multiple sources including "
            f"parameter uncertainty, structural assumptions, and external factors."
        )

        # Technical basis
        technical_basis = (
            "FACET region-based counterfactual analysis with Monte Carlo sampling "
            "(10,000 iterations) and uncertainty propagation"
        )

        # Assumptions
        assumptions = [
            "Structural equations correctly specified",
            "Prior distributions reflect true uncertainty",
            "No unmeasured confounders affect outcome",
            "Intervention can be implemented as specified",
        ]

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis=technical_basis,
            assumptions=assumptions,
        )

    @staticmethod
    def generate_team_alignment_explanation(
        num_perspectives: int,
        agreement_level: float,
        top_option: str,
        satisfaction_score: float,
        num_conflicts: int,
    ) -> ExplanationMetadata:
        """
        Generate explanation for team alignment.

        Args:
            num_perspectives: Number of team perspectives
            agreement_level: Agreement percentage (0-100)
            top_option: Recommended option ID
            satisfaction_score: Satisfaction score for top option
            num_conflicts: Number of conflicts identified

        Returns:
            ExplanationMetadata: Structured explanation
        """
        if agreement_level >= 80:
            alignment_desc = "strong alignment"
        elif agreement_level >= 60:
            alignment_desc = "moderate alignment"
        else:
            alignment_desc = "limited alignment"

        summary = f"Team shows {alignment_desc} with {top_option} satisfying {satisfaction_score:.0f}% of priorities"

        if num_conflicts == 0:
            conflict_note = "with no major conflicts"
        elif num_conflicts == 1:
            conflict_note = "with one minor conflict to resolve"
        else:
            conflict_note = f"with {num_conflicts} conflicts to address"

        reasoning = (
            f"Across {num_perspectives} team perspectives, there is {agreement_level:.0f}% "
            f"agreement on shared goals and constraints. The recommended option '{top_option}' "
            f"achieves the highest overall satisfaction ({satisfaction_score:.0f}%) {conflict_note}. "
            f"This option balances the priorities of all stakeholders while acknowledging "
            f"necessary tradeoffs."
        )

        technical_basis = (
            "Overlap analysis with priority weighting and satisfaction scoring. "
            "Conflicts identified through mutual exclusivity detection."
        )

        assumptions = [
            "Stated priorities accurately reflect true values",
            "Options can be objectively evaluated on stated criteria",
            "Team will engage in good-faith negotiation on tradeoffs",
        ]

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis=technical_basis,
            assumptions=assumptions,
        )

    @staticmethod
    def generate_sensitivity_explanation(
        baseline_result: float,
        robustness_level: str,
        num_critical: int,
        num_moderate: int,
        num_minor: int,
    ) -> ExplanationMetadata:
        """
        Generate explanation for sensitivity analysis.

        Args:
            baseline_result: Baseline result value
            robustness_level: Overall robustness (robust/moderate/fragile)
            num_critical: Number of critical assumptions
            num_moderate: Number of moderate importance assumptions
            num_minor: Number of minor importance assumptions

        Returns:
            ExplanationMetadata: Structured explanation
        """
        if robustness_level == "robust":
            robustness_desc = "highly robust"
            conclusion = "holds across a wide range of assumption variations"
        elif robustness_level == "moderate":
            robustness_desc = "moderately robust"
            conclusion = "holds under most reasonable assumption variations"
        else:
            robustness_desc = "fragile"
            conclusion = "depends heavily on specific assumptions being correct"

        summary = f"Conclusion is {robustness_desc} with {num_critical} critical assumption(s)"

        reasoning = (
            f"The baseline result of {baseline_result:.0f} was tested against variations "
            f"in {num_critical + num_moderate + num_minor} assumptions. The analysis found "
            f"{num_critical} critical assumptions, {num_moderate} moderately important assumptions, "
            f"and {num_minor} minor assumptions. The conclusion {conclusion}."
        )

        if num_critical > 0:
            reasoning += (
                f" However, the {num_critical} critical assumption(s) should be validated "
                f"before making final decisions."
            )

        technical_basis = (
            "One-at-a-time sensitivity analysis with ±30% perturbations on each assumption. "
            "Variance decomposition used to attribute uncertainty to specific factors."
        )

        assumptions = [
            "Assumptions are independent (interaction effects not tested)",
            "Linear perturbation ranges are appropriate",
            "Historical data predicts future behavior",
        ]

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis=technical_basis,
            assumptions=assumptions,
        )
