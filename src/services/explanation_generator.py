"""
Explanation generator service for producing layered, audience-appropriate explanations.

Generates clear, non-technical explanations using templates and progressive disclosure.
"""

import logging
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProgressiveExplanation(BaseModel):
    """Progressive disclosure explanation with three levels."""

    headline: str = Field(
        ...,
        description="Simple, non-technical headline (1 sentence)",
        max_length=200
    )

    summary: str = Field(
        ...,
        description="Intermediate explanation for business users (2-3 sentences)",
        max_length=500
    )

    details: str = Field(
        ...,
        description="Technical details for experts",
        max_length=1000
    )

    visual_aid: Optional[str] = Field(
        None,
        description="Suggested visualization type"
    )

    learn_more_url: Optional[str] = Field(
        None,
        description="Link to documentation"
    )


class ExplanationGenerator:
    """
    Generate layered explanations for different audiences.

    Provides three levels of explanation:
    - Simple: Non-technical, for general audience
    - Intermediate: Business users with domain knowledge
    - Technical: Data scientists and researchers
    """

    TEMPLATES = {
        "conformal_prediction": {
            "simple": "This prediction has a {confidence}% guaranteed accuracy range.",
            "intermediate": "Using conformal prediction, we guarantee {confidence}% coverage: the true value will fall in [{lower:.0f}, {upper:.0f}] at least {confidence}% of the time.",
            "technical": "Conformal prediction provides distribution-free, finite-sample valid intervals with coverage ≥ {guaranteed_coverage}% using {n_calibration} calibration points and split conformal algorithm.",
            "visual": "interval_plot",
            "url": "/docs/methods/conformal-prediction"
        },
        "non_identifiable": {
            "simple": "We can't estimate this effect reliably due to confounding.",
            "intermediate": "Confounding means other factors affect both {treatment} and {outcome}, preventing us from isolating the causal effect.",
            "technical": "The causal effect is non-identifiable: no adjustment set blocks all backdoor paths from {treatment} to {outcome}. Graph analysis shows {num_backdoor} unblocked backdoor paths.",
            "visual": "path_diagram",
            "url": "/docs/concepts/identification"
        },
        "identifiable_backdoor": {
            "simple": "We can estimate this effect by controlling for {num_controls} key variables.",
            "intermediate": "By controlling for {controls}, we can block confounding paths and isolate the true causal effect of {treatment} on {outcome}.",
            "technical": "The effect is identifiable via backdoor adjustment. Minimal adjustment set: {{{adjustment_set}}}. This blocks all {num_backdoor} backdoor paths while avoiding collider bias.",
            "visual": "dag_with_adjustment",
            "url": "/docs/methods/backdoor-adjustment"
        },
        "identifiable_frontdoor": {
            "simple": "We can estimate this effect through intermediate variables.",
            "intermediate": "Even with hidden confounding, we can use mediator variables ({mediators}) to estimate the causal effect.",
            "technical": "The effect is identifiable via frontdoor adjustment through mediators {{{mediator_set}}}. This works because mediators are: (1) fully mediate the effect, (2) have no confounders with outcome, (3) are blocked from spurious paths.",
            "visual": "frontdoor_diagram",
            "url": "/docs/methods/frontdoor-adjustment"
        },
        "counterfactual_uncertainty": {
            "simple": "The predicted outcome is {point_estimate:.0f}, but could reasonably range from {lower:.0f} to {upper:.0f}.",
            "intermediate": "Our model predicts {point_estimate:.0f} for {outcome} when we set {treatment} to {value}. The {confidence}% uncertainty interval is [{lower:.0f}, {upper:.0f}].",
            "technical": "Counterfactual prediction for do({treatment}={value}) yields {point_estimate:.0f} ± {stderr:.0f} (SE). {confidence}% credible interval: [{lower:.0f}, {upper:.0f}]. Prediction based on {n_samples} Monte Carlo samples from structural equations.",
            "visual": "uncertainty_plot",
            "url": "/docs/methods/counterfactuals"
        },
        "high_sensitivity": {
            "simple": "This result is very sensitive to assumptions - small changes in assumptions can flip conclusions.",
            "intermediate": "The assumption '{assumption}' is critical: a {threshold}% violation changes the outcome by {impact}%. Use caution when making decisions.",
            "technical": "Sensitivity analysis shows elasticity of {elasticity:.2f} for '{assumption}'. This exceeds the criticality threshold ({critical_threshold}). Robustness score: {robustness:.2f}/1.0. Recommend strengthening this assumption or collecting additional data.",
            "visual": "sensitivity_plot",
            "url": "/docs/concepts/sensitivity-analysis"
        },
        "low_sensitivity": {
            "simple": "This result is robust - it holds even if assumptions are somewhat wrong.",
            "intermediate": "Even with {threshold}% violations of key assumptions, the outcome only changes by {impact}%. This gives confidence in the conclusions.",
            "technical": "Sensitivity analysis shows low elasticity ({elasticity:.2f}) across all tested assumptions. Robustness score: {robustness:.2f}/1.0. Results are stable under assumption violations up to {max_violation}%.",
            "visual": "robustness_plot",
            "url": "/docs/concepts/sensitivity-analysis"
        },
        "transportability_success": {
            "simple": "This causal effect can be applied to the new context.",
            "intermediate": "The causal relationship discovered in {source_domain} is transportable to {target_domain} because the key mechanisms are preserved.",
            "technical": "Transportability holds via {method}. Selection diagram analysis confirms all selection nodes ({selection_nodes}) satisfy the back-door criterion relative to transported effect. Estimated transported effect: {effect:.2f}.",
            "visual": "selection_diagram",
            "url": "/docs/methods/transportability"
        },
        "transportability_failure": {
            "simple": "This causal effect may not apply to the new context.",
            "intermediate": "The relationship from {source_domain} cannot be safely transported to {target_domain} due to differences in {blocking_factors}.",
            "technical": "Transportability fails: selection nodes {selection_nodes} create unblocked paths in the selection diagram. Violations detected in {num_violations} transportability conditions. Direct experimentation in target domain recommended.",
            "visual": "selection_diagram",
            "url": "/docs/methods/transportability"
        },
        "causal_discovery_success": {
            "simple": "We discovered {num_edges} causal relationships from the data.",
            "intermediate": "Structure learning identified {num_edges} likely causal edges with {confidence}% average confidence. Top relationships: {top_edges}.",
            "technical": "Discovered DAG using {algorithm} algorithm with {n_samples} samples and {n_vars} variables. {num_edges} edges identified. FDR-adjusted p-values < {alpha}. Structural Hamming Distance from true graph (if known): {shd}.",
            "visual": "discovered_dag",
            "url": "/docs/methods/causal-discovery"
        },
        "batch_optimization": {
            "simple": "Processed {n_scenarios} scenarios in {time:.1f} seconds.",
            "intermediate": "Batch analysis of {n_scenarios} intervention scenarios completed. Best outcome: {best_outcome:.0f} at {best_intervention}. Worst: {worst_outcome:.0f} at {worst_intervention}.",
            "technical": "Batch counterfactual computation via vectorized SCM evaluation. {n_scenarios} scenarios × {n_samples} samples = {total_evals} total evaluations. Parallelized across {n_workers} workers. Throughput: {throughput:.0f} scenarios/sec.",
            "visual": "heatmap",
            "url": "/docs/features/batch-processing"
        },
        "validation_strategies": {
            "simple": "Found {num_strategies} ways to make this effect identifiable.",
            "intermediate": "To identify the causal effect, you could: {strategy_summary}. Each strategy requires different data or assumptions.",
            "technical": "Generated {num_strategies} adjustment strategies via graph-theoretic analysis. Strategies ranked by expected identifiability ({range_str}). Best strategy: {best_strategy} with {best_score:.0%} expected success.",
            "visual": "strategy_comparison",
            "url": "/docs/features/validation-strategies"
        },
        "experiment_recommendation": {
            "simple": "Next, test {intervention} to learn the most while improving outcomes.",
            "intermediate": "Thompson sampling recommends {intervention} as the next experiment. This balances learning (information gain: {info_gain:.2f}) with optimization (expected outcome: {exp_outcome:.0f}).",
            "technical": "Recommendation via Thompson sampling with {n_posterior} posterior samples. Exploration weight: {exploration:.2f}. Expected information gain: {info_gain:.2f} nats. Cost-adjusted value: {value:.2f}. Alternative considered: {alternative}.",
            "visual": "posterior_distributions",
            "url": "/docs/methods/sequential-optimization"
        },
        "insufficient_data": {
            "simple": "Not enough data to make reliable conclusions.",
            "intermediate": "With only {n_samples} observations and {n_vars} variables, we need at least {min_samples} samples for reliable causal inference.",
            "technical": "Insufficient sample size for reliable estimation. Current: n={n_samples}, p={n_vars}. Minimum recommended: n>{min_samples} (based on {criterion} criterion). Power analysis suggests {recommended_n} samples for {power}% power.",
            "visual": "power_curve",
            "url": "/docs/best-practices/sample-size"
        },
        "model_uncertainty": {
            "simple": "Multiple models fit the data equally well, leading to uncertain conclusions.",
            "intermediate": "Model averaging across {n_models} candidate structures yields outcome range [{lower:.0f}, {upper:.0f}]. True value likely within this range.",
            "technical": "Structural uncertainty quantified via Bayesian model averaging over {n_models} DAGs with posterior probabilities {prob_range}. Model-averaged effect: {avg_effect:.2f} ± {model_se:.2f} (model uncertainty) ± {estimation_se:.2f} (estimation uncertainty).",
            "visual": "model_ensemble",
            "url": "/docs/concepts/model-uncertainty"
        }
    }

    # Documentation base URL
    DOCS_BASE = "https://docs.inference-service-layer.com"

    def __init__(self):
        """Initialize the explanation generator."""
        self.logger = logger

    def generate(
        self,
        concept: str,
        data: Dict,
        level: Literal["simple", "intermediate", "technical"] = "intermediate"
    ) -> str:
        """
        Generate explanation at appropriate level.

        Args:
            concept: The concept to explain (key from TEMPLATES)
            data: Data to fill template placeholders
            level: Explanation level (simple/intermediate/technical)

        Returns:
            Generated explanation string
        """
        if concept not in self.TEMPLATES:
            self.logger.warning(f"Unknown concept: {concept}, using generic explanation")
            return self._generate_generic(data, level)

        template_set = self.TEMPLATES[concept]

        if level not in template_set:
            # Fall back to intermediate if level not available
            level = "intermediate"

        template = template_set[level]

        try:
            explanation = template.format(**data)
            return explanation
        except KeyError as e:
            self.logger.error(f"Missing template data key: {e}")
            return f"Explanation unavailable (missing data: {e})"
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return "Explanation unavailable"

    def generate_progressive(
        self,
        concept: str,
        data: Dict
    ) -> ProgressiveExplanation:
        """
        Generate all three levels for progressive disclosure.

        Args:
            concept: The concept to explain
            data: Data to fill template placeholders

        Returns:
            ProgressiveExplanation with all three levels
        """
        if concept not in self.TEMPLATES:
            self.logger.warning(f"Unknown concept: {concept}")
            return self._generate_generic_progressive(data)

        template_set = self.TEMPLATES[concept]

        headline = self.generate(concept, data, "simple")
        summary = self.generate(concept, data, "intermediate")
        details = self.generate(concept, data, "technical")

        visual_aid = template_set.get("visual")
        learn_more_url = self.DOCS_BASE + template_set.get("url", "")

        return ProgressiveExplanation(
            headline=headline,
            summary=summary,
            details=details,
            visual_aid=visual_aid,
            learn_more_url=learn_more_url
        )

    def _generate_generic(self, data: Dict, level: str) -> str:
        """Generate generic explanation when concept not found."""
        if level == "simple":
            return "Analysis completed successfully."
        elif level == "intermediate":
            return f"Completed analysis with {len(data)} data points."
        else:
            return f"Technical analysis complete. Data: {data}"

    def _generate_generic_progressive(self, data: Dict) -> ProgressiveExplanation:
        """Generate generic progressive explanation."""
        return ProgressiveExplanation(
            headline="Analysis completed successfully.",
            summary=f"Completed analysis with results: {list(data.keys())[:3]}",
            details=f"Full analysis data: {data}",
            visual_aid="summary_table",
            learn_more_url=self.DOCS_BASE + "/docs"
        )

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for text.

        Uses simplified versions of standard readability formulas.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with readability scores
        """
        # Count sentences, words, syllables
        sentences = len(re.split(r'[.!?]+', text))
        sentences = max(1, sentences)  # Avoid division by zero

        words = len(text.split())
        words = max(1, words)

        # Simplified syllable count (very approximate)
        syllables = self._count_syllables(text)

        # Flesch Reading Ease: 206.835 - 1.015(words/sentences) - 84.6(syllables/words)
        flesch_reading_ease = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))  # Clamp 0-100

        # Flesch-Kincaid Grade Level: 0.39(words/sentences) + 11.8(syllables/words) - 15.59
        flesch_kincaid_grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        flesch_kincaid_grade = max(0, flesch_kincaid_grade)

        # SMOG (Simplified Measure of Gobbledygook) - simplified
        # Typically: 1.043 * sqrt(polysyllables * 30/sentences) + 3.1291
        # We'll use a simplified approximation
        complex_words = sum(1 for word in text.split() if self._count_syllables_word(word) >= 3)
        smog = 1.043 * (complex_words ** 0.5) + 3.1291

        return {
            "flesch_reading_ease": round(flesch_reading_ease, 1),
            "flesch_kincaid_grade": round(flesch_kincaid_grade, 1),
            "smog_index": round(smog, 1),
            "sentences": sentences,
            "words": words,
            "words_per_sentence": round(words / sentences, 1),
            "syllables_per_word": round(syllables / words, 2)
        }

    def _count_syllables(self, text: str) -> int:
        """Count approximate syllables in text."""
        return sum(self._count_syllables_word(word) for word in text.split())

    def _count_syllables_word(self, word: str) -> int:
        """
        Count syllables in a word (simplified approximation).

        Not perfect, but good enough for readability scoring.
        """
        word = word.lower().strip()
        if len(word) == 0:
            return 0

        # Remove non-letters
        word = re.sub(r'[^a-z]', '', word)
        if len(word) == 0:
            return 0

        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1

        # Every word has at least one syllable
        return max(1, syllable_count)

    def validate_explanation_quality(
        self,
        explanation: str,
        target_level: Literal["simple", "intermediate", "technical"]
    ) -> Dict[str, any]:
        """
        Validate that explanation meets quality standards for target level.

        Args:
            explanation: The explanation text
            target_level: The intended audience level

        Returns:
            Dictionary with validation results
        """
        scores = self.calculate_readability_score(explanation)

        # Define target ranges for each level
        targets = {
            "simple": {
                "flesch_reading_ease": (60, 100),  # Easy to very easy
                "flesch_kincaid_grade": (0, 8),    # Elementary to middle school
                "max_words_per_sentence": 15
            },
            "intermediate": {
                "flesch_reading_ease": (40, 70),   # Moderate
                "flesch_kincaid_grade": (8, 12),   # Middle to high school
                "max_words_per_sentence": 20
            },
            "technical": {
                "flesch_reading_ease": (0, 50),    # Difficult
                "flesch_kincaid_grade": (12, 20),  # College+
                "max_words_per_sentence": 30
            }
        }

        target = targets[target_level]

        # Check if metrics are in range
        flesch_ok = target["flesch_reading_ease"][0] <= scores["flesch_reading_ease"] <= target["flesch_reading_ease"][1]
        grade_ok = target["flesch_kincaid_grade"][0] <= scores["flesch_kincaid_grade"] <= target["flesch_kincaid_grade"][1]
        sentence_length_ok = scores["words_per_sentence"] <= target["max_words_per_sentence"]

        passed = flesch_ok and grade_ok and sentence_length_ok

        return {
            "passed": passed,
            "target_level": target_level,
            "scores": scores,
            "checks": {
                "flesch_reading_ease": flesch_ok,
                "grade_level": grade_ok,
                "sentence_length": sentence_length_ok
            },
            "recommendations": self._generate_recommendations(scores, target_level, targets[target_level])
        }

    def _generate_recommendations(
        self,
        scores: Dict,
        target_level: str,
        target_ranges: Dict
    ) -> List[str]:
        """Generate recommendations for improving explanation."""
        recommendations = []

        if scores["flesch_reading_ease"] < target_ranges["flesch_reading_ease"][0]:
            recommendations.append("Text is too difficult. Use simpler words and shorter sentences.")
        elif scores["flesch_reading_ease"] > target_ranges["flesch_reading_ease"][1]:
            recommendations.append("Text may be too simple for target audience.")

        if scores["words_per_sentence"] > target_ranges["max_words_per_sentence"]:
            recommendations.append(f"Sentences are too long (avg {scores['words_per_sentence']:.1f} words). Break into shorter sentences.")

        if scores["syllables_per_word"] > 2.0 and target_level == "simple":
            recommendations.append("Use simpler vocabulary with fewer syllables.")

        if not recommendations:
            recommendations.append("Explanation meets quality standards.")

        return recommendations

    def generate_counterfactual_explanation(
        self,
        outcome: str,
        intervention: Dict[str, float],
        point_estimate: float,
        ci_lower: float,
        ci_upper: float,
        uncertainty_level: str,
        robustness_level: str,
    ) -> "ExplanationMetadata":
        """
        Generate explanation for counterfactual analysis.

        Args:
            outcome: The outcome variable name
            intervention: Dict of intervention variable names to values
            point_estimate: Point estimate of the counterfactual outcome
            ci_lower: Lower bound of confidence interval
            ci_upper: Upper bound of confidence interval
            uncertainty_level: Level of uncertainty (e.g., "low", "medium", "high")
            robustness_level: Robustness score level (e.g., "high", "medium", "low")

        Returns:
            ExplanationMetadata object with structured explanation
        """
        from src.models.shared import ExplanationMetadata

        # Format intervention as readable string
        intervention_str = ", ".join(
            f"{var}={val}" for var, val in intervention.items()
        )

        # Build summary
        summary = f"Predicted {outcome} = {point_estimate:.2f} when setting {intervention_str}"

        # Build reasoning
        reasoning_parts = [
            f"Setting {intervention_str} yields a predicted {outcome} of {point_estimate:.2f}.",
            f"The 95% confidence interval is [{ci_lower:.2f}, {ci_upper:.2f}].",
        ]

        if uncertainty_level == "low":
            reasoning_parts.append("The prediction has low uncertainty, indicating high confidence.")
        elif uncertainty_level == "high":
            reasoning_parts.append("Note: This prediction has high uncertainty; interpret with caution.")

        if robustness_level == "high":
            reasoning_parts.append("The result is robust to assumption violations.")
        elif robustness_level == "low":
            reasoning_parts.append("Warning: The result is sensitive to assumptions.")

        reasoning = " ".join(reasoning_parts)

        # Build technical basis
        technical_basis = (
            f"Monte Carlo simulation with intervention do({intervention_str}). "
            f"Point estimate: {point_estimate:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]. "
            f"Uncertainty level: {uncertainty_level}, Robustness: {robustness_level}."
        )

        # Key assumptions
        assumptions = [
            "Structural causal model correctly specified",
            "No unmeasured confounding",
            "Intervention feasibility assumed",
        ]

        # Simple explanation for non-technical audience
        simple_explanation = f"If we set {intervention_str}, we predict {outcome} would be about {point_estimate:.0f}."

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis=technical_basis,
            assumptions=assumptions,
            simple_explanation=simple_explanation,
            learn_more_url="https://docs.inference-service-layer.com/docs/methods/counterfactuals",
            visual_type="uncertainty_plot",
        )

    def generate_causal_validation_explanation(
        self,
        status: str,
        treatment: str,
        outcome: str,
        adjustment_sets: Optional[List[List[str]]] = None,
        minimal_set: Optional[List[str]] = None,
        issues: Optional[List[Dict]] = None,
    ) -> "ExplanationMetadata":
        """
        Generate explanation for causal validation results.

        Args:
            status: Validation status ("identifiable", "cannot_identify", "degraded")
            treatment: Treatment variable name
            outcome: Outcome variable name
            adjustment_sets: List of valid adjustment sets (for identifiable)
            minimal_set: Minimal adjustment set (for identifiable)
            issues: List of validation issues (for cannot_identify)

        Returns:
            ExplanationMetadata object with structured explanation
        """
        from src.models.shared import ExplanationMetadata

        if status == "identifiable":
            minimal_str = ", ".join(minimal_set) if minimal_set else "none"
            summary = f"Effect of {treatment} on {outcome} is identifiable by controlling for {minimal_str}"

            reasoning = (
                f"The causal effect of {treatment} on {outcome} can be identified. "
                f"By controlling for the variables {{{minimal_str}}}, we can block all confounding paths "
                f"and isolate the true causal effect."
            )

            technical_basis = (
                f"Backdoor criterion satisfied with adjustment set {{{minimal_str}}}. "
                f"Found {len(adjustment_sets or [])} valid adjustment sets."
            )

            assumptions = [
                "No unmeasured confounding given the specified adjustment set",
                "Causal graph correctly specified",
                "Positivity: all covariate combinations have non-zero probability",
            ]

            simple_explanation = f"We can estimate this causal effect by controlling for {minimal_str}."

            return ExplanationMetadata(
                summary=summary,
                reasoning=reasoning,
                technical_basis=technical_basis,
                assumptions=assumptions,
                simple_explanation=simple_explanation,
                learn_more_url="https://docs.inference-service-layer.com/docs/methods/backdoor-adjustment",
                visual_type="dag_with_adjustment",
            )

        elif status == "cannot_identify":
            summary = f"Effect of {treatment} on {outcome} cannot be identified"

            if issues:
                issue_descriptions = [issue.get("description", "Unknown issue") for issue in issues]
                issues_text = "; ".join(issue_descriptions)
            else:
                issues_text = "Unblocked confounding paths exist"

            reasoning = (
                f"The causal effect of {treatment} on {outcome} cannot be reliably estimated. "
                f"Issues found: {issues_text}. "
                "Consider collecting additional data or revising the causal model."
            )

            technical_basis = (
                f"No valid adjustment set exists to block all backdoor paths from {treatment} to {outcome}. "
                "Graph analysis indicates unmeasured confounding or structural issues."
            )

            assumptions = [
                "Current graph structure is correctly specified",
                "Listed variables are the only relevant factors",
            ]

            simple_explanation = "We cannot reliably estimate this causal effect due to confounding."

            return ExplanationMetadata(
                summary=summary,
                reasoning=reasoning,
                technical_basis=technical_basis,
                assumptions=assumptions,
                simple_explanation=simple_explanation,
                learn_more_url="https://docs.inference-service-layer.com/docs/concepts/identification",
                visual_type="path_diagram",
            )

        else:  # degraded
            summary = f"Analysis of {treatment} → {outcome} completed with degraded accuracy"

            reasoning = (
                f"The analysis for the effect of {treatment} on {outcome} was completed, "
                "but encountered issues that may affect reliability. "
                "Results should be interpreted with caution."
            )

            technical_basis = (
                "Primary analysis method encountered errors. "
                "Fallback assessment was used to provide partial results."
            )

            assumptions = [
                "Fallback assessment may miss some edge cases",
                "Results are approximate and should be verified",
            ]

            simple_explanation = "We completed the analysis but with reduced confidence in the results."

            return ExplanationMetadata(
                summary=summary,
                reasoning=reasoning,
                technical_basis=technical_basis,
                assumptions=assumptions,
                simple_explanation=simple_explanation,
                learn_more_url="https://docs.inference-service-layer.com/docs/concepts/identification",
                visual_type="dag_with_issues",
            )
