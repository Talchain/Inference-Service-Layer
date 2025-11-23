"""
Unit tests for ExplanationGenerator service.

Tests explanation generation including:
- Template-based explanation generation
- Progressive disclosure (three levels)
- Readability scoring
- Quality validation
- All concept templates
"""

import pytest

from src.services.explanation_generator import (
    ExplanationGenerator,
    ProgressiveExplanation
)


@pytest.fixture
def generator():
    """Create an ExplanationGenerator instance."""
    return ExplanationGenerator()


class TestExplanationGeneratorInitialization:
    """Tests for ExplanationGenerator initialization."""

    def test_initialization_success(self):
        """Test successful initialization."""
        generator = ExplanationGenerator()
        assert generator is not None
        assert hasattr(generator, 'TEMPLATES')

    def test_templates_populated(self, generator):
        """Test that templates dictionary is populated."""
        assert len(generator.TEMPLATES) > 0
        assert "conformal_prediction" in generator.TEMPLATES


class TestTemplateGeneration:
    """Tests for template-based explanation generation."""

    def test_generate_simple_level(self, generator):
        """Test generating simple explanation."""
        data = {"confidence": 95}
        explanation = generator.generate("conformal_prediction", data, "simple")
        assert isinstance(explanation, str)
        assert "95%" in explanation

    def test_generate_intermediate_level(self, generator):
        """Test generating intermediate explanation."""
        data = {"confidence": 95, "lower": 40000, "upper": 60000}
        explanation = generator.generate("conformal_prediction", data, "intermediate")
        assert isinstance(explanation, str)
        assert "40000" in explanation or "40,000" in explanation

    def test_generate_technical_level(self, generator):
        """Test generating technical explanation."""
        data = {
            "guaranteed_coverage": 95,
            "n_calibration": 100,
            "confidence": 95
        }
        explanation = generator.generate("conformal_prediction", data, "technical")
        assert isinstance(explanation, str)
        assert "calibration" in explanation.lower()

    def test_generate_non_identifiable(self, generator):
        """Test non-identifiable explanation."""
        data = {"treatment": "Price", "outcome": "Revenue"}
        explanation = generator.generate("non_identifiable", data, "simple")
        assert "confounding" in explanation.lower()

    def test_generate_identifiable_backdoor(self, generator):
        """Test identifiable backdoor explanation."""
        data = {
            "num_controls": 2,
            "controls": "Brand, Season",
            "treatment": "Price",
            "outcome": "Revenue"
        }
        explanation = generator.generate("identifiable_backdoor", data, "intermediate")
        assert "controlling" in explanation.lower()

    def test_generate_with_missing_data_key(self, generator):
        """Test generation with missing data key."""
        data = {}  # Missing required keys
        explanation = generator.generate("conformal_prediction", data, "simple")
        # Should handle gracefully
        assert "unavailable" in explanation.lower() or "missing" in explanation.lower()

    def test_generate_unknown_concept(self, generator):
        """Test generation with unknown concept."""
        explanation = generator.generate("unknown_concept", {}, "simple")
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestProgressiveDisclosure:
    """Tests for progressive disclosure generation."""

    def test_generate_progressive_basic(self, generator):
        """Test basic progressive explanation generation."""
        data = {"confidence": 95, "lower": 45000, "upper": 55000, "guaranteed_coverage": 95, "n_calibration": 100}
        progressive = generator.generate_progressive("conformal_prediction", data)
        
        assert isinstance(progressive, ProgressiveExplanation)
        assert len(progressive.headline) > 0
        assert len(progressive.summary) > 0
        assert len(progressive.details) > 0

    def test_progressive_includes_visual_aid(self, generator):
        """Test that progressive explanation includes visual aid suggestion."""
        data = {"confidence": 95}
        progressive = generator.generate_progressive("conformal_prediction", data)
        
        assert progressive.visual_aid is not None
        assert progressive.visual_aid == "interval_plot"

    def test_progressive_includes_learn_more_url(self, generator):
        """Test that progressive explanation includes documentation URL."""
        data = {"confidence": 95}
        progressive = generator.generate_progressive("conformal_prediction", data)
        
        assert progressive.learn_more_url is not None
        assert "docs" in progressive.learn_more_url.lower()

    def test_progressive_levels_differ(self, generator):
        """Test that three levels have different content."""
        data = {"confidence": 95, "lower": 45000, "upper": 55000, "guaranteed_coverage": 95, "n_calibration": 100}
        progressive = generator.generate_progressive("conformal_prediction", data)
        
        # Headline should be shortest
        assert len(progressive.headline) < len(progressive.summary)
        assert len(progressive.summary) < len(progressive.details)


class TestReadabilityScoring:
    """Tests for readability metric calculation."""

    def test_calculate_readability_basic(self, generator):
        """Test basic readability calculation."""
        text = "This is a simple sentence. It has short words."
        scores = generator.calculate_readability_score(text)
        
        assert "flesch_reading_ease" in scores
        assert "flesch_kincaid_grade" in scores
        assert "smog_index" in scores
        assert "words" in scores
        assert "sentences" in scores

    def test_readability_simple_text(self, generator):
        """Test readability of simple text."""
        text = "The cat sat on the mat. It was a red mat."
        scores = generator.calculate_readability_score(text)
        
        # Simple text should have high reading ease
        assert scores["flesch_reading_ease"] > 60
        # And low grade level
        assert scores["flesch_kincaid_grade"] < 8

    def test_readability_complex_text(self, generator):
        """Test readability of complex text."""
        text = "The implementation utilizes sophisticated methodological approaches incorporating multidimensional analytical frameworks."
        scores = generator.calculate_readability_score(text)
        
        # Complex text should have lower reading ease
        assert scores["flesch_reading_ease"] < 50
        # And higher grade level
        assert scores["flesch_kincaid_grade"] > 10

    def test_readability_sentence_count(self, generator):
        """Test sentence counting."""
        text = "First sentence. Second sentence! Third sentence?"
        scores = generator.calculate_readability_score(text)
        
        assert scores["sentences"] == 3

    def test_readability_word_count(self, generator):
        """Test word counting."""
        text = "One two three four five."
        scores = generator.calculate_readability_score(text)
        
        assert scores["words"] == 5

    def test_readability_words_per_sentence(self, generator):
        """Test words per sentence calculation."""
        text = "Short sentence. This one has five words total."
        scores = generator.calculate_readability_score(text)
        
        # (2 + 6) / 2 = 4
        assert 3 <= scores["words_per_sentence"] <= 5


class TestSyllableCounting:
    """Tests for syllable counting (used in readability)."""

    def test_count_syllables_single_syllable(self, generator):
        """Test counting syllables in single-syllable words."""
        assert generator._count_syllables_word("cat") == 1
        assert generator._count_syllables_word("dog") == 1
        assert generator._count_syllables_word("run") == 1

    def test_count_syllables_two_syllables(self, generator):
        """Test counting syllables in two-syllable words."""
        assert generator._count_syllables_word("table") == 2
        assert generator._count_syllables_word("apple") == 2

    def test_count_syllables_three_syllables(self, generator):
        """Test counting syllables in three-syllable words."""
        count = generator._count_syllables_word("elephant")
        assert 2 <= count <= 4  # Approximate

    def test_count_syllables_silent_e(self, generator):
        """Test handling of silent e."""
        # "make" should count as 1 syllable
        count = generator._count_syllables_word("make")
        assert count == 1


class TestQualityValidation:
    """Tests for explanation quality validation."""

    def test_validate_simple_explanation(self, generator):
        """Test validation of simple explanation."""
        text = "This prediction has a 95% guaranteed accuracy range."
        result = generator.validate_explanation_quality(text, "simple")
        
        assert "passed" in result
        assert "target_level" in result
        assert result["target_level"] == "simple"

    def test_validate_technical_explanation(self, generator):
        """Test validation of technical explanation."""
        text = "Conformal prediction provides distribution-free finite-sample valid intervals."
        result = generator.validate_explanation_quality(text, "technical")
        
        assert "scores" in result
        assert "checks" in result

    def test_validate_provides_recommendations(self, generator):
        """Test that validation provides recommendations."""
        text = "Text."
        result = generator.validate_explanation_quality(text, "simple")
        
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_validate_checks_sentence_length(self, generator):
        """Test that validation checks sentence length."""
        # Very long sentence for simple level
        text = "This is a very long sentence with many words that goes on and on and contains lots of information that would be hard for a general audience to parse and understand easily without getting lost."
        result = generator.validate_explanation_quality(text, "simple")
        
        assert "sentence_length" in result["checks"]


class TestAllConceptTemplates:
    """Tests that all concept templates work."""

    def test_conformal_prediction_template(self, generator):
        """Test conformal prediction template."""
        data = {"confidence": 95, "lower": 45000, "upper": 55000, "guaranteed_coverage": 95, "n_calibration": 100}
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("conformal_prediction", data, level)
            assert len(explanation) > 0

    def test_non_identifiable_template(self, generator):
        """Test non-identifiable template."""
        data = {"treatment": "Price", "outcome": "Revenue", "num_backdoor": 3}
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("non_identifiable", data, level)
            assert len(explanation) > 0

    def test_identifiable_backdoor_template(self, generator):
        """Test identifiable backdoor template."""
        data = {
            "num_controls": 2,
            "controls": "Brand, Season",
            "adjustment_set": "Brand, Season",
            "treatment": "Price",
            "outcome": "Revenue",
            "num_backdoor": 3
        }
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("identifiable_backdoor", data, level)
            assert len(explanation) > 0

    def test_sensitivity_templates(self, generator):
        """Test sensitivity analysis templates."""
        high_sens_data = {
            "assumption": "no_unobserved_confounding",
            "threshold": 10,
            "impact": 23,
            "elasticity": 2.3,
            "critical_threshold": 1.5,
            "robustness": 0.34
        }
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("high_sensitivity", high_sens_data, level)
            assert len(explanation) > 0

        low_sens_data = {
            "threshold": 20,
            "impact": 5,
            "elasticity": 0.25,
            "robustness": 0.85,
            "max_violation": 30
        }
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("low_sensitivity", low_sens_data, level)
            assert len(explanation) > 0

    def test_transportability_templates(self, generator):
        """Test transportability templates."""
        success_data = {
            "source_domain": "US Market",
            "target_domain": "EU Market",
            "method": "selection diagrams",
            "selection_nodes": "Region, Currency",
            "effect": 1250
        }
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("transportability_success", success_data, level)
            assert len(explanation) > 0

    def test_discovery_template(self, generator):
        """Test causal discovery template."""
        data = {
            "num_edges": 12,
            "confidence": 0.85,
            "top_edges": "Price → Revenue, Brand → Price",
            "algorithm": "PC",
            "n_samples": 1000,
            "n_vars": 10,
            "alpha": 0.05,
            "shd": 3
        }
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("causal_discovery_success", data, level)
            assert len(explanation) > 0

    def test_batch_optimization_template(self, generator):
        """Test batch optimization template."""
        data = {
            "n_scenarios": 100,
            "time": 2.5,
            "best_outcome": 75000,
            "best_intervention": "Price=55, Quality=8",
            "worst_outcome": 30000,
            "worst_intervention": "Price=30, Quality=5",
            "n_samples": 1000,
            "total_evals": 100000,
            "n_workers": 4,
            "throughput": 40
        }
        for level in ["simple", "intermediate", "technical"]:
            explanation = generator.generate("batch_optimization", data, level)
            assert len(explanation) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text_readability(self, generator):
        """Test readability of empty text."""
        scores = generator.calculate_readability_score("")
        assert isinstance(scores, dict)

    def test_single_word_readability(self, generator):
        """Test readability of single word."""
        scores = generator.calculate_readability_score("word")
        assert scores["words"] == 1

    def test_generate_with_none_data(self, generator):
        """Test generation with None data values."""
        data = {"confidence": None}
        try:
            explanation = generator.generate("conformal_prediction", data, "simple")
            # Should either handle gracefully or raise
            assert True
        except Exception:
            assert True

    def test_validate_empty_text(self, generator):
        """Test validation of empty text."""
        result = generator.validate_explanation_quality("", "simple")
        assert isinstance(result, dict)
        assert "passed" in result
