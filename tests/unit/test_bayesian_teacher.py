"""
Unit tests for BayesianTeacher service.

Tests Bayesian teaching implementation for pedagogically optimal examples.
"""

import pytest

from src.models.phase1_models import DecisionContext, UserBeliefModel
from src.models.shared import Distribution, DistributionType
from src.services.bayesian_teacher import BayesianTeacher


@pytest.fixture
def bayesian_teacher():
    """Bayesian teacher instance."""
    return BayesianTeacher()


@pytest.fixture
def sample_beliefs():
    """Sample user belief model."""
    return UserBeliefModel(
        value_weights={
            "revenue": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.7, "std": 0.2},
            ),
            "churn": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.4, "std": 0.2},
            ),
            "brand": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.3, "std": 0.2},
            ),
        },
        risk_tolerance=Distribution(
            type=DistributionType.BETA,
            parameters={"alpha": 2, "beta": 2},
        ),
        time_horizon=Distribution(
            type=DistributionType.NORMAL,
            parameters={"mean": 12, "std": 3},
        ),
        uncertainty_estimates={
            "revenue_weight": 0.3,
            "churn_weight": 0.4,
            "brand_weight": 0.5,
        },
    )


@pytest.fixture
def pricing_context():
    """Pricing decision context."""
    return DecisionContext(
        domain="pricing",
        variables=["revenue", "churn", "brand"],
        constraints={"industry": "SaaS", "current_price": 49},
    )


def test_generate_teaching_examples_confounding(
    bayesian_teacher, sample_beliefs, pricing_context
):
    """Test generating teaching examples for confounding."""
    examples, explanation, objectives, time = bayesian_teacher.generate_teaching_examples(
        target_concept="confounding",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=3,
    )

    # Should return requested number of examples
    assert len(examples) <= 3
    assert len(examples) > 0

    # Each example should have required fields
    for example in examples:
        assert example.scenario is not None
        assert len(example.key_insight) > 0
        assert len(example.why_this_example) > 0
        assert example.information_value >= 0

    # Should have explanation
    assert len(explanation) > 0
    assert "confounding" in explanation.lower()

    # Should have objectives
    assert len(objectives) > 0

    # Should have time estimate
    assert "minute" in time.lower()


def test_generate_teaching_examples_tradeoffs(
    bayesian_teacher, sample_beliefs, pricing_context
):
    """Test generating teaching examples for trade-offs."""
    examples, explanation, objectives, time = bayesian_teacher.generate_teaching_examples(
        target_concept="trade_offs",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=2,
    )

    assert len(examples) <= 2
    assert len(examples) > 0

    # Examples should mention trade-offs
    for example in examples:
        assert len(example.scenario.trade_offs) > 0

    assert "trade" in explanation.lower() or "competing" in explanation.lower()


def test_generate_teaching_examples_causal(
    bayesian_teacher, sample_beliefs, pricing_context
):
    """Test generating teaching examples for causal mechanisms."""
    examples, explanation, objectives, time = bayesian_teacher.generate_teaching_examples(
        target_concept="causal_mechanism",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=2,
    )

    assert len(examples) <= 2
    assert len(examples) > 0

    # Should involve causal relationships
    for example in examples:
        assert example.scenario is not None


def test_generate_teaching_examples_uncertainty(
    bayesian_teacher, sample_beliefs, pricing_context
):
    """Test generating teaching examples for uncertainty."""
    examples, explanation, objectives, time = bayesian_teacher.generate_teaching_examples(
        target_concept="uncertainty",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=2,
    )

    assert len(examples) <= 2
    assert len(examples) > 0


def test_generate_teaching_examples_optimization(
    bayesian_teacher, sample_beliefs, pricing_context
):
    """Test generating teaching examples for optimization."""
    examples, explanation, objectives, time = bayesian_teacher.generate_teaching_examples(
        target_concept="optimization",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=2,
    )

    assert len(examples) <= 2


def test_generate_teaching_examples_unknown_concept(
    bayesian_teacher, sample_beliefs, pricing_context
):
    """Test handling of unknown concept."""
    examples, explanation, objectives, time = bayesian_teacher.generate_teaching_examples(
        target_concept="unknown_concept",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=2,
    )

    # Should default to trade_offs
    assert len(examples) > 0


def test_generate_teaching_examples_deterministic(
    bayesian_teacher, sample_beliefs, pricing_context
):
    """Test that teaching example generation is deterministic."""
    examples1, _, _, _ = bayesian_teacher.generate_teaching_examples(
        target_concept="trade_offs",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=3,
    )

    examples2, _, _, _ = bayesian_teacher.generate_teaching_examples(
        target_concept="trade_offs",
        current_beliefs=sample_beliefs,
        context=pricing_context,
        max_examples=3,
    )

    # Should generate same examples
    assert len(examples1) == len(examples2)
    for ex1, ex2 in zip(examples1, examples2):
        assert ex1.information_value == ex2.information_value


def test_compute_teaching_value(bayesian_teacher, sample_beliefs):
    """Test computing teaching value of examples."""
    from src.models.phase1_models import Scenario, TeachingExample

    example = TeachingExample(
        scenario=Scenario(
            description="Test scenario",
            outcomes={"revenue": 80, "churn": 0.02},
            trade_offs=["High revenue", "Low churn"],
        ),
        key_insight="Test insight",
        why_this_example="Test reason",
        information_value=0.0,
    )

    value = bayesian_teacher._compute_teaching_value(
        example, "trade_offs", sample_beliefs
    )

    # Should return a value between 0 and 1
    assert 0 <= value <= 1


def test_compute_novelty(bayesian_teacher, sample_beliefs):
    """Test novelty computation."""
    from src.models.phase1_models import Scenario, TeachingExample

    # Novel example (high revenue, different from expected)
    novel_example = TeachingExample(
        scenario=Scenario(
            description="Novel",
            outcomes={"revenue": 95, "churn": 0.01},
            trade_offs=[],
        ),
        key_insight="",
        why_this_example="",
        information_value=0.0,
    )

    novelty = bayesian_teacher._compute_novelty(novel_example, sample_beliefs)
    assert novelty >= 0


def test_compute_clarity(bayesian_teacher):
    """Test clarity computation."""
    from src.models.phase1_models import Scenario, TeachingExample

    # Simple example (few variables, clear trade-offs)
    simple_example = TeachingExample(
        scenario=Scenario(
            description="Simple",
            outcomes={"var1": 50, "var2": 60},
            trade_offs=["Clear trade-off 1", "Clear trade-off 2"],
        ),
        key_insight="",
        why_this_example="",
        information_value=0.0,
    )

    clarity = bayesian_teacher._compute_clarity(simple_example, "trade_offs")
    assert 0 <= clarity <= 1


def test_compute_relevance(bayesian_teacher, sample_beliefs):
    """Test relevance computation."""
    from src.models.phase1_models import Scenario, TeachingExample

    # Relevant example (involves variables user cares about)
    relevant_example = TeachingExample(
        scenario=Scenario(
            description="Relevant",
            outcomes={"revenue": 70, "churn": 0.05},  # User cares about these
            trade_offs=[],
        ),
        key_insight="",
        why_this_example="",
        information_value=0.0,
    )

    relevance = bayesian_teacher._compute_relevance(relevant_example, sample_beliefs)
    assert 0 <= relevance <= 1


def test_rank_by_teaching_value(bayesian_teacher, sample_beliefs):
    """Test ranking examples by teaching value."""
    from src.models.phase1_models import Scenario, TeachingExample

    examples = [
        TeachingExample(
            scenario=Scenario(
                description=f"Example {i}",
                outcomes={"revenue": 50 + i * 10, "churn": 0.05},
                trade_offs=["Trade-off"],
            ),
            key_insight=f"Insight {i}",
            why_this_example=f"Reason {i}",
            information_value=0.0,
        )
        for i in range(5)
    ]

    ranked = bayesian_teacher._rank_by_teaching_value(
        examples, "trade_offs", sample_beliefs
    )

    # Should return all examples
    assert len(ranked) == 5

    # Should be sorted by teaching value (descending)
    values = [ex.information_value for ex in ranked]
    assert values == sorted(values, reverse=True)


def test_generate_confounding_examples(bayesian_teacher, pricing_context, sample_beliefs):
    """Test confounding example generation."""
    examples = bayesian_teacher._generate_confounding_examples(
        pricing_context, sample_beliefs
    )

    assert len(examples) > 0

    # Should mention confounding concepts
    for example in examples:
        insight_lower = example.key_insight.lower()
        assert (
            "confound" in insight_lower
            or "correlation" in insight_lower
            or "causation" in insight_lower
            or "control" in insight_lower
        )


def test_generate_tradeoff_examples(bayesian_teacher, pricing_context, sample_beliefs):
    """Test trade-off example generation."""
    examples = bayesian_teacher._generate_tradeoff_examples(
        pricing_context, sample_beliefs
    )

    assert len(examples) > 0

    # Should have trade-offs
    for example in examples:
        assert len(example.scenario.trade_offs) > 0


def test_learning_objectives(bayesian_teacher):
    """Test learning objective generation."""
    from src.models.phase1_models import Scenario, TeachingExample

    examples = [
        TeachingExample(
            scenario=Scenario(
                description="Example",
                outcomes={"var": 50},
                trade_offs=[],
            ),
            key_insight="Key insight here",
            why_this_example="Reason",
            information_value=0.5,
        )
    ]

    objectives = bayesian_teacher._define_learning_objectives("trade_offs", examples)

    # Should have multiple objectives
    assert len(objectives) > 1

    # Should include concept understanding
    assert any("trade_offs" in obj.lower() for obj in objectives)


def test_estimate_learning_time(bayesian_teacher):
    """Test learning time estimation."""
    time_2 = bayesian_teacher._estimate_learning_time("trade_offs", 2)
    time_5 = bayesian_teacher._estimate_learning_time("trade_offs", 5)

    # Should return time strings
    assert "minute" in time_2.lower()
    assert "minute" in time_5.lower()

    # More examples should take more time (roughly)
    # Extract numbers from strings for comparison
    import re

    nums_2 = [int(n) for n in re.findall(r"\d+", time_2)]
    nums_5 = [int(n) for n in re.findall(r"\d+", time_5)]

    if nums_2 and nums_5:
        assert max(nums_5) >= max(nums_2)


def test_teaching_explanation_generation(bayesian_teacher, sample_beliefs):
    """Test teaching explanation generation."""
    from src.models.phase1_models import Scenario, TeachingExample

    examples = [
        TeachingExample(
            scenario=Scenario(description="", outcomes={}, trade_offs=[]),
            key_insight="",
            why_this_example="",
            information_value=0.5,
        )
    ]

    explanation = bayesian_teacher._generate_teaching_explanation(
        "trade_offs", examples, sample_beliefs
    )

    assert len(explanation) > 0
    # Should mention the concept
    assert "trade" in explanation.lower() or "competing" in explanation.lower()
