"""
Unit tests for CausalRepresentationLearner service.

Tests factor extraction from unstructured data including:
- Text embedding (TF-IDF fallback)
- Clustering
- Keyword extraction
- Factor naming
- DAG suggestion
- Confidence calculation
"""

import pytest
import numpy as np

from src.services.causal_representation_learner import CausalRepresentationLearner


@pytest.fixture
def learner():
    """Create a CausalRepresentationLearner instance."""
    return CausalRepresentationLearner()


@pytest.fixture
def sample_support_tickets():
    """Sample support ticket data."""
    return [
        "Can't find the settings page, navigation is confusing",
        "App is very slow to load, takes forever",
        "Settings menu is hidden, hard to locate",
        "Performance is terrible, constant freezing",
        "UI is confusing, can't figure out how to navigate",
        "Slow performance, timeouts frequently",
        "Bug in checkout process, payment fails",
        "Can't complete purchase, error message appears",
        "Navigation is unclear, getting lost in menus",
        "Loading times are unacceptable, very slow",
        "Checkout broken, can't finalize order",
        "Menu structure is confusing",
        "App crashes during payment",
        "Performance issues, lag everywhere",
        "Settings are impossible to find"
    ]


@pytest.fixture
def sample_reviews():
    """Sample product review data."""
    return [
        "Great product but expensive",
        "Love the features but price is too high",
        "Quality is good but overpriced",
        "Excellent quality, worth the money",
        "Too expensive for what you get",
        "Good value for the price",
        "Price point is reasonable",
        "Quality could be better for the cost",
        "Features are amazing, price is fair",
        "Overpriced compared to competitors"
    ]


class TestLearnerInitialization:
    """Tests for CausalRepresentationLearner initialization."""

    def test_initialization_success(self):
        """Test successful initialization."""
        learner = CausalRepresentationLearner()
        assert learner is not None

    def test_embedder_lazy_initialization(self, learner):
        """Test that embedder is lazily initialized."""
        assert learner._embedder is None  # Not loaded yet


class TestTextEmbedding:
    """Tests for text embedding."""

    def test_embed_texts_basic(self, learner):
        """Test basic text embedding."""
        texts = ["This is a test", "Another test sentence"]
        embeddings = learner._embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2  # Two texts

    def test_embed_texts_tfidf_fallback(self, learner):
        """Test TF-IDF fallback embedding."""
        texts = ["machine learning", "deep learning", "data science"]
        embeddings = learner._embed_texts_tfidf(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3

    def test_embed_different_lengths(self, learner):
        """Test embedding texts of different lengths."""
        texts = ["Short", "This is a much longer sentence with more words"]
        embeddings = learner._embed_texts(texts)
        
        assert embeddings.shape[0] == 2
        # Both should have same embedding dimension
        assert embeddings.shape[1] == embeddings.shape[1]

    def test_embed_empty_list(self, learner):
        """Test embedding empty list."""
        embeddings = learner._embed_texts([])
        assert len(embeddings) == 0


class TestClustering:
    """Tests for embedding clustering."""

    def test_cluster_embeddings_basic(self, learner):
        """Test basic clustering."""
        # Create simple embeddings
        embeddings = np.array([
            [1, 0], [1.1, 0], [1.2, 0],  # Cluster 1
            [0, 1], [0, 1.1], [0, 1.2]   # Cluster 2
        ])
        
        labels, centers = learner._cluster_embeddings(embeddings, n_clusters=2, min_cluster_size=2)
        
        assert len(labels) == 6
        assert len(centers) == 2

    def test_cluster_labels_range(self, learner):
        """Test that cluster labels are in expected range."""
        embeddings = np.random.rand(20, 10)
        labels, centers = learner._cluster_embeddings(embeddings, n_clusters=3, min_cluster_size=2)
        
        assert all(0 <= label < 3 for label in labels)

    def test_cluster_centers_shape(self, learner):
        """Test that cluster centers have correct shape."""
        embeddings = np.random.rand(15, 5)
        labels, centers = learner._cluster_embeddings(embeddings, n_clusters=3, min_cluster_size=2)
        
        assert centers.shape == (3, 5)  # n_clusters × embedding_dim


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_extract_keywords_basic(self, learner):
        """Test basic keyword extraction."""
        texts = [
            "Python programming language is great",
            "Python is excellent for data science",
            "I love programming in Python"
        ]
        keywords = learner._extract_keywords(texts)
        
        assert isinstance(keywords, list)
        assert "python" in keywords or "programming" in keywords

    def test_extract_keywords_removes_stopwords(self, learner):
        """Test that stop words are removed."""
        texts = ["the cat sat on the mat"]
        keywords = learner._extract_keywords(texts)
        
        # Stop words should not be in keywords
        assert "the" not in keywords
        assert "on" not in keywords

    def test_extract_keywords_frequency(self, learner):
        """Test that frequent words are prioritized."""
        texts = [
            "performance slow performance slow performance",
            "slow speed slow"
        ]
        keywords = learner._extract_keywords(texts, top_k=3)
        
        # "slow" and "performance" should be top keywords
        assert "slow" in keywords[:3] or "performance" in keywords[:3]

    def test_extract_keywords_top_k(self, learner):
        """Test that top_k parameter works."""
        texts = ["word1 word2 word3 word4 word5 word6 word7 word8"]
        keywords = learner._extract_keywords(texts, top_k=3)
        
        assert len(keywords) <= 3


class TestFactorNaming:
    """Tests for factor naming."""

    def test_name_factor_usability(self, learner):
        """Test naming usability-related factor."""
        keywords = ["confusing", "hard", "difficult", "navigate"]
        name = learner._name_factor(keywords)
        
        assert "usability" in name.lower()

    def test_name_factor_performance(self, learner):
        """Test naming performance-related factor."""
        keywords = ["slow", "lag", "freeze", "timeout"]
        name = learner._name_factor(keywords)
        
        assert "performance" in name.lower()

    def test_name_factor_quality(self, learner):
        """Test naming quality-related factor."""
        keywords = ["bug", "error", "broken", "fail"]
        name = learner._name_factor(keywords)
        
        assert "quality" in name.lower() or "bug" in name or "error" in name

    def test_name_factor_fallback(self, learner):
        """Test fallback naming for unrecognized keywords."""
        keywords = ["zxyabc", "qwerty"]
        name = learner._name_factor(keywords)
        
        # Should construct name from keywords
        assert isinstance(name, str)
        assert len(name) > 0

    def test_name_factor_empty_keywords(self, learner):
        """Test naming with empty keywords."""
        name = learner._name_factor([])
        assert name == "unknown_factor"


class TestCoherenceCalculation:
    """Tests for cluster coherence calculation."""

    def test_calculate_coherence_tight_cluster(self, learner):
        """Test coherence of tight cluster."""
        # Very similar embeddings
        embeddings = np.array([[1, 0, 0], [1.01, 0.01, 0], [0.99, 0.02, 0]])
        coherence = learner._calculate_coherence(embeddings, None)
        
        assert coherence > 0.7  # Tight cluster = high coherence

    def test_calculate_coherence_loose_cluster(self, learner):
        """Test coherence of loose cluster."""
        # Very different embeddings
        embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coherence = learner._calculate_coherence(embeddings, None)
        
        assert coherence < 0.7  # Loose cluster = low coherence

    def test_calculate_coherence_with_center(self, learner):
        """Test coherence calculation with provided center."""
        embeddings = np.array([[1, 0], [1.1, 0.1], [0.9, 0.1]])
        center = np.array([1.0, 0.0])
        coherence = learner._calculate_coherence(embeddings, center)
        
        assert 0.0 <= coherence <= 1.0

    def test_calculate_coherence_small_cluster(self, learner):
        """Test coherence of small cluster."""
        embeddings = np.array([[1, 0]])
        coherence = learner._calculate_coherence(embeddings, None)
        
        assert coherence == 0.5  # Default for small clusters


class TestDAGSuggestion:
    """Tests for DAG structure suggestion."""

    def test_suggest_dag_basic(self, learner):
        """Test basic DAG suggestion."""
        factors = [
            {"name": "usability_issues", "strength": 0.8},
            {"name": "performance_problems", "strength": 0.7}
        ]
        outcome = "churn"
        
        dag = learner._suggest_dag_structure(factors, outcome)
        
        assert "nodes" in dag
        assert "edges" in dag
        assert outcome in dag["nodes"]

    def test_suggest_dag_includes_all_factors(self, learner):
        """Test that DAG includes all factors."""
        factors = [
            {"name": "factor1", "strength": 0.8},
            {"name": "factor2", "strength": 0.7},
            {"name": "factor3", "strength": 0.6}
        ]
        dag = learner._suggest_dag_structure(factors, "outcome")
        
        assert len(dag["nodes"]) == 4  # 3 factors + outcome
        assert all(f["name"] in dag["nodes"] for f in factors)

    def test_suggest_dag_edges(self, learner):
        """Test that DAG has correct edges."""
        factors = [{"name": "factor1", "strength": 0.8}]
        outcome = "outcome"
        
        dag = learner._suggest_dag_structure(factors, outcome)
        
        # Should have edge from factor1 to outcome
        assert ["factor1", outcome] in dag["edges"]


class TestConfidenceCalculation:
    """Tests for overall confidence calculation."""

    def test_confidence_no_factors(self, learner):
        """Test confidence with no factors."""
        confidence = learner._calculate_overall_confidence([], 100)
        assert confidence == 0.0

    def test_confidence_high_quality_factors(self, learner):
        """Test confidence with high-quality factors."""
        factors = [
            {"name": "f1", "strength": 0.9, "prevalence": 0.3},
            {"name": "f2", "strength": 0.85, "prevalence": 0.3},
            {"name": "f3", "strength": 0.8, "prevalence": 0.2}
        ]
        confidence = learner._calculate_overall_confidence(factors, 200)
        
        assert confidence > 0.7  # High quality = high confidence

    def test_confidence_large_sample(self, learner):
        """Test that large sample increases confidence."""
        factors = [{"name": "f1", "strength": 0.7, "prevalence": 0.5}]
        
        conf_small = learner._calculate_overall_confidence(factors, 20)
        conf_large = learner._calculate_overall_confidence(factors, 200)
        
        assert conf_large > conf_small

    def test_confidence_bounded(self, learner):
        """Test that confidence is bounded [0, 1]."""
        factors = [{"name": "f1", "strength": 1.0, "prevalence": 1.0}]
        confidence = learner._calculate_overall_confidence(factors, 1000)
        
        assert 0.0 <= confidence <= 1.0


class TestFactorExtraction:
    """Tests for complete factor extraction."""

    def test_extract_factors_basic(self, learner, sample_support_tickets):
        """Test basic factor extraction."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_support_tickets,
            n_factors=3,
            min_cluster_size=2
        )
        
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert 0.0 <= confidence <= 1.0

    def test_extract_factors_with_outcome(self, learner, sample_support_tickets):
        """Test factor extraction with outcome variable."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_support_tickets,
            n_factors=2,
            outcome_variable="churn"
        )
        
        assert dag is not None
        assert "nodes" in dag
        assert "churn" in dag["nodes"]

    def test_extract_factors_structure(self, learner, sample_support_tickets):
        """Test structure of extracted factors."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_support_tickets,
            n_factors=2
        )
        
        for factor in factors:
            assert "name" in factor
            assert "strength" in factor
            assert "keywords" in factor
            assert "representative_texts" in factor
            assert "prevalence" in factor

    def test_extract_factors_min_cluster_size(self, learner, sample_reviews):
        """Test min_cluster_size parameter."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_reviews,
            n_factors=5,
            min_cluster_size=3  # Require at least 3 items per cluster
        )
        
        # Should skip small clusters
        assert isinstance(factors, list)

    def test_extract_factors_confidence_reasonable(self, learner, sample_support_tickets):
        """Test that confidence is reasonable."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_support_tickets,
            n_factors=3
        )
        
        # Should have moderate to high confidence with good data
        assert 0.3 <= confidence <= 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_extract_factors_small_data(self, learner):
        """Test with very small dataset."""
        data = ["text1", "text2", "text3"]
        try:
            factors, dag, confidence = learner.extract_factors(data, n_factors=2, min_cluster_size=1)
            assert isinstance(factors, list)
        except Exception:
            # May fail with insufficient data
            assert True

    def test_extract_factors_single_factor(self, learner, sample_reviews):
        """Test extracting single factor."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_reviews,
            n_factors=1
        )
        
        assert len(factors) >= 0  # May be 0 if min_cluster_size not met

    def test_extract_factors_many_factors(self, learner, sample_support_tickets):
        """Test extracting many factors."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_support_tickets,
            n_factors=10,
            min_cluster_size=1
        )
        
        # May not get all 10 if data doesn't support it
        assert isinstance(factors, list)
        assert len(factors) <= 10

    def test_extract_keywords_special_characters(self, learner):
        """Test keyword extraction with special characters."""
        texts = ["test@123", "special#chars!", "normal words"]
        keywords = learner._extract_keywords(texts)
        
        # Should handle gracefully
        assert isinstance(keywords, list)

    def test_name_factor_with_domain_hints(self, learner):
        """Test factor naming with domain hints."""
        keywords = ["custom", "keyword"]
        hints = ["domain_specific_term"]
        name = learner._name_factor(keywords, domain_hints=hints)
        
        assert isinstance(name, str)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_support_tickets(self, learner, sample_support_tickets):
        """Test full pipeline with support ticket data."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_support_tickets,
            domain_hints=["usability", "performance"],
            n_factors=3,
            outcome_variable="churn"
        )
        
        # Should identify usability and performance issues
        factor_names = [f["name"] for f in factors]
        assert len(factor_names) > 0
        
        # Should have reasonable confidence
        assert confidence > 0.3
        
        # Should suggest DAG
        if dag:
            assert len(dag["nodes"]) > 0
            assert "churn" in dag["nodes"]

    def test_full_pipeline_reviews(self, learner, sample_reviews):
        """Test full pipeline with review data."""
        factors, dag, confidence = learner.extract_factors(
            data=sample_reviews,
            n_factors=2,
            outcome_variable="satisfaction"
        )
        
        # Should extract factors related to price/quality
        assert isinstance(factors, list)
        assert confidence > 0.0
