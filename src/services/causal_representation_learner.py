"""
Causal representation learning service for extracting factors from unstructured data.

Uses text embedding and clustering to identify latent causal factors.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class CausalRepresentationLearner:
    """
    Extract causal factors from unstructured data.

    Example: Support tickets → ["product_quality", "onboarding_difficulty"]

    Uses:
    1. Text embedding (sentence transformers)
    2. Clustering (K-means or HDBSCAN)
    3. Keyword extraction (TF-IDF)
    4. Factor naming (heuristic)
    5. DAG suggestion (correlation-based)
    """

    def __init__(self) -> None:
        """Initialize the causal representation learner."""
        self.logger = logger
        self._embedder = None

    def extract_factors(
        self,
        data: List[str],
        domain_hints: Optional[List[str]] = None,
        n_factors: int = 5,
        min_cluster_size: int = 3,
        outcome_variable: Optional[str] = None
    ) -> Tuple[List[Dict], Optional[Dict], float]:
        """
        Extract causal factors from unstructured text data.

        Args:
            data: Raw text data
            domain_hints: Optional domain-specific keywords
            n_factors: Number of factors to extract
            min_cluster_size: Minimum cluster size
            outcome_variable: Optional outcome variable

        Returns:
            Tuple of (factors, suggested_dag, confidence)
        """
        self.logger.info(f"Extracting {n_factors} factors from {len(data)} texts")

        # Step 1: Embed texts
        embeddings = self._embed_texts(data)
        self.logger.debug(f"Embeddings shape: {embeddings.shape}")

        # Step 2: Cluster embeddings
        labels, cluster_centers = self._cluster_embeddings(
            embeddings,
            n_clusters=n_factors,
            min_cluster_size=min_cluster_size
        )

        # Step 3: Extract factors from clusters
        factors = []
        for cluster_id in range(n_factors):
            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) < min_cluster_size:
                self.logger.warning(f"Cluster {cluster_id} too small ({len(cluster_indices)} texts), skipping")
                continue

            cluster_texts = [data[i] for i in cluster_indices]

            # Extract keywords
            keywords = self._extract_keywords(cluster_texts)

            # Name the factor
            factor_name = self._name_factor(keywords, domain_hints)

            # Calculate strength (cluster coherence)
            strength = self._calculate_coherence(
                embeddings[cluster_indices],
                cluster_centers[cluster_id] if cluster_id < len(cluster_centers) else None
            )

            # Prevalence
            prevalence = len(cluster_indices) / len(data)

            factor = {
                "name": factor_name,
                "strength": float(strength),
                "representative_texts": cluster_texts[:3],  # Top 3
                "keywords": keywords[:10],
                "prevalence": float(prevalence)
            }
            factors.append(factor)

        # Step 4: Suggest DAG structure
        suggested_dag = None
        if outcome_variable and len(factors) > 0:
            suggested_dag = self._suggest_dag_structure(factors, outcome_variable)

        # Step 5: Calculate confidence
        confidence = self._calculate_overall_confidence(factors, len(data))

        self.logger.info(f"Extracted {len(factors)} factors with confidence {confidence:.2f}")

        return factors, suggested_dag, confidence

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts into vector space.

        Uses sentence transformers if available, otherwise falls back to
        simple bag-of-words + TF-IDF.
        """
        try:
            from sentence_transformers import SentenceTransformer

            if self._embedder is None:
                self.logger.info("Loading sentence-transformers model...")
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

            embeddings = self._embedder.encode(texts, show_progress_bar=False)
            return np.array(embeddings)

        except ImportError:
            self.logger.warning("sentence-transformers not available, using TF-IDF fallback")
            return self._embed_texts_tfidf(texts)

    def _embed_texts_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fallback embedding using TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Handle empty list
        if not texts:
            return np.array([]).reshape(0, 100)

        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )

        try:
            embeddings = vectorizer.fit_transform(texts).toarray()
            return embeddings
        except ValueError as e:
            # Handle case where all texts are stop words or empty
            self.logger.warning(f"TF-IDF vectorization failed: {e}, returning zero embeddings")
            return np.zeros((len(texts), 100))

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        min_cluster_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings to identify factors.

        Args:
            embeddings: Text embeddings
            n_clusters: Number of clusters
            min_cluster_size: Minimum cluster size

        Returns:
            Tuple of (labels, cluster_centers)
        """
        from sklearn.cluster import KMeans

        # Use K-means for simplicity (could use HDBSCAN for auto-clustering)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_

        return labels, cluster_centers

    def _extract_keywords(self, texts: List[str], top_k: int = 10) -> List[str]:
        """
        Extract keywords from cluster texts.

        Uses simple word frequency with stop word filtering.
        """
        # Combine all texts
        combined = " ".join(texts).lower()

        # Tokenize
        words = re.findall(r'\b[a-z]{3,15}\b', combined)

        # Remove common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are',
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'but', 'also', 'into', 'for', 'with',
            'about', 'from', 'there', 'their', 'out'
        }

        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]

        # Count frequencies
        word_counts = Counter(filtered_words)

        # Return top k
        keywords = [word for word, count in word_counts.most_common(top_k)]

        return keywords

    def _name_factor(
        self,
        keywords: List[str],
        domain_hints: Optional[List[str]] = None
    ) -> str:
        """
        Generate a name for the factor based on keywords.

        Uses heuristic matching with domain hints if available.
        """
        if not keywords:
            return "unknown_factor"

        # Try to find semantic themes
        themes = {
            "usability": ["confusing", "hard", "difficult", "find", "navigate", "understand", "unclear", "complex"],
            "performance": ["slow", "freeze", "lag", "timeout", "crash", "hang", "loading", "speed"],
            "quality": ["bug", "error", "broken", "issue", "problem", "fail", "wrong", "defect"],
            "support": ["help", "support", "question", "unclear", "documentation", "guide", "manual"],
            "pricing": ["price", "cost", "expensive", "cheap", "value", "pricing", "subscription"],
            "features": ["feature", "functionality", "capability", "missing", "add", "request", "want"],
            "onboarding": ["setup", "start", "onboarding", "getting", "started", "initial", "first"],
            "integration": ["integrate", "api", "connect", "sync", "export", "import", "plugin"]
        }

        # Score each theme
        theme_scores = {}
        for theme, theme_keywords in themes.items():
            score = sum(1 for kw in keywords if kw in theme_keywords)
            if score > 0:
                theme_scores[theme] = score

        # Return best matching theme or construct from keywords
        if theme_scores:
            best_theme = max(theme_scores, key=theme_scores.get)
            return f"{best_theme}_issues"
        else:
            # Construct name from top 2 keywords
            return "_".join(keywords[:2]) + "_issues"

    def _calculate_coherence(
        self,
        cluster_embeddings: np.ndarray,
        center: Optional[np.ndarray]
    ) -> float:
        """
        Calculate cluster coherence (tightness).

        Higher coherence = more consistent factor.
        """
        if len(cluster_embeddings) < 2:
            return 0.5  # Default for small clusters

        if center is None:
            center = cluster_embeddings.mean(axis=0)

        # Calculate average distance to center
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        avg_distance = distances.mean()

        # Convert to coherence score (inverse of distance, scaled)
        # Typical distances are 0-2, so we scale
        coherence = max(0.0, min(1.0, 1.0 - (avg_distance / 2.0)))

        return coherence

    def _suggest_dag_structure(
        self,
        factors: List[Dict],
        outcome_variable: str
    ) -> Dict:
        """
        Suggest DAG structure with factors as nodes.

        Simple heuristic: all factors → outcome
        (could be enhanced with correlation analysis)
        """
        nodes = [f["name"] for f in factors] + [outcome_variable]
        edges = [[f["name"], outcome_variable] for f in factors]

        dag = {
            "nodes": nodes,
            "edges": edges
        }

        return dag

    def _calculate_overall_confidence(
        self,
        factors: List[Dict],
        n_texts: int
    ) -> float:
        """Calculate overall confidence in factor extraction."""
        if not factors:
            return 0.0

        # Factors based on:
        # 1. Number of factors found (more is better, up to a point)
        # 2. Average strength (coherence)
        # 3. Data coverage (prevalence)
        # 4. Sample size

        # Average strength
        avg_strength = np.mean([f["strength"] for f in factors])

        # Total coverage
        total_coverage = sum(f["prevalence"] for f in factors)
        coverage_score = min(1.0, total_coverage / 0.8)  # Target 80% coverage

        # Sample size score (more data = more confident)
        sample_score = min(1.0, n_texts / 100.0)  # 100+ texts = full confidence

        # Number of factors score (prefer 3-7 factors)
        n_factors_score = 1.0 if 3 <= len(factors) <= 7 else 0.7

        # Weighted combination
        confidence = (
            0.4 * avg_strength +
            0.3 * coverage_score +
            0.2 * sample_score +
            0.1 * n_factors_score
        )

        return float(min(1.0, confidence))
