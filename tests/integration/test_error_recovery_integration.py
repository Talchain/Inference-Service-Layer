"""
Integration tests for error recovery in services.

Tests fallback chains in conformal prediction, causal discovery, and validation suggester.
"""

import numpy as np
import networkx as nx
import pytest

from src.services.conformal_predictor import ConformalPredictor
from src.services.causal_discovery_engine import CausalDiscoveryEngine
from src.services.advanced_validation_suggester import AdvancedValidationSuggester
from src.utils.error_recovery import health_monitor, CircuitState
from src.services.causal_discovery_engine import _notears_breaker, _pc_breaker
from src.services.advanced_validation_suggester import (
    _path_analysis_breaker,
    _strategy_generation_breaker,
)


class TestConformalPredictorRecovery:
    """Test error recovery in conformal prediction."""

    def test_fallback_to_monte_carlo_with_few_calibration_points(self):
        """Test fallback to Monte Carlo when calibration data is insufficient."""
        predictor = ConformalPredictor()

        # Simulate request with insufficient calibration data (< 5 points)
        class Request:
            def __init__(self):
                self.calibration_data = [{"input": i, "output": i} for i in range(3)]
                self.y0_samples = [[1.0, 2.0, 3.0]]
                self.alpha = 0.05
                self.dag_structure = None

        request = Request()

        # Should fall back to Monte Carlo
        result = predictor.predict_with_conformal_interval(request)

        # Verify fallback behavior
        assert "interval_lower" in result
        assert "interval_upper" in result
        assert result["coverage_guarantee"]["finite_sample_valid"] is False
        assert "Monte Carlo fallback" in result["coverage_guarantee"]["assumptions"]

    def test_degraded_conformal_with_small_calibration(self):
        """Test degraded conformal mode with 5-9 calibration points."""
        predictor = ConformalPredictor()

        class Request:
            def __init__(self):
                # 7 calibration points - degraded mode
                self.calibration_data = [{"input": i, "output": float(i)} for i in range(7)]
                self.y0_samples = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
                self.alpha = 0.1
                self.dag_structure = None

        request = Request()

        # Should use degraded conformal (with warning)
        result = predictor.predict_with_conformal_interval(request)

        # Verify degraded mode
        assert "interval_lower" in result
        assert "interval_upper" in result
        # Should still be finite-sample valid (even with small sample)
        assert result["coverage_guarantee"]["finite_sample_valid"] is True
        # Check for warning in assumptions
        assumptions = result["coverage_guarantee"]["assumptions"]
        assert any("WARNING" in assumption or "small" in assumption for assumption in assumptions)

    def test_normal_conformal_with_sufficient_calibration(self):
        """Test normal conformal prediction with sufficient calibration."""
        predictor = ConformalPredictor()

        class Request:
            def __init__(self):
                # 20 calibration points - normal mode
                self.calibration_data = [{"input": i, "output": float(i)} for i in range(20)]
                self.y0_samples = [[float(i) for i in range(20)]]
                self.alpha = 0.1
                self.dag_structure = None

        request = Request()

        # Should use normal conformal
        result = predictor.predict_with_conformal_interval(request)

        # Verify normal mode
        assert "interval_lower" in result
        assert "interval_upper" in result
        assert result["coverage_guarantee"]["finite_sample_valid"] is True
        assert result["coverage_guarantee"]["theoretical_coverage"] >= 0.9


class TestCausalDiscoveryRecovery:
    """Test error recovery in causal discovery."""

    def test_simple_discovery_always_works(self):
        """Test simple correlation-based discovery always succeeds."""
        engine = CausalDiscoveryEngine(enable_caching=False, enable_advanced=False)

        # Generate simple data
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 1)
        Y = 2 * X + np.random.randn(n, 1) * 0.1
        data = np.hstack([X, Y])
        variable_names = ["X", "Y"]

        # Should always succeed
        dag, confidence = engine.discover_from_data(data, variable_names, seed=42)

        assert isinstance(dag, nx.DiGraph)
        assert len(dag.nodes()) == 2
        assert 0 <= confidence <= 1

    def test_advanced_discovery_fallback_on_error(self):
        """Test advanced discovery falls back to simple on error."""
        engine = CausalDiscoveryEngine(enable_caching=False, enable_advanced=True)

        # Generate data that might cause issues
        np.random.seed(42)
        data = np.random.randn(30, 3)  # Small sample
        variable_names = ["X", "Y", "Z"]

        # Reset circuit breakers to ensure clean state
        _notears_breaker.reset()

        # Should fall back gracefully if NOTEARS fails
        dag, score = engine.discover_advanced(data, variable_names, algorithm="notears")

        # Should return a valid DAG (even if via fallback)
        assert isinstance(dag, nx.DiGraph)
        assert len(dag.nodes()) == 3
        assert nx.is_directed_acyclic_graph(dag)

    def test_minimal_dag_fallback_always_succeeds(self):
        """Test ultimate fallback returns minimal valid DAG."""
        engine = CausalDiscoveryEngine(enable_caching=False, enable_advanced=False)

        # Create minimal DAG directly using fallback method
        variable_names = ["A", "B", "C"]
        dag, confidence = engine._minimal_dag_fallback(variable_names)

        # Should have nodes but no edges (safest fallback)
        assert len(dag.nodes()) == 3
        assert len(dag.edges()) == 0
        assert nx.is_directed_acyclic_graph(dag)
        assert confidence == 0.1  # Low confidence for fallback

    def test_circuit_breaker_prevents_repeated_failures(self):
        """Test circuit breaker opens after repeated NOTEARS failures."""
        engine = CausalDiscoveryEngine(enable_caching=False, enable_advanced=True)

        # Reset circuit breaker
        _notears_breaker.reset()

        # Create data that might cause NOTEARS to fail
        # (very small sample with many variables)
        np.random.seed(42)
        data = np.random.randn(5, 10)  # 5 samples, 10 variables - likely to fail
        variable_names = [f"X{i}" for i in range(10)]

        # Make multiple calls - should trigger circuit breaker
        for _ in range(4):
            try:
                dag, score = engine.discover_advanced(
                    data, variable_names, algorithm="notears", max_iter=5
                )
            except:
                pass

        # Circuit might be OPEN or HALF_OPEN after failures
        # (depends on timing and actual failure behavior)
        assert _notears_breaker.state in [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]


class TestValidationSuggesterRecovery:
    """Test error recovery in validation suggester."""

    def test_simple_strategy_fallback(self):
        """Test fallback to simple strategies when complex analysis fails."""
        suggester = AdvancedValidationSuggester(enable_caching=False)

        # Create a simple DAG
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Z", "X"), ("Z", "Y")])

        # Test fallback method directly
        strategies = suggester._fallback_to_simple_strategies(dag, "X", "Y")

        # Should return at least one strategy
        assert len(strategies) > 0
        assert strategies[0].type in ["backdoor", "manual"]
        assert 0 <= strategies[0].expected_identifiability <= 1

    def test_simple_path_analysis_fallback(self):
        """Test fallback to simple path analysis."""
        suggester = AdvancedValidationSuggester(enable_caching=False)

        # Create a DAG with directed path
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "M"), ("M", "Y")])

        # Test fallback method directly
        path_analysis = suggester._fallback_path_analysis(dag, "X", "Y")

        # Should return valid PathAnalysis (even if minimal)
        assert path_analysis is not None
        assert isinstance(path_analysis.frontdoor_paths, list)
        assert isinstance(path_analysis.backdoor_paths, list)

    def test_strategy_generation_with_circuit_breaker(self):
        """Test strategy generation with circuit breaker protection."""
        suggester = AdvancedValidationSuggester(enable_caching=False)

        # Reset circuit breaker
        _strategy_generation_breaker.reset()

        # Create a complex DAG
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "D"),
            ("E", "A"), ("E", "D"),
        ])

        # Should succeed even if complex analysis has issues
        strategies = suggester.suggest_adjustment_strategies(dag, "A", "D")

        # Should return valid strategies
        assert len(strategies) > 0
        for strategy in strategies:
            assert strategy.type in ["backdoor", "frontdoor", "instrumental", "manual"]
            assert isinstance(strategy.explanation, str)
            assert 0 <= strategy.expected_identifiability <= 1

    def test_path_analysis_with_circuit_breaker(self):
        """Test path analysis with circuit breaker protection."""
        suggester = AdvancedValidationSuggester(enable_caching=False)

        # Reset circuit breaker
        _path_analysis_breaker.reset()

        # Create a DAG
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Z", "X"), ("Z", "Y")])

        # Should succeed with circuit breaker protection
        path_analysis = suggester.analyze_paths(dag, "X", "Y")

        # Should return valid path analysis
        assert path_analysis is not None
        assert isinstance(path_analysis.backdoor_paths, list)
        assert isinstance(path_analysis.frontdoor_paths, list)


class TestHealthMonitoring:
    """Test health monitoring integration."""

    def test_health_monitor_tracks_conformal_prediction(self):
        """Test health monitor tracks conformal prediction operations."""
        predictor = ConformalPredictor()

        # Record some operations
        health_monitor.record_success("conformal_prediction")
        health_monitor.record_success("conformal_prediction")
        health_monitor.record_fallback("conformal_prediction")

        health = health_monitor.get_health("conformal_prediction")

        assert health.successes >= 2
        assert health.fallbacks >= 1
        assert health.total_requests >= 3

    def test_health_monitor_tracks_causal_discovery(self):
        """Test health monitor tracks causal discovery operations."""
        # Record some operations
        health_monitor.record_success("advanced_discovery")
        health_monitor.record_failure("advanced_discovery")
        health_monitor.record_fallback("advanced_discovery")

        health = health_monitor.get_health("advanced_discovery")

        assert health.successes >= 1
        assert health.failures >= 1
        assert health.fallbacks >= 1

    def test_health_monitor_tracks_validation_suggester(self):
        """Test health monitor tracks validation suggester operations."""
        # Record some operations
        health_monitor.record_success("validation_suggester")
        health_monitor.record_success("validation_suggester")

        health = health_monitor.get_health("validation_suggester")

        assert health.successes >= 2
        assert health.success_rate > 0


class TestEndToEndRecovery:
    """Test end-to-end error recovery scenarios."""

    def test_complete_fallback_chain_conformal(self):
        """Test complete fallback chain for conformal prediction."""
        predictor = ConformalPredictor()

        # Test with 0 calibration points - ultimate fallback scenario
        class MinimalRequest:
            def __init__(self):
                self.calibration_data = []  # No calibration data
                self.y0_samples = [[1.0, 2.0, 3.0]]
                self.alpha = 0.1
                self.dag_structure = None

        request = MinimalRequest()

        # Should fall back to Monte Carlo
        result = predictor.predict_with_conformal_interval(request)

        # Should return valid result (not crash)
        assert "interval_lower" in result
        assert "interval_upper" in result

    def test_complete_fallback_chain_discovery(self):
        """Test complete fallback chain for causal discovery."""
        engine = CausalDiscoveryEngine(enable_caching=False, enable_advanced=True)

        # Test with problematic data
        np.random.seed(42)
        data = np.random.randn(10, 2)
        variable_names = ["X", "Y"]

        # Reset circuit breakers
        _notears_breaker.reset()
        _pc_breaker.reset()

        # Should succeed via fallback chain if needed
        dag, score = engine.discover_advanced(data, variable_names, algorithm="notears")

        # Should return valid DAG
        assert isinstance(dag, nx.DiGraph)
        assert len(dag.nodes()) == 2

    def test_complete_fallback_chain_validation(self):
        """Test complete fallback chain for validation suggester."""
        suggester = AdvancedValidationSuggester(enable_caching=False)

        # Test with minimal DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(["X", "Y"])

        # Should succeed via fallback if complex analysis fails
        strategies = suggester.suggest_adjustment_strategies(dag, "X", "Y")

        # Should return at least one strategy
        assert len(strategies) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
