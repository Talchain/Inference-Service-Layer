"""
Structural Model Parser for Conformal Prediction.

Parses StructuralModel into an executable SCM for simulation.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.shared import StructuralModel

logger = logging.getLogger(__name__)


class ParsedSCM:
    """
    Parsed Structural Causal Model for simulation.

    Evaluates structural equations with interventions and samples
    from specified distributions.
    """

    def __init__(
        self,
        variables: List[str],
        equations: Dict[str, str],
        distributions: Dict[str, Dict[str, Any]],
    ):
        """
        Initialize parsed SCM.

        Args:
            variables: List of variable names
            equations: Mapping from variable to equation string
            distributions: Mapping from variable to distribution spec
        """
        self.variables = variables
        self.equations = equations
        self.distributions = distributions
        self._rng = np.random.default_rng()

    def simulate(
        self,
        intervention: Dict[str, float],
        samples: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate from the SCM with interventions.

        Args:
            intervention: Variables to intervene on (do(X=x))
            samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping each variable to array of sampled values
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        results: Dict[str, np.ndarray] = {}

        # For each sample, evaluate the SCM
        for var in self.variables:
            results[var] = np.zeros(samples)

        for i in range(samples):
            # Sample exogenous variables
            exogenous = self._sample_exogenous()

            # Evaluate in topological order
            values = self._evaluate_single(intervention, exogenous)

            for var, val in values.items():
                results[var][i] = val

        return results

    def _sample_exogenous(self) -> Dict[str, float]:
        """Sample values for exogenous variables from their distributions."""
        exogenous = {}

        for var, dist_spec in self.distributions.items():
            dist_type = dist_spec.get("type", "normal")
            params = dist_spec.get("parameters", {})

            if dist_type == "normal":
                mean = params.get("mean", 0.0)
                std = params.get("std", 1.0)
                exogenous[var] = float(self._rng.normal(mean, std))

            elif dist_type == "uniform":
                low = params.get("low", params.get("min", 0.0))
                high = params.get("high", params.get("max", 1.0))
                exogenous[var] = float(self._rng.uniform(low, high))

            elif dist_type == "beta":
                alpha = params.get("alpha", params.get("a", 2.0))
                beta = params.get("beta", params.get("b", 2.0))
                exogenous[var] = float(self._rng.beta(alpha, beta))

            elif dist_type == "exponential":
                scale = params.get("scale", params.get("lambda", 1.0))
                exogenous[var] = float(self._rng.exponential(scale))

            else:
                # Default to standard normal
                logger.warning(f"Unknown distribution '{dist_type}', using normal(0,1)")
                exogenous[var] = float(self._rng.normal(0, 1))

        return exogenous

    def _evaluate_single(
        self, intervention: Dict[str, float], exogenous: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate SCM for a single sample.

        Args:
            intervention: Intervention values (override structural equations)
            exogenous: Sampled exogenous values

        Returns:
            Dictionary of variable values
        """
        values: Dict[str, float] = {}

        # Start with exogenous values
        values.update(exogenous)

        # Apply interventions first (do-calculus: cut incoming edges)
        values.update(intervention)

        # Evaluate equations in order (assuming topological order in variables list)
        for var in self.variables:
            if var in intervention:
                # Intervention overrides structural equation
                values[var] = intervention[var]
            elif var in self.equations:
                # Evaluate structural equation
                eq = self.equations[var]
                try:
                    values[var] = self._evaluate_equation(eq, values)
                except Exception as e:
                    logger.warning(f"Equation eval failed for {var}: {e}")
                    values[var] = 0.0
            # else: exogenous value already set

        return values

    def _evaluate_equation(self, equation: str, values: Dict[str, float]) -> float:
        """
        Safely evaluate a structural equation.

        Args:
            equation: Equation string (e.g., "10 + 2*X + 3*Z")
            values: Current variable values

        Returns:
            Evaluated result
        """
        # Create safe namespace with only math operations and current values
        safe_namespace = {
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
            "round": round,
        }

        # Add numpy functions for more complex operations
        safe_namespace.update({
            "sqrt": np.sqrt,
            "exp": np.exp,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
        })

        # Add current variable values
        safe_namespace.update(values)

        # Evaluate the equation
        try:
            result = eval(equation, safe_namespace)
            return float(result)
        except Exception as e:
            logger.error(f"Failed to evaluate equation '{equation}': {e}")
            raise


class StructuralModelParser:
    """
    Parser for StructuralModel into executable ParsedSCM.
    """

    def parse(self, model: StructuralModel) -> ParsedSCM:
        """
        Parse a StructuralModel into an executable SCM.

        Args:
            model: StructuralModel instance with variables, equations, distributions

        Returns:
            ParsedSCM instance ready for simulation
        """
        # Extract distribution specs
        distributions = {}
        for var, dist in model.distributions.items():
            distributions[var] = {
                "type": dist.type,
                "parameters": dist.parameters,
            }

        return ParsedSCM(
            variables=list(model.variables),
            equations=dict(model.equations),
            distributions=distributions,
        )
