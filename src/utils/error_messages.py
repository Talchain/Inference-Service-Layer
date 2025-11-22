"""
Enhanced error messages with actionable suggestions.

Provides developer-friendly error messages with:
- Clear descriptions
- Actionable suggestions
- Links to documentation
- Context information
"""

from typing import Any, Dict, List, Optional


class EnhancedError(Exception):
    """Base class for errors with actionable suggestions."""

    def __init__(
        self,
        code: str,
        message: str,
        suggestions: List[str],
        documentation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced error.

        Args:
            code: Error code (e.g., "CAUSAL_NOT_IDENTIFIABLE")
            message: Human-readable error message
            suggestions: List of actionable suggestions
            documentation: URL to relevant documentation
            context: Additional context information
        """
        self.code = code
        self.message = message
        self.suggestions = suggestions
        self.documentation = documentation or "https://docs.olumi.com"
        self.context = context or {}

        super().__init__(message)

    def to_response(self) -> Dict[str, Any]:
        """
        Convert to API error response format.

        Returns:
            Dict suitable for JSON response
        """
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "suggestions": self.suggestions,
                "documentation": self.documentation,
                "context": self.context
            }
        }


# Pre-defined enhanced errors

class CausalNotIdentifiableError(EnhancedError):
    """Causal effect cannot be identified from the DAG."""

    def __init__(self, treatment: str, outcome: str, reason: str = ""):
        super().__init__(
            code="CAUSAL_NOT_IDENTIFIABLE",
            message=f"Causal effect of {treatment} on {outcome} is not identifiable from this DAG",
            suggestions=[
                "Add conditional independence assumptions to block backdoor paths",
                "Include measurements of confounding variables",
                "Check for unmeasured common causes",
                "Verify the causal structure is correctly specified",
                f"Reason: {reason}" if reason else "Use frontdoor or instrumental variable methods"
            ],
            documentation="https://docs.olumi.com/causal-identification",
            context={"treatment": treatment, "outcome": outcome, "reason": reason}
        )


class InvalidDAGCyclicError(EnhancedError):
    """DAG contains cycles."""

    def __init__(self, cycle_nodes: List[str]):
        super().__init__(
            code="INVALID_DAG_CYCLIC",
            message="DAG contains cycles (not a valid directed acyclic graph)",
            suggestions=[
                f"Remove edges that create the cycle: {' → '.join(cycle_nodes)}",
                "Check that edge directions are correct",
                "Use topological sort to verify acyclicity",
                "Consider using a feedback loop representation instead"
            ],
            documentation="https://docs.olumi.com/dag-validation",
            context={"cycle_nodes": cycle_nodes}
        )


class BatchSizeExceededError(EnhancedError):
    """Batch request exceeds size limit."""

    def __init__(self, batch_size: int, max_size: int, endpoint: str):
        super().__init__(
            code="BATCH_SIZE_EXCEEDED",
            message=f"Batch size {batch_size} exceeds limit of {max_size}",
            suggestions=[
                f"Split into multiple batches of {max_size} requests each",
                f"Maximum for {endpoint}: {max_size} requests per batch",
                "Use pagination or chunking for larger datasets",
                "Consider using streaming endpoints for very large batches"
            ],
            documentation="https://docs.olumi.com/batch-endpoints",
            context={
                "batch_size": batch_size,
                "max_size": max_size,
                "endpoint": endpoint
            }
        )


class EquationEvaluationError(EnhancedError):
    """Equation cannot be evaluated."""

    def __init__(self, equation: str, variable: str, reason: str):
        super().__init__(
            code="EQUATION_EVALUATION_ERROR",
            message=f"Cannot evaluate equation for variable '{variable}': {reason}",
            suggestions=[
                "Check equation syntax (allowed: +, -, *, /, **, %, //)",
                "Verify all referenced variables exist in the model",
                "Use only whitelisted functions: sqrt, exp, log, abs, sin, cos, tan",
                f"Problematic equation: {equation}",
                "Ensure variables are defined before use (topological order)"
            ],
            documentation="https://docs.olumi.com/structural-equations",
            context={
                "equation": equation,
                "variable": variable,
                "reason": reason
            }
        )


class MissingVariableError(EnhancedError):
    """Required variable not found in model."""

    def __init__(self, variable: str, available_variables: List[str]):
        super().__init__(
            code="MISSING_VARIABLE",
            message=f"Variable '{variable}' not found in model",
            suggestions=[
                f"Add '{variable}' to the variables list",
                f"Available variables: {', '.join(available_variables)}",
                "Check for typos in variable names",
                "Ensure variable is defined before use in equations"
            ],
            documentation="https://docs.olumi.com/variable-definitions",
            context={
                "missing_variable": variable,
                "available_variables": available_variables
            }
        )


class MonteCarloConvergenceError(EnhancedError):
    """Monte Carlo simulation did not converge."""

    def __init__(self, iterations: int, variance: float):
        super().__init__(
            code="MONTE_CARLO_NOT_CONVERGED",
            message=f"Monte Carlo did not converge after {iterations} iterations (variance: {variance:.4f})",
            suggestions=[
                f"Increase max iterations (current: {iterations})",
                "Check for high-variance distributions in the model",
                "Consider using importance sampling for better convergence",
                "Verify distribution parameters are reasonable",
                "Use narrower prior distributions if possible"
            ],
            documentation="https://docs.olumi.com/monte-carlo",
            context={
                "iterations": iterations,
                "variance": variance
            }
        )


class RateLimitExceededError(EnhancedError):
    """API rate limit exceeded."""

    def __init__(self, limit: int, retry_after: int):
        super().__init__(
            code="RATE_LIMIT_EXCEEDED",
            message=f"Rate limit of {limit} requests per minute exceeded",
            suggestions=[
                f"Wait {retry_after} seconds before retrying",
                "Implement exponential backoff in your client",
                "Use batch endpoints to reduce number of requests",
                "Contact support to increase rate limits if needed",
                "Cache responses to reduce redundant requests"
            ],
            documentation="https://docs.olumi.com/rate-limits",
            context={
                "limit": limit,
                "retry_after": retry_after
            }
        )


class InvalidDistributionError(EnhancedError):
    """Distribution specification is invalid."""

    def __init__(self, dist_type: str, reason: str):
        super().__init__(
            code="INVALID_DISTRIBUTION",
            message=f"Invalid distribution '{dist_type}': {reason}",
            suggestions=[
                "Supported distributions: normal, uniform, beta, exponential",
                "Check distribution parameters are valid",
                "Normal: requires mean and std (std > 0)",
                "Uniform: requires min and max (min < max)",
                "Beta: requires alpha and beta (both > 0)",
                "Exponential: requires scale (scale > 0)"
            ],
            documentation="https://docs.olumi.com/distributions",
            context={
                "distribution_type": dist_type,
                "reason": reason
            }
        )


class MemoryLimitError(EnhancedError):
    """Memory usage limit exceeded."""

    def __init__(self, used_mb: float, limit_mb: int):
        super().__init__(
            code="MEMORY_LIMIT_EXCEEDED",
            message=f"Operation used {used_mb:.1f}MB, exceeds limit of {limit_mb}MB",
            suggestions=[
                "Reduce batch size or number of Monte Carlo iterations",
                "Simplify the structural model (fewer variables/equations)",
                "Use adaptive sampling to reduce memory usage",
                "Process data in smaller chunks",
                "Contact support for higher memory limits"
            ],
            documentation="https://docs.olumi.com/resource-limits",
            context={
                "used_mb": used_mb,
                "limit_mb": limit_mb
            }
        )


class TimeoutError(EnhancedError):
    """Request timed out."""

    def __init__(self, timeout_seconds: int, operation: str):
        super().__init__(
            code="REQUEST_TIMEOUT",
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            suggestions=[
                "Simplify the request (fewer variables, smaller DAG)",
                "Reduce Monte Carlo iterations for faster results",
                "Use adaptive sampling for automatic convergence",
                "Split large batch requests into smaller chunks",
                "Increase timeout limit if operation is expected to be slow"
            ],
            documentation="https://docs.olumi.com/timeouts",
            context={
                "timeout_seconds": timeout_seconds,
                "operation": operation
            }
        )
