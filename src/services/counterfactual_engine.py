"""
Counterfactual Engine for "what-if" scenario analysis.

Implements counterfactual reasoning using structural causal models
with Monte Carlo simulation for uncertainty quantification.
"""

import ast
import logging
import math
import operator
import re
import threading
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from src.config import get_settings
from src.models.requests import CounterfactualRequest
from src.models.responses import (
    ConfidenceInterval,
    CounterfactualResponse,
    PredictionResults,
    ScenarioDescription,
    SensitivityRange,
    UncertaintyBreakdown,
)
from src.models.shared import UncertaintyLevel
from src.services.explanation_generator import ExplanationGenerator
from src.utils.determinism import canonical_hash, make_deterministic
from src.utils.rng import SeededRNG

logger = logging.getLogger(__name__)
settings = get_settings()

# C2(cf): bound for the topological-sort cache. The former cache was an unbounded dict
# keyed by json.dumps(equations) (client-controllable), never evicted, on the live
# request path (this engine is a module-level singleton in causal.py) — a monotonic
# memory leak (same class as phase4's removed _policy_cache). It is a real read-through
# optimisation (repeated identical models reuse the sort — including the several
# re-sorts within one adaptive-MC request), and it is exercised as a feature by
# tests/performance/test_optimization_gains.py, so it is BOUNDED rather than removed:
# at capacity the oldest-inserted entry is evicted, capping growth regardless of how
# many distinct models are seen. Entries are small ((var, equation) lists) and
# content-addressed, so a cache hit is always correct for that exact equation set.
_TOPO_SORT_CACHE_MAX = 128


class CounterfactualEngine:
    """
    Performs counterfactual analysis using structural causal models.

    Uses Monte Carlo simulation to propagate uncertainty and generate
    robust predictions with confidence intervals.
    """

    def __init__(self, enable_adaptive_sampling: bool = True) -> None:
        """
        Initialize the engine.

        Args:
            enable_adaptive_sampling: Use adaptive sampling for efficiency (default: True)
        """
        self.explanation_generator = ExplanationGenerator()
        self.num_iterations = settings.MAX_MONTE_CARLO_ITERATIONS
        self.enable_adaptive_sampling = enable_adaptive_sampling

        # Bounded read-through cache for topological-sort results (see
        # _TOPO_SORT_CACHE_MAX): capped so it cannot grow unbounded on the live
        # request path.
        self._topo_sort_cache: Dict[str, List[Tuple[str, str]]] = {}
        # Guards every read/insert/evict of the cache above. This engine is a
        # module-level singleton (`counterfactual_engine` in causal.py) served on a
        # concurrent request path, so the unlocked dict's evict step —
        # `pop(next(iter(...)))` racing a concurrent `[key] = result` — can raise
        # "dictionary changed size during iteration" or corrupt the FIFO ordering.
        # A plain Lock preserves the EXACT FIFO-128 / no-TTL semantics from PR #89
        # and keeps `_topo_sort_cache` a real dict (test_optimization_gains reads
        # `len(engine._topo_sort_cache)`). The repo's thread-safe cache utilities —
        # utils/cache.TTLCache and infrastructure/memory_cache.get_named_cache —
        # were REJECTED: both are LRU + mandatory-TTL (MemoryCache also spawns a
        # background cleanup thread and is a shared singleton), so neither can
        # preserve FIFO/no-TTL, and TTLCache has no __len__ for the perf test.
        self._topo_sort_cache_lock = threading.Lock()

    def analyze(self, request: CounterfactualRequest) -> CounterfactualResponse:
        """
        Perform counterfactual analysis.

        Args:
            request: Counterfactual request with model and intervention

        Returns:
            CounterfactualResponse: Prediction results with uncertainty
        """
        # Input hardening (A3, 2026-07-22): fail loud on an unresolvable outcome
        # BEFORE any sampling, so a client-input defect maps to a clean 422 (via the
        # route's D-12(cf) except-ValueError) instead of a mislabeled KeyError-500.
        self._require_resolvable_outcome(request)

        # Create per-request RNG for thread-safe determinism
        rng = make_deterministic(request.model_dump())

        # F11 (A3, 2026-07-22): NEVER log raw intervention values — they are the
        # client's private scenario inputs. Log only the sorted intervention
        # variable NAMES + a count; the R-004 correlation key is the
        # request_hash (canonical_hash) already logged one line above, which is a
        # digest, not a recoverable value.
        logger.info(
            "counterfactual_analysis_started",
            extra={
                "request_hash": canonical_hash(request.model_dump()),
                "outcome": request.outcome,
                "intervention_keys": sorted(request.intervention),
                "intervention_count": len(request.intervention),
                "seed": rng.seed,
            },
        )

        try:
            # Run Monte Carlo simulation
            samples = self._run_monte_carlo(request, rng)

            # Input hardening (A3, 2026-07-22): fail loud on a non-finite outcome
            # BEFORE building the response, so a mathematically-invalid model maps to
            # a clean 422 (via the route's D-12(cf)) instead of a 500 raised at
            # JSON serialization (Starlette's allow_nan=False), OUTSIDE the route try.
            self._require_finite_outcome(request.outcome, samples[request.outcome])

            # Compute prediction results
            prediction = self._compute_prediction(samples, request.outcome)

            # Analyze overall outcome uncertainty (F3a/F3c, A3 2026-07-22: the
            # per-factor `sources` breakdown is OMITTED — it labeled each input's
            # own variance as its "impact" on the outcome, a leverage-blind
            # fabrication whose ranking inverts. Only the honest overall level,
            # derived from the OUTCOME samples' coefficient of variation, remains.)
            uncertainty = self._analyze_uncertainty(request, samples)

            # Create scenario description
            scenario = ScenarioDescription(
                intervention=request.intervention,
                outcome=request.outcome,
                context=request.context,
            )

            # Generate explanation. Robustness is OMITTED (F3c): the former
            # "robustness/critical-assumptions" block reported a flat
            # abs(baseline*0.15) "perturbation" and a score that was a constant
            # function of the baseline sign — neither parsed or perturbed the
            # equations, so it implemented none of its label.
            explanation = self.explanation_generator.generate_counterfactual_explanation(
                outcome=request.outcome,
                intervention=request.intervention,
                point_estimate=prediction.point_estimate,
                ci_lower=prediction.confidence_interval.lower,
                ci_upper=prediction.confidence_interval.upper,
                uncertainty_level=uncertainty.overall.value,
            )

            return CounterfactualResponse(
                scenario=scenario,
                prediction=prediction,
                uncertainty=uncertainty,
                explanation=explanation,
            )

        except ValueError as e:
            # Client-input defects — an undefined/dangling outcome, a malformed or
            # circular structural equation, a non-finite outcome, or a numpy
            # scale<0 — fail loud as ValueError, and the route's D-12(cf) handler
            # maps them to a clean 422 (the only internal ValueError source,
            # unknown distribution type, is unreachable behind the Pydantic enum).
            # These are client errors, not server incidents: log at WARNING without
            # a server-path traceback (the route also logs counterfactual_invalid_input
            # at WARNING). Reserve ERROR + traceback for genuinely unexpected faults.
            logger.warning("counterfactual_analysis_failed", extra={"error": str(e)})
            raise
        except Exception as e:
            logger.error("counterfactual_analysis_failed", exc_info=True)
            raise

    def _require_resolvable_outcome(self, request: CounterfactualRequest) -> None:
        """Fail loud (input hardening) when `outcome` names a variable the model
        can never produce a value for.

        `_run_fixed_monte_carlo` populates the `samples` dict from exactly four
        sources: the exogenous distributions, the intervention, the context, and
        the structural equations. An `outcome` absent from all four is never
        sampled, so `_run_adaptive_monte_carlo`'s `batch_samples[request.outcome]`
        (and, later, `_compute_prediction`'s `samples[outcome_var]`) raises
        KeyError -> a mislabeled 500. That is a client-input defect (a typo'd or
        dangling outcome name from the upstream graph-builder), not an internal
        failure: raise ValueError so the route's D-12(cf) handler maps it to a clean
        422 naming the unresolved variable. Mirrors phase4's
        `_require_valued_decision_node`. Single source of the message.

        Also rejects (F2) a variable set in BOTH `intervention` and `context`:
        do(X) and observe(X) are contradictory, and the sampler's write order
        lets context silently overwrite the intervention.
        """
        # F2 (A3, 2026-07-22): do(X) and observe(X) are contradictory. A variable
        # in BOTH `intervention` and `context` is a client-input defect — the
        # sampler applies the intervention and then the context OVER it (write
        # order), so a declared do(X=10) is silently computed at X=context
        # (Y=2X with intervention X=10 + context X=3 returns 6, not 20). Reject
        # BEFORE sampling with a clean 422 (route D-12(cf)) naming the colliding
        # variable(s), rather than returning a plausible-looking wrong value.
        collisions = sorted(set(request.intervention) & set(request.context or {}))
        if collisions:
            raise ValueError(
                f"Variable(s) {collisions} appear in BOTH `intervention` and "
                f"`context`: do({', '.join(collisions)}) and "
                f"observe({', '.join(collisions)}) are contradictory. An "
                f"intervention FIXES a variable; a context OBSERVES it. Provide "
                f"each variable in exactly one of `intervention` or `context`."
            )

        # F3d residual (A3, 2026-07-23): reject an intervention/context key that
        # names a variable ABSENT from the structural model, rather than silently
        # no-opping it. `_run_fixed_monte_carlo` writes `samples[key] = np.full(...)`
        # for EVERY intervention and context key, but a key that is not a declared
        # variable, an equation-defined variable, or an exogenous distribution is
        # never READ by any equation — so the model returns the OBSERVATIONAL
        # BASELINE with HTTP 200: a plausible-looking answer to a question that was
        # never evaluated (a client typo do(Q=5) on a model of X,Z,Y). The `context`
        # channel has the identical hole (same unread np.full write). Validate here,
        # the single validation home that already carries the do/observe collision
        # guard above, so there is ONE place a key is checked against the model.
        #
        # Redaction (F11 discipline): the message names the unknown KEYS and the
        # known-variable-set SIZE only — never the request VALUES (a client's
        # private scenario inputs). This message is surfaced verbatim to the client
        # AND written to the route's `counterfactual_invalid_input` warning log.
        known_variables = (
            set(request.model.variables)
            | set(request.model.equations)
            | set(request.model.distributions)
        )
        unknown_intervention = sorted(set(request.intervention) - known_variables)
        unknown_context = sorted(set(request.context or {}) - known_variables)
        if unknown_intervention or unknown_context:
            channels = []
            if unknown_intervention:
                channels.append(f"intervention key(s) {unknown_intervention}")
            if unknown_context:
                channels.append(f"context key(s) {unknown_context}")
            raise ValueError(
                f"{' and '.join(channels)} name variable(s) that are not defined "
                f"by the structural model (the model declares {len(known_variables)} "
                f"known variable(s): its `variables`, structural-equation, and "
                f"exogenous-distribution names). An intervention or context on an "
                f"unknown variable is silently ignored by the sampler and returns "
                f"the observational baseline for a question that was never "
                f"evaluated. Provide each key from the model's known variables, or "
                f"add the variable to the model."
            )

        # C1 residual (F-1, A3 2026-07-23): reject a NON-FINITE intervention/context
        # VALUE (NaN or +/-inf) BEFORE sampling. The unknown-key guard above admits
        # any key in the model's known set, and `_require_finite_outcome` checks only
        # the OUTCOME samples — so a non-finite value on a declared-but-DISCONNECTED
        # variable (one no equation reads) passes both guards, leaves the outcome
        # finite, and is echoed back into `scenario.intervention`/`scenario.context`.
        # Starlette's JSONResponse renders with `allow_nan=False` and raises OUTSIDE
        # the route try -> an unhandled 500 (ISL_COMPUTATION_ERROR, retryable:true —
        # which is false: it fails deterministically forever). That is a client-input
        # defect, not a server incident: raise ValueError -> route D-12(cf) -> a clean
        # 422. Validated HERE, the single validation home, alongside the do/observe
        # and unknown-key guards.
        #
        # Redaction (F11 discipline): the message names the offending KEY(s) and the
        # category word "non-finite" only — never the request VALUE (a client's
        # private scenario input; this message is surfaced to the client AND written
        # to the route's `counterfactual_invalid_input` warning log). Values are
        # `Dict[str, float]` per CounterfactualRequest, so `math.isfinite` applies.
        nonfinite_keys = sorted(
            {
                key
                for source in (request.intervention, request.context or {})
                for key, value in source.items()
                if not math.isfinite(value)
            }
        )
        if nonfinite_keys:
            raise ValueError(
                f"Intervention/context key(s) {nonfinite_keys} have a non-finite "
                f"value (NaN or +/-infinity). A non-finite input cannot be used in a "
                f"structural computation and would fail response serialization. "
                f"Provide a finite numeric value for each key."
            )

        resolvable = (
            set(request.model.distributions)
            | set(request.intervention)
            | set(request.context or {})
            | set(request.model.equations)
        )
        if request.outcome not in resolvable:
            raise ValueError(
                f"Outcome variable '{request.outcome}' is not defined by the "
                f"structural model: it has no structural equation and is not an "
                f"exogenous distribution, an intervention, or a context variable, "
                f"so the model can never compute a value for it. Add an equation "
                f"for '{request.outcome}', or supply it as a distribution, "
                f"intervention, or context value."
            )

    def _require_finite_outcome(self, outcome_var: str, outcome_samples: np.ndarray) -> None:
        """Fail loud (input hardening) when the structural model computes a
        non-finite outcome (NaN or +/-inf) on the sampled inputs.

        A `log()` of a non-positive number, a division by zero, or an overflow in a
        structural equation yields NaN/inf sample values. A non-finite point
        estimate / interval then serialize-fails in Starlette's JSONResponse
        (`allow_nan=False`) -> an unhandled 500 at RESPONSE RENDERING, OUTSIDE the
        route's try. It is a client-input defect (a mathematically invalid model on
        the intervened/context inputs), not an internal failure: raise ValueError ->
        route D-12(cf) -> 422.

        We REJECT rather than clamp: any finite substitute for an undefined
        computation would be a fabricated value (the absent-as-0 fabrication class).
        Fail loud is the doctrine. A legitimately-zero or large-but-finite outcome
        is finite and passes untouched (see the positive-control tests). Single
        source of the message.
        """
        if not np.all(np.isfinite(np.asarray(outcome_samples, dtype=float))):
            raise ValueError(
                f"The structural model produced a non-finite value for outcome "
                f"'{outcome_var}' (check for log of a non-positive number, division "
                f"by zero, or overflow in the equations)."
            )

    def _topological_sort_equations(self, equations: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        Topologically sort equations based on variable dependencies.

        OPTIMIZATION: Results are cached (bounded, see _TOPO_SORT_CACHE_MAX) to avoid
        recomputation for identical models.

        Args:
            equations: Dict mapping variable names to equations

        Returns:
            List of (var_name, equation) tuples in dependency order
        """
        # Create cache key from equations
        import json

        cache_key = json.dumps(equations, sort_keys=True)

        # Check cache first (under the lock — see __init__). The topo sort itself
        # is a pure function of `equations`, so it runs OUTSIDE the lock; only the
        # dict read/insert/evict are guarded.
        with self._topo_sort_cache_lock:
            if cache_key in self._topo_sort_cache:
                return self._topo_sort_cache[cache_key]

        # Build dependency graph
        dependencies: Dict[str, set] = {}
        for var_name, equation in equations.items():
            # Find all variable references in the equation
            # Match variable names (alphanumeric + underscore)
            var_refs = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", equation))
            # Remove known non-variables (math functions, constants)
            var_refs -= {"exp", "log", "sqrt", "abs", "min", "max", "pow"}
            # Only keep variables that are in the equations dict
            dependencies[var_name] = var_refs & set(equations.keys())

        # Topological sort using Kahn's algorithm
        sorted_vars = []
        in_degree = {var: len(deps) for var, deps in dependencies.items()}
        queue = [var for var, degree in in_degree.items() if degree == 0]

        while queue:
            var = queue.pop(0)
            sorted_vars.append(var)

            # Reduce in-degree for dependent variables
            for other_var, deps in dependencies.items():
                if var in deps and other_var not in sorted_vars:
                    in_degree[other_var] -= 1
                    if in_degree[other_var] == 0:
                        queue.append(other_var)

        # Check for cycles
        if len(sorted_vars) != len(equations):
            raise ValueError("Circular dependencies detected in structural equations")

        # Cache the result, evicting the oldest-inserted entry at capacity so the
        # cache cannot grow unbounded on a client-controllable key (dict preserves
        # insertion order, Python 3.7+).
        result = [(var, equations[var]) for var in sorted_vars]
        with self._topo_sort_cache_lock:
            if len(self._topo_sort_cache) >= _TOPO_SORT_CACHE_MAX:
                self._topo_sort_cache.pop(next(iter(self._topo_sort_cache)))
            self._topo_sort_cache[cache_key] = result

        return result

    def _run_monte_carlo(
        self, request: CounterfactualRequest, rng: SeededRNG
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation of the structural model.

        OPTIMIZATION: Adaptive sampling - starts small, grows if needed.
        Provides 2-5x speedup for low-variance models.

        Args:
            request: Counterfactual request
            rng: Per-request random number generator

        Returns:
            Dict mapping variable names to arrays of sampled values
        """
        if self.enable_adaptive_sampling:
            return self._run_adaptive_monte_carlo(request, rng)
        else:
            return self._run_fixed_monte_carlo(request, self.num_iterations, rng)

    def _run_adaptive_monte_carlo(
        self, request: CounterfactualRequest, rng: SeededRNG
    ) -> Dict[str, np.ndarray]:
        """
        Adaptive Monte Carlo: start small, grow if variance is high.

        Strategy:
        - Start with 100 samples
        - Check coefficient of variation (CV = std/mean)
        - If CV < 0.1 (converged), stop early
        - Otherwise, double samples and continue
        - Max: self.num_iterations

        Args:
            request: Counterfactual request
            rng: Per-request random number generator
        """
        batch_size = 100
        all_samples: Dict[str, List[float]] = {request.outcome: []}

        while len(all_samples[request.outcome]) < self.num_iterations:
            # Run batch
            current_size = min(batch_size, self.num_iterations - len(all_samples[request.outcome]))
            batch_samples = self._run_fixed_monte_carlo(request, current_size, rng)

            # Accumulate samples for outcome variable
            if not all_samples[request.outcome]:
                all_samples[request.outcome] = batch_samples[request.outcome].tolist()
            else:
                all_samples[request.outcome].extend(batch_samples[request.outcome].tolist())

            # Check convergence (only if we have enough samples)
            if len(all_samples[request.outcome]) >= 100:
                outcome_array = np.array(all_samples[request.outcome])
                mean_val = np.mean(outcome_array)
                std_val = np.std(outcome_array)

                # Coefficient of variation
                cv = std_val / abs(mean_val) if mean_val != 0 else float("inf")

                if cv < 0.1:  # Converged!
                    logger.info(
                        "adaptive_sampling_converged",
                        extra={
                            "samples": len(all_samples[request.outcome]),
                            "cv": cv,
                            "savings": f"{(1 - len(all_samples[request.outcome]) / self.num_iterations) * 100:.0f}%",
                        },
                    )
                    break

            # Increase batch size for next iteration
            batch_size = min(batch_size * 2, self.num_iterations)

        # Run one final simulation with the determined sample count
        # (we only tracked outcome for convergence, need all variables)
        final_count = len(all_samples[request.outcome])
        return self._run_fixed_monte_carlo(request, final_count, rng)

    def _run_fixed_monte_carlo(
        self, request: CounterfactualRequest, num_samples: int, rng: SeededRNG
    ) -> Dict[str, np.ndarray]:
        """
        Run fixed-size Monte Carlo simulation.

        Args:
            request: Counterfactual request
            num_samples: Number of samples to generate
            rng: Per-request random number generator

        Returns:
            Dict mapping variable names to arrays of sampled values
        """
        samples: Dict[str, np.ndarray] = {}

        # Sample exogenous variables from their distributions
        for var_name, dist in request.model.distributions.items():
            try:
                samples[var_name] = self._sample_distribution(
                    dist.type.value, dist.parameters, num_samples, rng
                )
            except KeyError as e:
                # Input hardening (A3, 2026-07-22): `_sample_distribution`
                # dereferences ONLY the client-supplied distribution parameters
                # (`params[...]`); the RNG `*_array` helpers it calls take positional
                # floats and access no dict (src/utils/rng.py). A KeyError here is
                # therefore UNAMBIGUOUSLY a missing required parameter for this
                # exogenous variable's distribution — a client-input defect, not an
                # internal bug — so re-raise it as ValueError (route D-12(cf) -> 422)
                # naming the variable and the exact missing parameter DERIVED from the
                # KeyError (not a hand-maintained required-params mirror). Scoped to
                # this single call so a genuine internal KeyError elsewhere still
                # surfaces as a 500.
                missing = e.args[0] if e.args else str(e)
                raise ValueError(
                    f"Exogenous distribution for variable '{var_name}' (type "
                    f"'{dist.type.value}') is missing required parameter '{missing}'."
                ) from e
            except ValueError as e:
                # Input hardening (F3): a client-supplied distribution parameter
                # outside its valid domain — most reachably a NEGATIVE `std` on a
                # normal, since the Distribution model does not constrain std >= 0,
                # which numpy rejects as a bare `ValueError: scale < 0` — reaches
                # the RNG helper as an un-actionable message that does not name the
                # variable. Re-raise it NAMING the offending variable and its
                # distribution (route D-12(cf) -> 422), preserving the underlying
                # domain reason. Scoped to this single sample call: `_sample_
                # distribution` only invokes numpy RNG helpers on client params, so
                # any ValueError here is a client parameter-domain defect, not an
                # internal bug (its sole internal ValueError, an unknown distribution
                # type, is unreachable behind the Pydantic-validated enum).
                raise ValueError(
                    f"Exogenous distribution for variable '{var_name}' (type "
                    f"'{dist.type.value}') has an invalid parameter: {e}."
                ) from e

        # Apply intervention (set intervened variables to fixed values)
        for var_name, value in request.intervention.items():
            samples[var_name] = np.full(num_samples, value)

        # Apply context (observed values)
        if request.context:
            for var_name, value in request.context.items():
                samples[var_name] = np.full(num_samples, value)

        # Evaluate structural equations topologically (in dependency order)
        # OPTIMIZATION: Topological sort is cached, no recomputation
        sorted_equations = self._topological_sort_equations(request.model.equations)
        for var_name, equation in sorted_equations:
            if var_name not in samples:  # Skip if already set by intervention/context
                samples[var_name] = self._evaluate_equation(equation, samples)

        return samples

    def _sample_distribution(
        self, dist_type: str, params: Dict[str, float], size: int, rng: SeededRNG
    ) -> np.ndarray:
        """
        Sample from a probability distribution using per-request RNG.

        Args:
            dist_type: Distribution type (normal, uniform, beta, exponential)
            params: Distribution parameters
            size: Number of samples
            rng: Per-request random number generator

        Returns:
            Array of samples
        """
        if dist_type == "normal":
            return rng.normal_array(params["mean"], params["std"], size)
        elif dist_type == "uniform":
            return rng.uniform_array(params["min"], params["max"], size)
        elif dist_type == "beta":
            return rng.beta_array(params["alpha"], params["beta"], size)
        elif dist_type == "exponential":
            # Use inverse transform sampling for exponential
            # X = -scale * ln(U) where U ~ Uniform(0, 1)
            u = rng.uniform_array(0.0, 1.0, size)
            return np.asarray(-params["scale"] * np.log(u))
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def _evaluate_equation(self, equation: str, samples: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Safely evaluate a structural equation using AST parsing instead of eval().

        This prevents code injection by only allowing whitelisted operations.

        Args:
            equation: Mathematical expression
            samples: Dictionary of variable samples

        Returns:
            Array of evaluated results

        Security Note:
            Uses AST-based evaluation to prevent code injection attacks.
            Only allows: +, -, *, /, ** and whitelisted functions.
        """
        try:
            # Parse equation into AST
            tree = ast.parse(equation, mode="eval")

            # Evaluate AST with safety checks
            result = self._eval_ast_node(tree.body, samples, depth=0)

            return np.array(result)
        except SyntaxError as e:
            logger.error(f"Syntax error in equation '{equation}': {e}")
            raise ValueError(f"Invalid equation syntax: {equation}")
        except Exception as e:
            logger.error(f"Failed to evaluate equation '{equation}': {e}")
            raise ValueError(f"Invalid equation: {equation}")

    def _eval_ast_node(self, node: ast.AST, samples: Dict[str, np.ndarray], depth: int = 0) -> Any:
        """
        Recursively evaluate AST node with security checks.

        Args:
            node: AST node to evaluate
            samples: Dictionary of variable samples
            depth: Current recursion depth (prevents stack overflow)

        Returns:
            Evaluation result

        Raises:
            ValueError: If unsafe operations detected or depth limit exceeded
        """
        # Prevent deeply nested expressions (DoS protection)
        MAX_DEPTH = 20
        if depth > MAX_DEPTH:
            raise ValueError(f"Equation too complex (max nesting depth {MAX_DEPTH} exceeded)")

        # Whitelist of safe binary operations
        SAFE_OPS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
        }

        # Whitelist of safe unary operations
        SAFE_UNARY_OPS = {
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        # Whitelist of safe functions
        SAFE_FUNCS = {
            "sqrt": np.sqrt,
            "exp": np.exp,
            "log": np.log,
            "abs": np.abs,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
        }

        # Handle different node types
        if isinstance(node, ast.Constant):  # Python ≥3.8
            return node.value
        elif isinstance(node, ast.Num):  # Python <3.8 compatibility
            return node.n
        elif isinstance(node, ast.Name):
            # Variable reference
            if node.id in samples:
                return samples[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            # Binary operation (+, -, *, /, **)
            if type(node.op) not in SAFE_OPS:
                raise ValueError(
                    f"Unsafe operation: {type(node.op).__name__}. "
                    f"Allowed: +, -, *, /, **, %, //"
                )
            left = self._eval_ast_node(node.left, samples, depth + 1)
            right = self._eval_ast_node(node.right, samples, depth + 1)
            return SAFE_OPS[type(node.op)](left, right)  # type: ignore[operator]
        elif isinstance(node, ast.UnaryOp):
            # Unary operation (-, +)
            if type(node.op) not in SAFE_UNARY_OPS:
                raise ValueError(f"Unsafe unary operation: {type(node.op).__name__}")
            operand = self._eval_ast_node(node.operand, samples, depth + 1)
            return SAFE_UNARY_OPS[type(node.op)](operand)  # type: ignore[operator]
        elif isinstance(node, ast.Call):
            # Function call
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in SAFE_FUNCS:
                raise ValueError(
                    f"Unsafe function: {func_name}. " f"Allowed: {', '.join(SAFE_FUNCS.keys())}"
                )
            # Evaluate arguments
            args = [self._eval_ast_node(arg, samples, depth + 1) for arg in node.args]
            if node.keywords:
                raise ValueError("Keyword arguments not allowed in equations")
            return SAFE_FUNCS[func_name](*args)  # type: ignore[operator]
        else:
            raise ValueError(
                f"Unsafe expression type: {type(node).__name__}. "
                f"Only basic arithmetic and whitelisted functions allowed."
            )

    def _compute_prediction(
        self, samples: Dict[str, np.ndarray], outcome_var: str
    ) -> PredictionResults:
        """
        Compute prediction statistics from samples.

        Args:
            samples: Dictionary of variable samples
            outcome_var: Outcome variable name

        Returns:
            PredictionResults with point estimate and intervals
        """
        outcome_samples = samples[outcome_var]

        # Point estimate (median for robustness to outliers)
        point_estimate = float(np.median(outcome_samples))

        # Prediction interval (95% by default)
        # Note: Named "confidence_interval" in API for legacy reasons, but these are
        # percentile-based bounds from Monte Carlo simulation (see ConfidenceInterval docstring)
        ci_lower = float(np.percentile(outcome_samples, 2.5))
        ci_upper = float(np.percentile(outcome_samples, 97.5))

        # Sensitivity range (10th to 90th percentile) - narrower band for typical outcomes
        sens_lower = float(np.percentile(outcome_samples, 10))
        sens_upper = float(np.percentile(outcome_samples, 90))

        return PredictionResults(
            point_estimate=point_estimate,
            confidence_interval=ConfidenceInterval(
                lower=ci_lower,
                upper=ci_upper,
                confidence_level=0.95,
            ),
            sensitivity_range=SensitivityRange(
                optimistic=sens_upper,
                pessimistic=sens_lower,
                explanation="Range accounts for uncertainty in model parameters and external factors",
            ),
        )

    def _analyze_uncertainty(
        self, request: CounterfactualRequest, samples: Dict[str, np.ndarray]
    ) -> UncertaintyBreakdown:
        """
        Classify the OVERALL uncertainty of the outcome from its Monte Carlo
        samples' coefficient of variation.

        F3a/F3c (A3, 2026-07-22): the former per-factor `sources` breakdown is
        OMITTED. It labeled each exogenous input's OWN variance as its "impact"
        on the outcome — a leverage-blind fabrication (a high-variance input with
        a near-zero coefficient outranked a low-variance input with a large one),
        so the ranking it published inverted the true sensitivity. Only the
        honest overall level (from the OUTCOME distribution) remains. A real
        variance-decomposition is a modeling roadmap item.

        Args:
            request: Counterfactual request
            samples: Monte Carlo samples

        Returns:
            UncertaintyBreakdown with the overall level only
        """
        outcome_samples = samples[request.outcome]
        mean_out = float(np.mean(outcome_samples))
        std_out = float(np.std(outcome_samples))

        # Overall level from the coefficient of variation of the OUTCOME samples.
        # F3b (A3, 2026-07-22): when the mean is 0 the CV is UNDEFINED. The old
        # code substituted cv=0 and reported LOW uncertainty — claiming high
        # confidence for a value it could not actually assess. A genuinely
        # degenerate outcome (std==0, e.g. a fully-intervened deterministic model)
        # is legitimately zero-uncertainty (LOW); a non-zero spread around a zero
        # mean has an undefined CV and must NOT be reported as LOW — fail safe to
        # HIGH rather than fabricate confidence.
        if mean_out != 0:
            cv = std_out / abs(mean_out)
            if cv < 0.1:
                overall = UncertaintyLevel.LOW
            elif cv < 0.3:
                overall = UncertaintyLevel.MEDIUM
            else:
                overall = UncertaintyLevel.HIGH
        else:
            overall = UncertaintyLevel.LOW if std_out == 0 else UncertaintyLevel.HIGH

        return UncertaintyBreakdown(overall=overall)
