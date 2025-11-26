"""
Causal inference endpoints.

Provides endpoints for:
- Causal model validation (Y₀)
- Counterfactual analysis (FACET)
"""

import logging
import uuid

from fastapi import APIRouter, Header, HTTPException
from typing import Optional

from src.models.metadata import create_response_metadata
from src.models.requests import (
    BatchCounterfactualRequest,
    CausalValidationRequest,
    ConformalCounterfactualRequest,
    CounterfactualRequest,
    DiscoveryFromDataRequest,
    DiscoveryFromKnowledgeRequest,
    ExperimentRecommendationRequest,
    TransportabilityRequest,
    ValidationStrategyRequest,
)
from src.models.representation import FactorExtractionRequest
from src.models.sensitivity import SensitivityRequest, SensitivityReport
from src.models.responses import (
    BatchCounterfactualResponse,
    CausalValidationResponse,
    ConformalCounterfactualResponse,
    CounterfactualResponse,
    DiscoveryResponse,
    ErrorCode,
    ErrorResponse,
    ExperimentRecommendationResponse,
    TransportabilityResponse,
    ValidationStrategyResponse,
)
from src.services.advanced_validation_suggester import AdvancedValidationSuggester
from src.services.batch_counterfactual_engine import BatchCounterfactualEngine
from src.services.causal_discovery_engine import CausalDiscoveryEngine
from src.services.causal_transporter import CausalTransporter
from src.services.causal_validator import CausalValidator
from src.services.conformal_predictor import ConformalPredictor
from src.services.counterfactual_engine import CounterfactualEngine
from src.services.sensitivity_analyzer import EnhancedSensitivityAnalyzer
from src.services.sequential_optimizer import SequentialOptimizer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
causal_validator = CausalValidator()
counterfactual_engine = CounterfactualEngine()
batch_counterfactual_engine = BatchCounterfactualEngine()
causal_transporter = CausalTransporter()
conformal_predictor = ConformalPredictor(counterfactual_engine)
advanced_validation_suggester = AdvancedValidationSuggester()
causal_discovery_engine = CausalDiscoveryEngine()
sensitivity_analyzer = EnhancedSensitivityAnalyzer()
sequential_optimizer = SequentialOptimizer()

# Request size limits
MAX_DAG_NODES = 100
MAX_DATA_SAMPLES = 10000
MAX_DATA_VARIABLES = 50


def validate_dag_structure(dag_structure, request_id: str):
    """
    Validate DAG structure for common issues.

    Args:
        dag_structure: DAG structure from request
        request_id: Request ID for logging

    Raises:
        HTTPException: If validation fails
    """
    import networkx as nx

    # Check size limits
    if len(dag_structure.nodes) > MAX_DAG_NODES:
        raise HTTPException(
            status_code=400,
            detail=f"DAG too large: {len(dag_structure.nodes)} nodes exceeds limit of {MAX_DAG_NODES}"
        )

    if len(dag_structure.nodes) == 0:
        raise HTTPException(
            status_code=400,
            detail="DAG has no nodes"
        )

    # Convert to NetworkX and check for cycles
    dag = nx.DiGraph()
    dag.add_nodes_from(dag_structure.nodes)
    dag.add_edges_from(dag_structure.edges)

    if not nx.is_directed_acyclic_graph(dag):
        # Find a cycle to report
        try:
            cycle = nx.find_cycle(dag)
            cycle_str = " -> ".join([str(edge[0]) for edge in cycle])
            raise HTTPException(
                status_code=400,
                detail=f"Graph contains cycle: {cycle_str}. Please provide a valid DAG (Directed Acyclic Graph)."
            )
        except nx.NetworkXNoCycle:
            # Shouldn't happen, but handle gracefully
            raise HTTPException(
                status_code=400,
                detail="Graph is not a valid DAG (contains cycles)"
            )

    logger.info(
        "dag_validation_passed",
        extra={
            "request_id": request_id,
            "num_nodes": len(dag_structure.nodes),
            "num_edges": len(dag_structure.edges),
        }
    )

    return dag


@router.post(
    "/validate",
    response_model=CausalValidationResponse,
    summary="Validate causal model structure",
    description="""
    Validates whether a causal model (DAG) supports causal identification
    for a given treatment-outcome pair.

    Returns adjustment sets if identifiable, or specific issues if not.

    **Use when:** Building a decision model, before running scenarios.

    **Returns:**
    - `identifiable`: Valid adjustment sets provided (with method, formula, assumptions)
    - `uncertain`: Potential issues detected, clarification needed
    - `cannot_identify`: Fundamental structural problems (with reason and suggestions)
    - `degraded`: Advanced analysis failed, fallback assessment provided
    """,
    responses={
        200: {"description": "Validation completed successfully"},
        400: {"description": "Invalid input (e.g., empty DAG, node not found)"},
        500: {"description": "Internal computation error"},
    },
)
async def validate_causal_model(
    request: CausalValidationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> CausalValidationResponse:
    """
    Validate causal model for identifiability.

    Args:
        request: Causal validation request with DAG and variables
        x_request_id: Optional request ID for tracing

    Returns:
        CausalValidationResponse: Validation results with adjustment sets or issues
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "causal_validation_request",
            extra={
                "request_id": request_id,
                "treatment": request.treatment,
                "outcome": request.outcome,
                "num_nodes": len(request.dag.nodes),
                "num_edges": len(request.dag.edges),
            },
        )

        result = causal_validator.validate(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "causal_validation_completed",
            extra={
                "request_id": request_id,
                "status": result.status,
                "confidence": result.confidence,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("causal_validation_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to validate causal model. Check logs for details.",
        )


@router.post(
    "/counterfactual",
    response_model=CounterfactualResponse,
    summary="Perform counterfactual analysis",
    description="""
    Analyzes what would happen under a counterfactual intervention.

    Provides:
    - Point estimates and confidence intervals
    - Uncertainty breakdown by source
    - Robustness analysis
    - Critical assumptions

    **Use when:** Evaluating "what if" scenarios for decision making.
    """,
    responses={
        200: {"description": "Counterfactual analysis completed successfully"},
        400: {"description": "Invalid input (e.g., malformed structural model)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_counterfactual(
    request: CounterfactualRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> CounterfactualResponse:
    """
    Perform counterfactual analysis.

    Args:
        request: Counterfactual request with structural model and intervention
        x_request_id: Optional request ID for tracing

    Returns:
        CounterfactualResponse: Counterfactual predictions with uncertainty
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "counterfactual_request",
            extra={
                "request_id": request_id,
                "outcome": request.outcome,
                "intervention": request.intervention,
                "num_variables": len(request.model.variables),
            },
        )

        result = counterfactual_engine.analyze(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "counterfactual_completed",
            extra={
                "request_id": request_id,
                "point_estimate": result.prediction.point_estimate,
                "uncertainty": result.uncertainty.overall,
                "robustness": result.robustness.score,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("counterfactual_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to perform counterfactual analysis. Check logs for details.",
        )


@router.post(
    "/counterfactual/batch",
    response_model=BatchCounterfactualResponse,
    summary="Batch counterfactual analysis with interaction detection",
    description="""
    Analyzes multiple counterfactual scenarios in a single request with interaction detection.

    Provides:
    - Individual predictions for each scenario
    - Interaction analysis (synergistic/antagonistic effects)
    - Scenario comparison and ranking
    - Marginal gains analysis

    **Features:**
    - Shared exogenous samples for determinism across scenarios
    - Automatic detection of variable interactions
    - Efficient batch processing
    - Consistent uncertainty quantification

    **Use when:** Comparing multiple intervention strategies or testing parameter combinations.

    **Example Use Cases:**
    - Testing price AND quality changes together
    - Comparing aggressive vs conservative strategies
    - Detecting when interventions amplify or cancel each other
    """,
    responses={
        200: {"description": "Batch counterfactual analysis completed successfully"},
        400: {"description": "Invalid input (e.g., too few scenarios, malformed model)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_batch_counterfactual(
    request: BatchCounterfactualRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> BatchCounterfactualResponse:
    """
    Perform batch counterfactual analysis with interaction detection.

    Args:
        request: Batch counterfactual request with multiple scenarios
        x_request_id: Optional request ID for tracing

    Returns:
        BatchCounterfactualResponse: Results for all scenarios with interactions
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "batch_counterfactual_request",
            extra={
                "request_id": request_id,
                "num_scenarios": len(request.scenarios),
                "outcome": request.outcome,
                "analyze_interactions": request.analyze_interactions,
                "samples": request.samples,
            },
        )

        result = batch_counterfactual_engine.generate_batch_counterfactuals(
            request=request,
            request_id=request_id,
        )

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "batch_counterfactual_completed",
            extra={
                "request_id": request_id,
                "num_scenarios": len(result.scenarios),
                "best_outcome": result.comparison.best_outcome,
                "most_robust": result.comparison.most_robust,
                "num_interactions": len(result.interactions.pairwise) if result.interactions else 0,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "batch_counterfactual_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform batch counterfactual analysis. Check logs for details.",
        )


@router.post(
    "/counterfactual/conformal",
    response_model=ConformalCounterfactualResponse,
    summary="Conformal prediction for counterfactuals",
    description="""
    Generate counterfactual predictions with conformal prediction intervals.

    Provides finite-sample valid prediction intervals with guaranteed coverage,
    offering provable uncertainty quantification beyond standard Monte Carlo methods.

    **Key Features:**
    - Distribution-free (no parametric assumptions)
    - Finite-sample validity (not asymptotic)
    - Guaranteed coverage: P(Y ∈ interval) ≥ 1-α
    - Adaptive to local uncertainty
    - Comparison to Monte Carlo intervals

    **Use when:** You need rigorous uncertainty guarantees for counterfactual
    predictions, especially in high-stakes decisions where coverage matters.

    **Example Use Cases:**
    - Safety-critical predictions requiring provable bounds
    - Regulatory compliance needing formal guarantees
    - Scientific research requiring rigorous uncertainty quantification
    - Model validation and calibration assessment

    **Requirements:**
    - Calibration data: At least 10 historical observations (more is better)
    - Exchangeability: Calibration and test points from same distribution
    """,
    responses={
        200: {"description": "Conformal prediction completed successfully"},
        400: {"description": "Invalid input or insufficient calibration data"},
        500: {"description": "Internal computation error"},
    },
)
async def conformal_counterfactual_prediction(
    request: ConformalCounterfactualRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ConformalCounterfactualResponse:
    """
    Generate counterfactual with conformal prediction interval.

    Args:
        request: Conformal counterfactual request with calibration data
        x_request_id: Optional request ID for tracing

    Returns:
        ConformalCounterfactualResponse: Prediction with guaranteed coverage
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "conformal_counterfactual_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "confidence_level": request.confidence_level,
                "calibration_size": len(request.calibration_data) if request.calibration_data else 0,
            },
        )

        result = conformal_predictor.predict_with_conformal_interval(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "conformal_counterfactual_completed",
            extra={
                "request_id": request_id,
                "guaranteed_coverage": result.coverage_guarantee.guaranteed_coverage,
                "calibration_size": result.calibration_quality.calibration_size,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "conformal_counterfactual_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform conformal prediction. Check logs for details.",
        )


@router.post(
    "/transport",
    response_model=TransportabilityResponse,
    summary="Analyze causal effect transportability",
    description="""
    Determines whether a causal effect identified in a source domain
    can be validly transported to a target domain.

    Provides:
    - Transportability assessment (yes/no)
    - Transport formula if transportable
    - Required assumptions with testability
    - Robustness assessment
    - Suggestions if not transportable

    **Use when:** Deciding if findings from one context (e.g., UK market)
    apply to another context (e.g., Germany market).

    **Example Use Cases:**
    - Does price sensitivity learned in UK work in France?
    - Can clinical trial results from one population apply to another?
    - Will marketing effects from online channels work offline?
    """,
    responses={
        200: {"description": "Transportability analysis completed successfully"},
        400: {"description": "Invalid input (e.g., malformed DAG, missing nodes)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_transportability(
    request: TransportabilityRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> TransportabilityResponse:
    """
    Analyze transportability of causal effects between domains.

    Args:
        request: Transportability request with source/target domains
        x_request_id: Optional request ID for tracing

    Returns:
        TransportabilityResponse: Transportability analysis results
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "transportability_request",
            extra={
                "request_id": request_id,
                "source_domain": request.source_domain.name,
                "target_domain": request.target_domain.name,
                "treatment": request.treatment,
                "outcome": request.outcome,
            },
        )

        result = causal_transporter.analyze_transportability(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "transportability_completed",
            extra={
                "request_id": request_id,
                "transportable": result.transportable,
                "method": result.method,
                "robustness": result.robustness,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "transportability_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform transportability analysis. Check logs for details.",
        )


@router.post(
    "/validate/strategies",
    response_model=ValidationStrategyResponse,
    summary="Get complete adjustment strategies for identifiability",
    description="""
    Provides complete adjustment strategies for non-identifiable DAGs.

    Goes beyond simple validation to provide actionable suggestions on how to
    make a non-identifiable causal effect identifiable through:
    - Backdoor adjustment strategies
    - Frontdoor adjustment strategies
    - Instrumental variable strategies

    Each strategy specifies exactly which nodes and edges to add to the DAG
    to achieve identifiability, along with theoretical justification.

    **Features:**
    - Complete path analysis (backdoor, frontdoor, blocked paths)
    - Critical node identification
    - Ranked strategies by expected success
    - Theoretical basis for each strategy

    **Use when:** You have a non-identifiable DAG and need guidance on
    how to modify it or what additional data to collect.
    """,
    responses={
        200: {"description": "Strategy analysis completed successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def get_validation_strategies(
    request: ValidationStrategyRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ValidationStrategyResponse:
    """
    Get complete adjustment strategies for identifiability.

    Args:
        request: Validation strategy request with DAG structure
        x_request_id: Optional request ID for tracing

    Returns:
        ValidationStrategyResponse: Complete strategies and path analysis
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "validation_strategy_request",
            extra={
                "request_id": request_id,
                "treatment": request.treatment,
                "outcome": request.outcome,
                "num_nodes": len(request.dag.nodes),
            },
        )

        # Validate DAG structure (checks for cycles, size limits)
        dag = validate_dag_structure(request.dag, request_id)

        # Check that treatment and outcome are in the DAG
        if request.treatment not in dag.nodes():
            raise HTTPException(
                status_code=400,
                detail=f"Treatment variable '{request.treatment}' not found in DAG nodes"
            )

        if request.outcome not in dag.nodes():
            raise HTTPException(
                status_code=400,
                detail=f"Outcome variable '{request.outcome}' not found in DAG nodes"
            )

        # Get strategies and path analysis
        strategies = advanced_validation_suggester.suggest_adjustment_strategies(
            dag, request.treatment, request.outcome
        )
        path_analysis = advanced_validation_suggester.analyze_paths(
            dag, request.treatment, request.outcome
        )

        # Convert to response models
        from src.models.responses import AdjustmentStrategyDetail, PathAnalysisDetail, ExplanationMetadata
        from src.models.shared import ExplanationMetadata as ExplanationMetadataShared

        strategy_details = []
        for s in strategies:
            strategy_details.append(
                AdjustmentStrategyDetail(
                    strategy_type=s.type,
                    nodes_to_add=s.nodes_to_add,
                    edges_to_add=s.edges_to_add,
                    explanation=s.explanation,
                    theoretical_basis=s.theoretical_basis,
                    expected_identifiability=s.expected_identifiability,
                )
            )

        path_detail = PathAnalysisDetail(
            backdoor_paths=path_analysis.backdoor_paths,
            frontdoor_paths=path_analysis.frontdoor_paths,
            blocked_paths=path_analysis.blocked_paths,
            critical_nodes=path_analysis.critical_nodes,
        )

        # Create explanation
        if strategies:
            summary = f"Found {len(strategies)} adjustment strategies"
            reasoning = f"Best strategy: {strategies[0].explanation}"
        else:
            summary = "No adjustment strategies found - effect may already be identifiable"
            reasoning = "All paths are either blocked or direct"

        explanation = ExplanationMetadataShared(
            summary=summary,
            reasoning=reasoning,
            technical_basis="Path analysis and adjustment set identification",
            assumptions=["DAG structure is correct"],
        )

        result = ValidationStrategyResponse(
            strategies=strategy_details,
            path_analysis=path_detail,
            explanation=explanation,
        )

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "validation_strategy_completed",
            extra={
                "request_id": request_id,
                "num_strategies": len(strategies),
                "num_backdoor_paths": len(path_analysis.backdoor_paths),
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "validation_strategy_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate validation strategies. Check logs for details.",
        )


@router.post(
    "/discover/from-data",
    response_model=DiscoveryResponse,
    summary="Discover causal structure from data",
    description="""
    Automatically discovers DAG structures from observational data.

    Uses correlation-based structure learning to suggest plausible
    causal graphs based on statistical relationships in the data.

    **Features:**
    - Correlation-based edge detection
    - Prior knowledge integration (required/forbidden edges)
    - Confidence scoring for discovered structures
    - Automatic cycle detection and acyclicity enforcement

    **Use when:** You have observational data but no clear causal model,
    or want to validate your domain knowledge against data.

    **Requirements:**
    - At least 10 observations (more is better)
    - 2-50 variables
    - Data should be roughly stationary
    """,
    responses={
        200: {"description": "Discovery completed successfully"},
        400: {"description": "Invalid input or insufficient data"},
        500: {"description": "Internal computation error"},
    },
)
async def discover_from_data(
    request: DiscoveryFromDataRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> DiscoveryResponse:
    """
    Discover causal structure from data.

    Args:
        request: Discovery request with observational data
        x_request_id: Optional request ID for tracing

    Returns:
        DiscoveryResponse: Discovered DAG structures with confidence
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        # Validate request size
        if len(request.data) > MAX_DATA_SAMPLES:
            raise HTTPException(
                status_code=400,
                detail=f"Too many data samples: {len(request.data)} exceeds limit of {MAX_DATA_SAMPLES}"
            )

        if len(request.variable_names) > MAX_DATA_VARIABLES:
            raise HTTPException(
                status_code=400,
                detail=f"Too many variables: {len(request.variable_names)} exceeds limit of {MAX_DATA_VARIABLES}"
            )

        logger.info(
            "discovery_from_data_request",
            extra={
                "request_id": request_id,
                "num_samples": len(request.data),
                "num_variables": len(request.variable_names),
                "threshold": request.threshold,
            },
        )

        # Convert data to numpy array
        import numpy as np
        data_array = np.array(request.data)

        # Discover DAG
        dag, confidence = causal_discovery_engine.discover_from_data(
            data=data_array,
            variable_names=request.variable_names,
            prior_knowledge=request.prior_knowledge,
            threshold=request.threshold,
            seed=request.seed,
        )

        # Convert to response
        from src.models.responses import DiscoveredDAG, ExplanationMetadata
        from src.models.shared import ExplanationMetadata as ExplanationMetadataShared

        discovered_dag = DiscoveredDAG(
            nodes=list(dag.nodes()),
            edges=list(dag.edges()),
            confidence=confidence,
            method="correlation",
        )

        explanation = ExplanationMetadataShared(
            summary=f"Discovered DAG structure from data with {confidence:.0%} confidence",
            reasoning=f"Found {len(dag.edges())} edges using correlation threshold {request.threshold}",
            technical_basis="Correlation-based structure learning",
            assumptions=["Linear relationships", "No hidden confounders", "Stationarity"],
        )

        result = DiscoveryResponse(
            discovered_dags=[discovered_dag],
            explanation=explanation,
        )

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "discovery_from_data_completed",
            extra={
                "request_id": request_id,
                "num_edges": len(dag.edges()),
                "confidence": confidence,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "discovery_from_data_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to discover causal structure from data. Check logs for details.",
        )


@router.post(
    "/discover/from-knowledge",
    response_model=DiscoveryResponse,
    summary="Discover causal structure from domain knowledge",
    description="""
    Suggests DAG structures based on domain knowledge description.

    Uses heuristics and pattern matching to generate plausible causal
    structures from natural language descriptions.

    **Features:**
    - Multiple candidate DAGs ranked by plausibility
    - Prior knowledge integration
    - Common causal patterns (chains, forks, colliders)

    **Use when:** You have domain knowledge but need help formalizing
    it into a DAG structure.

    **Example:** "Price affects revenue, quality affects both price and
    revenue" → generates candidate DAGs
    """,
    responses={
        200: {"description": "Discovery completed successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def discover_from_knowledge(
    request: DiscoveryFromKnowledgeRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> DiscoveryResponse:
    """
    Discover causal structure from domain knowledge.

    Args:
        request: Discovery request with domain description
        x_request_id: Optional request ID for tracing

    Returns:
        DiscoveryResponse: Candidate DAG structures with confidence
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "discovery_from_knowledge_request",
            extra={
                "request_id": request_id,
                "num_variables": len(request.variable_names),
                "top_k": request.top_k,
            },
        )

        # Discover DAGs from knowledge
        dags = causal_discovery_engine.discover_from_knowledge(
            domain_description=request.domain_description,
            variable_names=request.variable_names,
            prior_knowledge=request.prior_knowledge,
            top_k=request.top_k,
        )

        # Convert to response
        from src.models.responses import DiscoveredDAG, ExplanationMetadata
        from src.models.shared import ExplanationMetadata as ExplanationMetadataShared

        discovered_dags = []
        for dag, confidence in dags:
            discovered_dags.append(
                DiscoveredDAG(
                    nodes=list(dag.nodes()),
                    edges=list(dag.edges()),
                    confidence=confidence,
                    method="knowledge",
                )
            )

        explanation = ExplanationMetadataShared(
            summary=f"Generated {len(discovered_dags)} candidate DAG structures from domain knowledge",
            reasoning="Used heuristic patterns to match domain description",
            technical_basis="Knowledge-guided structure generation with common causal patterns",
            assumptions=["Domain description is accurate"],
        )

        result = DiscoveryResponse(
            discovered_dags=discovered_dags,
            explanation=explanation,
        )

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "discovery_from_knowledge_completed",
            extra={
                "request_id": request_id,
                "num_dags": len(discovered_dags),
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "discovery_from_knowledge_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to discover causal structure from knowledge. Check logs for details.",
        )


@router.post(
    "/experiment/recommend",
    response_model=ExperimentRecommendationResponse,
    summary="Recommend next experiment using Thompson sampling",
    description="""
    Recommends the next experiment to run using Thompson sampling.

    Balances exploration (learning about parameters) with exploitation
    (optimizing the outcome) to maximize long-term value of experimentation.

    **Features:**
    - Thompson sampling for optimal exploration/exploitation
    - Bayesian belief updating
    - Information gain estimation
    - Cost-benefit analysis
    - Rationale generation

    **Use when:** Running sequential experiments where you want to
    learn efficiently while also optimizing outcomes.

    **Example Use Cases:**
    - A/B testing campaigns
    - Clinical trial design
    - Product optimization experiments
    - Scientific inquiry with limited budget
    """,
    responses={
        200: {"description": "Recommendation generated successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def recommend_experiment(
    request: ExperimentRecommendationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ExperimentRecommendationResponse:
    """
    Recommend next experiment using Thompson sampling.

    Args:
        request: Experiment recommendation request with beliefs and constraints
        x_request_id: Optional request ID for tracing

    Returns:
        ExperimentRecommendationResponse: Recommended experiment with rationale
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "experiment_recommendation_request",
            extra={
                "request_id": request_id,
                "num_beliefs": len(request.beliefs),
                "budget": request.constraints.budget,
                "time_horizon": request.constraints.time_horizon,
            },
        )

        # Convert request models to service models
        from src.services.sequential_optimizer import (
            BeliefState,
            OptimizationObjective,
            ExperimentConstraints,
        )

        # Build parameter distributions
        parameter_distributions = {}
        for belief in request.beliefs:
            parameter_distributions[belief.parameter_name] = {
                "type": belief.distribution_type,
                **belief.parameters,
            }

        beliefs = BeliefState(parameter_distributions)

        objective = OptimizationObjective(
            target_variable=request.objective.target_variable,
            goal=request.objective.goal,
            target_value=request.objective.target_value,
        )

        constraints = ExperimentConstraints(
            budget=request.constraints.budget,
            time_horizon=request.constraints.time_horizon,
            feasible_interventions=request.constraints.feasible_interventions,
        )

        # Convert history if present
        history = []
        if request.history:
            for h in request.history:
                history.append({
                    "intervention": h.intervention,
                    "outcome": h.outcome,
                    "cost": h.cost,
                })

        # Get recommendation
        recommendation = sequential_optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            history=history if history else None,
            seed=request.seed,
        )

        # Convert to response
        from src.models.responses import RecommendedExperimentDetail, ExplanationMetadata
        from src.models.shared import ExplanationMetadata as ExplanationMetadataShared

        rec_detail = RecommendedExperimentDetail(
            intervention=recommendation.intervention,
            expected_outcome=recommendation.expected_outcome,
            expected_information_gain=recommendation.expected_information_gain,
            cost_estimate=recommendation.cost_estimate,
            rationale=recommendation.rationale,
            exploration_vs_exploitation=recommendation.exploration_vs_exploitation,
        )

        exploration_type = "explore" if recommendation.exploration_vs_exploitation > 0.5 else "exploit"
        explanation = ExplanationMetadataShared(
            summary=f"Recommend {exploration_type} strategy: {recommendation.rationale}",
            reasoning=f"Information gain: {recommendation.expected_information_gain:.2f}, Cost: {recommendation.cost_estimate:.0f}",
            technical_basis="Thompson sampling with 100 posterior samples",
            assumptions=["Parameter beliefs accurate", "Cost estimates reasonable"],
        )

        result = ExperimentRecommendationResponse(
            recommendation=rec_detail,
            explanation=explanation,
        )

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "experiment_recommendation_completed",
            extra={
                "request_id": request_id,
                "information_gain": recommendation.expected_information_gain,
                "exploration_score": recommendation.exploration_vs_exploitation,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "experiment_recommendation_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to recommend experiment. Check logs for details.",
        )


@router.post(
    "/sensitivity/detailed",
    response_model=SensitivityReport,
    summary="Detailed sensitivity analysis for causal assumptions",
    description="""
    Quantifies how sensitive causal estimates are to assumption violations.

    Moves beyond discrete robustness categories (robust/moderate/fragile)
    to continuous sensitivity metrics with elasticity calculations.

    **Features:**
    - Quantitative sensitivity metrics (elasticity, outcome ranges)
    - Critical assumption identification
    - Elasticity: % change in outcome per % change in assumption violation
    - Robustness scores (0=fragile, 1=robust)
    - Recommendations for strengthening assumptions

    **Tested Assumptions:**
    - No unobserved confounding
    - Linear effects
    - No selection bias
    - Causal sufficiency
    - Positivity
    - Consistency

    **Use when:** You need to understand which assumptions matter most
    for your causal estimates and how robust your results are.

    **Example:** "If unobserved confounding increases by 10%, how much
    does my outcome estimate change?"
    """,
    responses={
        200: {"description": "Sensitivity analysis completed successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_detailed_sensitivity(
    request: SensitivityRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> SensitivityReport:
    """
    Analyze sensitivity to causal assumptions.

    Args:
        request: Sensitivity analysis request with model and assumptions
        x_request_id: Optional request ID for tracing

    Returns:
        SensitivityReport: Detailed sensitivity metrics for each assumption
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "sensitivity_analysis_request",
            extra={
                "request_id": request_id,
                "num_assumptions": len(request.assumptions),
                "num_violation_levels": len(request.violation_levels),
            },
        )

        # Perform sensitivity analysis
        result = sensitivity_analyzer.analyze_assumption_sensitivity(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "sensitivity_analysis_completed",
            extra={
                "request_id": request_id,
                "overall_robustness": result.overall_robustness_score,
                "num_critical": len(result.most_critical),
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "sensitivity_analysis_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform sensitivity analysis. Check logs for details.",
        )


@router.post(
    "/extract-factors",
    summary="Extract causal factors from unstructured data",
    description="""
    Extract latent causal factors from unstructured text data using representation learning.

    Uses text embedding and clustering to identify themes/factors in:
    - Support tickets
    - Customer reviews
    - User feedback
    - Survey responses
    - Error logs

    **Features:**
    - Automatic factor identification
    - Keyword extraction per factor  
    - Suggested DAG structure
    - Factor prevalence and strength scores

    **Use when:** You have qualitative data and want to identify potential
    causal factors to include in your model.

    **Example:** Support tickets → ["usability_issues", "performance_problems"]
    → suggested DAG with these as causes of churn
    """,
    responses={
        200: {"description": "Factor extraction completed successfully"},
        400: {"description": "Invalid input (e.g., not enough data)"},
        500: {"description": "Internal computation error"},
    },
)
async def extract_causal_factors(
    request: FactorExtractionRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    """
    Extract causal factors from unstructured data.

    Args:
        request: Factor extraction request with text data
        x_request_id: Optional request ID for tracing

    Returns:
        ExtractedFactors: Identified factors with suggested DAG
    """
    from src.models.representation import ExtractedFactors, CausalFactor
    from src.services.causal_representation_learner import CausalRepresentationLearner

    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    # Initialize learner (could be module-level singleton)
    learner = CausalRepresentationLearner()

    try:
        logger.info(
            "factor_extraction_request",
            extra={
                "request_id": request_id,
                "data_type": request.data_type,
                "num_texts": len(request.texts),
                "n_factors": request.n_factors,
            },
        )

        # Extract factors
        factors_data, suggested_dag, confidence = learner.extract_factors(
            data=request.texts,
            domain_hints=[request.domain] if request.domain else None,
            n_factors=request.n_factors,
            min_cluster_size=request.min_cluster_size,
            outcome_variable=request.outcome_variable
        )

        # Convert to response models
        factors = [CausalFactor(**f) for f in factors_data]

        # Convert DAG to DAGStructure if present
        dag_structure = None
        if suggested_dag:
            from src.models.shared import DAGStructure
            dag_structure = DAGStructure(
                nodes=suggested_dag["nodes"],
                edges=[tuple(e) for e in suggested_dag["edges"]]
            )

        # Create summary
        factor_names = [f.name for f in factors[:3]]
        summary = f"Identified {len(factors)} factors from {len(request.texts)} texts"
        if factors:
            summary += f": {', '.join(factor_names)}"
            if len(factors) > 3:
                summary += f" and {len(factors) - 3} more"

        result = ExtractedFactors(
            factors=factors,
            suggested_dag=dag_structure,
            confidence=confidence,
            method="Sentence embedding + clustering",
            summary=summary
        )

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "factor_extraction_completed",
            extra={
                "request_id": request_id,
                "num_factors": len(factors),
                "confidence": confidence,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "factor_extraction_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to extract causal factors. Check logs for details.",
        )
