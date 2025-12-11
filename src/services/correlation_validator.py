"""
Correlation Group Validator Service.

Validates correlation group specifications including:
- Correlation coefficient validity
- Positive semi-definite matrix check
- Factor reference validation
- Matrix construction from groups
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.models.requests import (
    CorrelationGroup,
    CorrelationMatrix,
    CorrelationValidationRequest,
)
from src.models.responses import (
    CorrelationMatrixAnalysis,
    CorrelationValidationResponse,
    CorrelationValidationWarning,
    ImpliedCorrelationMatrix,
    ValidatedCorrelationGroup,
)
from src.models.shared import GraphV1, NodeKind

logger = logging.getLogger(__name__)


class CorrelationValidator:
    """
    Service for validating factor correlation specifications.

    Handles:
    - Correlation group validation
    - Matrix positive semi-definite check
    - Factor reference validation against graph
    - Implied matrix construction from groups
    """

    # Constants
    HIGH_CORRELATION_THRESHOLD = 0.9
    PSD_TOLERANCE = -1e-10  # Allow small negative eigenvalues due to numerical precision
    CONDITION_WARNING_THRESHOLD = 100.0

    def validate(
        self, request: CorrelationValidationRequest
    ) -> CorrelationValidationResponse:
        """
        Validate correlation specification.

        Args:
            request: Correlation validation request

        Returns:
            CorrelationValidationResponse with validation results
        """
        warnings: List[CorrelationValidationWarning] = []
        errors: List[str] = []
        validated_groups: List[ValidatedCorrelationGroup] = []

        # Determine input mode
        if request.correlation_matrix is not None:
            # Matrix mode - validate directly
            matrix = np.array(request.correlation_matrix.matrix)
            factors = request.correlation_matrix.factors

            # Validate matrix
            matrix_analysis = self._analyze_matrix(
                matrix, request.check_positive_definite
            )

            implied_matrix = ImpliedCorrelationMatrix(
                factors=factors,
                matrix=matrix.tolist(),
            )

            # Check factor references in graph
            if request.graph:
                ref_warnings = self._validate_factor_references(
                    set(factors), request.graph
                )
                warnings.extend(ref_warnings)

        else:
            # Groups mode - validate groups and build matrix
            all_factors: Set[str] = set()
            factor_correlations: Dict[Tuple[str, str], float] = {}

            for group in request.correlation_groups:
                # Validate individual group
                group_result = self._validate_group(group)
                validated_groups.append(group_result)

                if group_result.is_valid:
                    # Collect factors and correlations
                    all_factors.update(group.factors)

                    # Store pairwise correlations
                    for i, f1 in enumerate(group.factors):
                        for f2 in group.factors[i + 1:]:
                            key = tuple(sorted([f1, f2]))
                            if key in factor_correlations:
                                if factor_correlations[key] != group.correlation:
                                    warnings.append(
                                        CorrelationValidationWarning(
                                            code="CONFLICTING_CORRELATION",
                                            message=(
                                                f"Factors {f1} and {f2} have different correlations "
                                                f"in different groups: {factor_correlations[key]} vs {group.correlation}"
                                            ),
                                            affected_groups=[group.group_id],
                                            affected_factors=[f1, f2],
                                        )
                                    )
                            else:
                                factor_correlations[key] = group.correlation

            # Build implied matrix from groups
            factors = sorted(all_factors)
            n = len(factors)

            if n > 0:
                matrix = np.eye(n)
                factor_idx = {f: i for i, f in enumerate(factors)}

                for (f1, f2), corr in factor_correlations.items():
                    i, j = factor_idx[f1], factor_idx[f2]
                    matrix[i, j] = corr
                    matrix[j, i] = corr

                implied_matrix = ImpliedCorrelationMatrix(
                    factors=factors,
                    matrix=matrix.tolist(),
                )

                # Analyze the matrix
                matrix_analysis = self._analyze_matrix(
                    matrix, request.check_positive_definite
                )
            else:
                implied_matrix = None
                matrix_analysis = None

            # Check factor references in graph
            if request.graph and all_factors:
                ref_warnings = self._validate_factor_references(
                    all_factors, request.graph
                )
                warnings.extend(ref_warnings)

        # Check for high correlations
        if implied_matrix:
            high_corr_warnings = self._check_high_correlations(implied_matrix)
            warnings.extend(high_corr_warnings)

        # Determine validity
        is_valid = True

        # Invalid if any groups are invalid
        if any(not g.is_valid for g in validated_groups):
            is_valid = False
            errors.append("One or more correlation groups have validation errors")

        # Invalid if matrix is not positive semi-definite
        if matrix_analysis and not matrix_analysis.is_positive_semi_definite:
            is_valid = False
            errors.append(
                "Correlation matrix is not positive semi-definite and cannot be "
                "used for sampling. Consider regularization."
            )

        return CorrelationValidationResponse(
            valid=is_valid,
            validated_groups=validated_groups,
            implied_matrix=implied_matrix,
            matrix_analysis=matrix_analysis,
            warnings=warnings,
            errors=errors,
        )

    def _validate_group(self, group: CorrelationGroup) -> ValidatedCorrelationGroup:
        """
        Validate a single correlation group.

        Args:
            group: Correlation group to validate

        Returns:
            ValidatedCorrelationGroup with validation results
        """
        issues = []

        # Check correlation value
        if group.correlation == 1.0 and len(group.factors) > 1:
            issues.append(
                "Perfect correlation (1.0) between distinct factors is unusual - "
                "consider if factors should be merged"
            )
        elif group.correlation == -1.0 and len(group.factors) > 2:
            issues.append(
                "Perfect negative correlation (-1.0) cannot exist among >2 factors "
                "in a valid correlation matrix"
            )

        # Check for too many factors with same correlation
        if len(group.factors) > 5 and abs(group.correlation) > 0.8:
            issues.append(
                f"High correlation ({group.correlation}) among {len(group.factors)} factors "
                "may cause numerical issues"
            )

        return ValidatedCorrelationGroup(
            group_id=group.group_id,
            factors=group.factors,
            correlation=group.correlation,
            is_valid=len(issues) == 0 or not any("cannot" in i for i in issues),
            issues=issues,
        )

    def _analyze_matrix(
        self, matrix: np.ndarray, check_psd: bool
    ) -> CorrelationMatrixAnalysis:
        """
        Analyze correlation matrix properties.

        Args:
            matrix: Numpy correlation matrix
            check_psd: Whether to check positive semi-definiteness

        Returns:
            CorrelationMatrixAnalysis with analysis results
        """
        is_psd = True
        min_eigenvalue = None
        condition_number = None
        suggested_regularization = None

        if check_psd:
            try:
                # Compute eigenvalues
                eigenvalues = np.linalg.eigvalsh(matrix)
                min_eigenvalue = float(np.min(eigenvalues))

                # Check PSD (allow small negative due to numerical precision)
                is_psd = min_eigenvalue >= self.PSD_TOLERANCE

                # Compute condition number
                max_eigenvalue = float(np.max(eigenvalues))
                if min_eigenvalue > 0:
                    condition_number = max_eigenvalue / min_eigenvalue
                else:
                    condition_number = float('inf')

                # Suggest regularization if not PSD
                if not is_psd:
                    # Suggest adding small value to diagonal to make PSD
                    suggested_regularization = abs(min_eigenvalue) + 0.01

            except np.linalg.LinAlgError as e:
                logger.warning(f"Matrix analysis failed: {e}")
                is_psd = False

        return CorrelationMatrixAnalysis(
            is_positive_semi_definite=is_psd,
            min_eigenvalue=round(min_eigenvalue, 6) if min_eigenvalue is not None else None,
            condition_number=round(condition_number, 2) if condition_number is not None and condition_number != float('inf') else None,
            suggested_regularization=round(suggested_regularization, 4) if suggested_regularization is not None else None,
        )

    def _validate_factor_references(
        self, factors: Set[str], graph: GraphV1
    ) -> List[CorrelationValidationWarning]:
        """
        Validate that factors reference nodes in the graph.

        Args:
            factors: Set of factor IDs
            graph: Graph to validate against

        Returns:
            List of warnings
        """
        warnings = []

        # Extract node IDs and factor nodes
        node_ids = {node.id for node in graph.nodes}
        factor_nodes = {
            node.id for node in graph.nodes if node.kind == NodeKind.FACTOR
        }

        missing = factors - node_ids
        if missing:
            warnings.append(
                CorrelationValidationWarning(
                    code="MISSING_FACTOR_NODES",
                    message=f"Factors not found in graph: {sorted(missing)}",
                    affected_factors=sorted(missing),
                )
            )

        non_factor = factors & node_ids - factor_nodes
        if non_factor:
            warnings.append(
                CorrelationValidationWarning(
                    code="NON_FACTOR_NODES",
                    message=(
                        f"Some factors reference non-FACTOR nodes: {sorted(non_factor)}. "
                        "Correlations typically apply to FACTOR (chance) nodes."
                    ),
                    affected_factors=sorted(non_factor),
                )
            )

        return warnings

    def _check_high_correlations(
        self, implied_matrix: ImpliedCorrelationMatrix
    ) -> List[CorrelationValidationWarning]:
        """
        Check for potentially problematic high correlations.

        Args:
            implied_matrix: The implied correlation matrix

        Returns:
            List of warnings
        """
        warnings = []
        factors = implied_matrix.factors
        matrix = implied_matrix.matrix
        n = len(factors)

        high_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(matrix[i][j])
                if corr >= self.HIGH_CORRELATION_THRESHOLD:
                    high_pairs.append((factors[i], factors[j], matrix[i][j]))

        if high_pairs:
            for f1, f2, corr in high_pairs:
                warnings.append(
                    CorrelationValidationWarning(
                        code="HIGH_CORRELATION",
                        message=(
                            f"High correlation ({corr:.2f}) between '{f1}' and '{f2}' "
                            "may indicate redundancy or multicollinearity"
                        ),
                        affected_factors=[f1, f2],
                    )
                )

        return warnings
