# Quarantined Test Manifest

**Generated:** 2026-02-27
**Total quarantined:** 554 tests (524 FAILED + 30 ERROR)
**Mechanism:** Tests listed in `tests/_quarantined/known_failures.txt` are auto-skipped
via the `pytest_collection_modifyitems` hook in `tests/conftest.py`. They receive the
`@pytest.mark.quarantined` marker and are skipped with the reason
"Quarantined: pre-existing failure". To run only quarantined tests: `pytest -m quarantined`.

These are pre-existing failures that predate current development work. They are
quarantined so that the CI/pre-push gate can enforce a green bar on all
non-quarantined tests. Each test listed here should be investigated and either
fixed or removed in a dedicated cleanup effort.

---

## Summary by Category

| Category | Files | Tests | Failure Reason |
|----------|------:|------:|----------------|
| Integration: endpoint tests (TestClient/ASGI) | 28 | 411 | HTTP 500/422/assertion errors or require running server |
| Smoke: production health checks | 1 | 14 | `pytest.ini` injects `--base-url`/`--api-key` CLI args; collection ERROR when run from root |
| Unit: service implementation mismatches | 17 | 93 | StructuralModel API drift, mock mismatches, fixture errors, assertion drift |
| Unit: duplicate files (accidental copies) | 6 | 32 | Files with space in name (`"file 2.py"`, `"file 3.py"`); duplicates of originals |
| Property: weight normalization | 1 | 4 | Hypothesis-based tests with service API drift |
| **TOTAL** | **53** | **554** | |

---

## Category 1: Integration Tests

411 tests across 28 files. These tests use FastAPI's `TestClient`, `httpx.AsyncClient`
with `ASGITransport`, or `httpx.AsyncClient` pointing at `localhost:8000`. Failures
stem from HTTP 500 (unhandled server errors), HTTP 422 (validation schema drift),
assertion mismatches where response schemas evolved, or requiring a running server
instance that is not available in CI.

| File | Tests |
|------|------:|
| `tests/integration/test_p2_verification.py` | 43 |
| `tests/integration/test_phase2_schema_endpoints.py` | 32 |
| `tests/integration/test_threshold_endpoint.py` | 27 |
| `tests/integration/test_dominance_endpoint.py` | 26 |
| `tests/integration/test_risk_adjustment_endpoint.py` | 23 |
| `tests/integration/test_parameter_recommendations.py` | 22 |
| `tests/integration/test_optimise_endpoint.py` | 22 |
| `tests/integration/test_phase4_endpoints.py` | 21 |
| `tests/integration/test_phase1_critical_endpoints.py` | 19 |
| `tests/integration/test_transportability_endpoints.py` | 17 |
| `tests/integration/test_validation_strategy_endpoints.py` | 14 |
| `tests/integration/test_multi_criteria_endpoint.py` | 14 |
| `tests/integration/test_contracts.py` | 14 |
| `tests/integration/test_cee_workflows.py` | 14 |
| `tests/integration/test_teaching_endpoint.py` | 13 |
| `tests/integration/test_error_recovery_integration.py` | 13 |
| `tests/integration/test_tae_workflows.py` | 11 |
| `tests/integration/test_discovery_and_optimization_endpoints.py` | 11 |
| `tests/integration/test_plot_workflows.py` | 10 |
| `tests/integration/test_contrastive_endpoints.py` | 10 |
| `tests/integration/test_phase3_e2e.py` | 9 |
| `tests/integration/test_conformal_endpoints.py` | 8 |
| `tests/integration/test_team_endpoint.py` | 4 |
| `tests/integration/test_production_excellence.py` | 4 |
| `tests/integration/test_batch_counterfactual_endpoints.py` | 4 |
| `tests/integration/test_observability_contract.py` | 3 |
| `tests/integration/test_error_schema_contract.py` | 2 |
| `tests/integration/test_causal_endpoints.py` | 1 |

### Detailed test list

#### `tests/integration/test_batch_counterfactual_endpoints.py` (4 tests)

```
tests/integration/test_batch_counterfactual_endpoints.py::test_batch_counterfactual_complex_model
tests/integration/test_batch_counterfactual_endpoints.py::test_batch_counterfactual_multiple_interventions
tests/integration/test_batch_counterfactual_endpoints.py::test_batch_counterfactual_no_interactions
tests/integration/test_batch_counterfactual_endpoints.py::test_batch_counterfactual_with_interactions
```

#### `tests/integration/test_causal_endpoints.py` (1 tests)

```
tests/integration/test_causal_endpoints.py::test_causal_validation_pricing_scenario
```

#### `tests/integration/test_cee_workflows.py` (14 tests)

```
tests/integration/test_cee_workflows.py::test_contrastive_for_critique
tests/integration/test_cee_workflows.py::test_enhanced_explanations_integration
tests/integration/test_cee_workflows.py::test_error_messages_for_critique
tests/integration/test_cee_workflows.py::test_explanation_levels_for_different_users
tests/integration/test_cee_workflows.py::test_extract_factors_from_text
tests/integration/test_cee_workflows.py::test_extract_validation_issues
tests/integration/test_cee_workflows.py::test_flag_critical_assumptions
tests/integration/test_cee_workflows.py::test_metadata_for_citations
tests/integration/test_cee_workflows.py::test_multiple_counterfactuals
tests/integration/test_cee_workflows.py::test_progressive_disclosure
tests/integration/test_cee_workflows.py::test_readability_validation
tests/integration/test_cee_workflows.py::test_real_time_validation_performance
tests/integration/test_cee_workflows.py::test_sensitivity_for_review
tests/integration/test_cee_workflows.py::test_validation_for_critique
```

#### `tests/integration/test_conformal_endpoints.py` (8 tests)

```
tests/integration/test_conformal_endpoints.py::TestConformalCalibrationMetrics::test_calibration_size_reasonable
tests/integration/test_conformal_endpoints.py::TestConformalCalibrationMetrics::test_residual_statistics_complete
tests/integration/test_conformal_endpoints.py::TestConformalComparison::test_width_ratio_present
tests/integration/test_conformal_endpoints.py::TestConformalConfidenceLevels::test_higher_confidence_wider_intervals
tests/integration/test_conformal_endpoints.py::TestConformalEndpointBasic::test_conformal_endpoint_has_coverage_guarantee
tests/integration/test_conformal_endpoints.py::TestConformalEndpointBasic::test_conformal_endpoint_has_prediction_interval
tests/integration/test_conformal_endpoints.py::TestConformalMetadata::test_metadata_present
tests/integration/test_conformal_endpoints.py::TestConformalValidation::test_insufficient_calibration_data
```

#### `tests/integration/test_contracts.py` (14 tests)

```
tests/integration/test_contracts.py::test_all_responses_include_metadata
tests/integration/test_contracts.py::test_batch_response_schema - http...
tests/integration/test_contracts.py::test_concurrent_requests_performance
tests/integration/test_contracts.py::test_counterfactual_response_schema
tests/integration/test_contracts.py::test_deprecated_fields_still_work
tests/integration/test_contracts.py::test_determinism_with_seeds - htt...
tests/integration/test_contracts.py::test_different_seeds_may_differ
tests/integration/test_contracts.py::test_error_responses_valid - http...
tests/integration/test_contracts.py::test_invalid_field_types_rejected
tests/integration/test_contracts.py::test_metadata_includes_version - ...
tests/integration/test_contracts.py::test_missing_required_fields_rejected
tests/integration/test_contracts.py::test_performance_under_5s - httpx...
tests/integration/test_contracts.py::test_validation_errors_descriptive
tests/integration/test_contracts.py::test_validation_response_schema
```

#### `tests/integration/test_contrastive_endpoints.py` (10 tests)

```
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_basic
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_complex_model
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_deterministic
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_explanation_quality
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_metadata
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_multi_variable_combinations
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_multiple_variables
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_ranking_by_cost
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_respects_fixed_constraints
tests/integration/test_contrastive_endpoints.py::test_contrastive_explanation_with_bounds
```

#### `tests/integration/test_discovery_and_optimization_endpoints.py` (11 tests)

```
tests/integration/test_discovery_and_optimization_endpoints.py::TestDiscoveryFromDataEndpoint::test_insufficient_data
tests/integration/test_discovery_and_optimization_endpoints.py::TestDiscoveryFromKnowledgeEndpoint::test_dag_structure
tests/integration/test_discovery_and_optimization_endpoints.py::TestDiscoveryFromKnowledgeEndpoint::test_minimal_description
tests/integration/test_discovery_and_optimization_endpoints.py::TestDiscoveryFromKnowledgeEndpoint::test_prior_knowledge_applied
tests/integration/test_discovery_and_optimization_endpoints.py::TestDiscoveryFromKnowledgeEndpoint::test_simple_knowledge_discovery
tests/integration/test_discovery_and_optimization_endpoints.py::TestDiscoveryFromKnowledgeEndpoint::test_top_k_parameter
tests/integration/test_discovery_and_optimization_endpoints.py::TestExperimentRecommendationEndpoint::test_explanation_structure
tests/integration/test_discovery_and_optimization_endpoints.py::TestExperimentRecommendationEndpoint::test_exploration_exploitation_range
tests/integration/test_discovery_and_optimization_endpoints.py::TestExperimentRecommendationEndpoint::test_information_gain_range
tests/integration/test_discovery_and_optimization_endpoints.py::TestExperimentRecommendationEndpoint::test_metadata_included
tests/integration/test_discovery_and_optimization_endpoints.py::TestExperimentRecommendationEndpoint::test_multiple_beliefs
```

#### `tests/integration/test_dominance_endpoint.py` (26 tests)

```
tests/integration/test_dominance_endpoint.py::test_clear_dominance_simple
tests/integration/test_dominance_endpoint.py::test_dominance_degree - ...
tests/integration/test_dominance_endpoint.py::test_dominated_relation_fields
tests/integration/test_dominance_endpoint.py::test_equal_scores_no_dominance
tests/integration/test_dominance_endpoint.py::test_extra_criterion_validation
tests/integration/test_dominance_endpoint.py::test_hundred_options_maximum
tests/integration/test_dominance_endpoint.py::test_maximum_options_validation
tests/integration/test_dominance_endpoint.py::test_minimum_options_validation
tests/integration/test_dominance_endpoint.py::test_missing_criterion_validation
tests/integration/test_dominance_endpoint.py::test_no_dominance_all_pareto
tests/integration/test_dominance_endpoint.py::test_pareto_all_on_frontier
tests/integration/test_dominance_endpoint.py::test_pareto_clear_domination
tests/integration/test_dominance_endpoint.py::test_pareto_default_max_frontier_size
tests/integration/test_dominance_endpoint.py::test_pareto_frontier_basic
tests/integration/test_dominance_endpoint.py::test_pareto_max_frontier_size
tests/integration/test_dominance_endpoint.py::test_pareto_request_id_tracking
tests/integration/test_dominance_endpoint.py::test_pareto_response_completeness
tests/integration/test_dominance_endpoint.py::test_pareto_single_criterion
tests/integration/test_dominance_endpoint.py::test_pareto_validation_maximum_options
tests/integration/test_dominance_endpoint.py::test_pareto_validation_minimum_options
tests/integration/test_dominance_endpoint.py::test_partial_dominance
tests/integration/test_dominance_endpoint.py::test_request_id_tracking
tests/integration/test_dominance_endpoint.py::test_response_fields_completeness
tests/integration/test_dominance_endpoint.py::test_score_range_validation
tests/integration/test_dominance_endpoint.py::test_single_criterion - ...
tests/integration/test_dominance_endpoint.py::test_two_options_minimum
```

#### `tests/integration/test_error_recovery_integration.py` (13 tests)

```
tests/integration/test_error_recovery_integration.py::TestCausalDiscoveryRecovery::test_advanced_discovery_fallback_on_error
tests/integration/test_error_recovery_integration.py::TestCausalDiscoveryRecovery::test_circuit_breaker_prevents_repeated_failures
tests/integration/test_error_recovery_integration.py::TestConformalPredictorRecovery::test_degraded_conformal_with_small_calibration
tests/integration/test_error_recovery_integration.py::TestConformalPredictorRecovery::test_fallback_to_monte_carlo_with_few_calibration_points
tests/integration/test_error_recovery_integration.py::TestConformalPredictorRecovery::test_normal_conformal_with_sufficient_calibration
tests/integration/test_error_recovery_integration.py::TestEndToEndRecovery::test_complete_fallback_chain_conformal
tests/integration/test_error_recovery_integration.py::TestEndToEndRecovery::test_complete_fallback_chain_discovery
tests/integration/test_error_recovery_integration.py::TestEndToEndRecovery::test_complete_fallback_chain_validation
tests/integration/test_error_recovery_integration.py::TestHealthMonitoring::test_health_monitor_tracks_causal_discovery
tests/integration/test_error_recovery_integration.py::TestHealthMonitoring::test_health_monitor_tracks_conformal_prediction
tests/integration/test_error_recovery_integration.py::TestHealthMonitoring::test_health_monitor_tracks_validation_suggester
tests/integration/test_error_recovery_integration.py::TestValidationSuggesterRecovery::test_path_analysis_with_circuit_breaker
tests/integration/test_error_recovery_integration.py::TestValidationSuggesterRecovery::test_strategy_generation_with_circuit_breaker
```

#### `tests/integration/test_error_schema_contract.py` (2 tests)

```
tests/integration/test_error_schema_contract.py::test_error_response_backward_compatible
tests/integration/test_error_schema_contract.py::test_not_found_error_schema
```

#### `tests/integration/test_multi_criteria_endpoint.py` (14 tests)

```
tests/integration/test_multi_criteria_endpoint.py::test_algorithm_comparison_same_input
tests/integration/test_multi_criteria_endpoint.py::test_normalized_weights_no_warning
tests/integration/test_multi_criteria_endpoint.py::test_request_id_tracking
tests/integration/test_multi_criteria_endpoint.py::test_response_metadata
tests/integration/test_multi_criteria_endpoint.py::test_response_structure_completeness
tests/integration/test_multi_criteria_endpoint.py::test_single_criterion_edge_case
tests/integration/test_multi_criteria_endpoint.py::test_trade_off_detection
tests/integration/test_multi_criteria_endpoint.py::test_trade_off_threshold_sensitivity
tests/integration/test_multi_criteria_endpoint.py::test_uniform_scores_edge_case
tests/integration/test_multi_criteria_endpoint.py::test_weight_normalization_warning
tests/integration/test_multi_criteria_endpoint.py::test_weighted_product_basic
tests/integration/test_multi_criteria_endpoint.py::test_weighted_product_zero_score_handling
tests/integration/test_multi_criteria_endpoint.py::test_weighted_sum_basic
tests/integration/test_multi_criteria_endpoint.py::test_weighted_sum_percentile_selection
```

#### `tests/integration/test_observability_contract.py` (3 tests)

```
tests/integration/test_observability_contract.py::TestHeaderPropagation::test_content_type_json_for_api_responses
tests/integration/test_observability_contract.py::TestHeaderPropagation::test_error_responses_json_content_type
tests/integration/test_observability_contract.py::TestSentryIntegration::test_error_structure_captures_context
```

#### `tests/integration/test_optimise_endpoint.py` (22 tests)

```
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointBasic::test_optimize_single_variable_maximize
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointBasic::test_optimize_single_variable_minimize
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointBasic::test_optimize_two_variables
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointConfidenceIntervals::test_confidence_interval_included
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointConfidenceIntervals::test_different_confidence_levels
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointConstraints::test_greater_than_constraint
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointConstraints::test_less_than_constraint
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointConstraints::test_multiple_constraints
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointConstraints::test_no_feasible_solution
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointGridMetrics::test_grid_metrics_reported
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointGridMetrics::test_two_variable_grid_size
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointMetadata::test_request_id_in_metadata
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointPerformance::test_performance_under_2_seconds
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointReproducibility::test_same_seed_same_result
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointSensitivityAnalysis::test_critical_variables_identified
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointSensitivityAnalysis::test_sensitivity_analysis_included
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointValidation::test_empty_coefficients_rejected
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointValidation::test_invalid_direction_rejected
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointValidation::test_too_few_grid_points_rejected
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointValidation::test_upper_bound_less_than_lower_rejected
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointWarnings::test_boundary_optimum_warning
tests/integration/test_optimise_endpoint.py::TestOptimiseEndpointWarnings::test_flat_objective_warning
```

#### `tests/integration/test_p2_verification.py` (43 tests)

```
tests/integration/test_p2_verification.py::TestCILF3Unified422Format::test_non_robustness_422_unchanged
tests/integration/test_p2_verification.py::TestCILF3Unified422Format::test_pydantic_422_has_analysis_status_blocked
tests/integration/test_p2_verification.py::TestCILF3Unified422Format::test_pydantic_422_has_critiques
tests/integration/test_p2_verification.py::TestCILF3Unified422Format::test_pydantic_422_has_request_id_from_header
tests/integration/test_p2_verification.py::TestCILF3Unified422Format::test_pydantic_422_has_status_reason
tests/integration/test_p2_verification.py::TestCILF3Unified422Format::test_pydantic_422_matches_custom_validator_shape
tests/integration/test_p2_verification.py::TestCILF3Unified422Format::test_pydantic_422_no_olumi_fields
tests/integration/test_p2_verification.py::TestExistsProbabilityOptional::test_boundary_exists_probability_one
tests/integration/test_p2_verification.py::TestExistsProbabilityOptional::test_boundary_exists_probability_zero_schema_valid
tests/integration/test_p2_verification.py::TestExistsProbabilityOptional::test_edge_without_exists_probability_accepted
tests/integration/test_p2_verification.py::TestExistsProbabilityOptional::test_explicit_exists_probability_honored
tests/integration/test_p2_verification.py::TestFactorConfidence::test_confidence_absent_without_bootstrap
tests/integration/test_p2_verification.py::TestFactorConfidence::test_confidence_determinism
tests/integration/test_p2_verification.py::TestFactorConfidence::test_confidence_populated_when_bootstrap_active
tests/integration/test_p2_verification.py::TestFactorConfidence::test_confidence_varies_across_factors
tests/integration/test_p2_verification.py::TestGoalThresholdProbabilityEndpoint::test_invalid_goal_threshold_rejected
tests/integration/test_p2_verification.py::TestGoalThresholdProbabilityEndpoint::test_probability_of_goal_absent_when_no_threshold
tests/integration/test_p2_verification.py::TestGoalThresholdProbabilityEndpoint::test_probability_of_goal_included_when_threshold_provided
tests/integration/test_p2_verification.py::TestGoalThresholdProbabilityEndpoint::test_probability_of_goal_values_correct_for_options
tests/integration/test_p2_verification.py::TestP2Task1EndpointExists::test_endpoint_exists
tests/integration/test_p2_verification.py::TestP2Task2V2ResponseFormat::test_response_has_analysis_status
tests/integration/test_p2_verification.py::TestP2Task2V2ResponseFormat::test_response_has_seed_used
tests/integration/test_p2_verification.py::TestP2Task2V2ResponseFormat::test_response_has_timestamp_iso8601
tests/integration/test_p2_verification.py::TestP2Task2V2ResponseFormat::test_response_has_version_field
tests/integration/test_p2_verification.py::TestP2Task2V2ResponseFormat::test_response_no_response_hash
tests/integration/test_p2_verification.py::TestP2Task3RequestIdTracing::test_processing_time_header_present
tests/integration/test_p2_verification.py::TestP2Task3RequestIdTracing::test_request_id_echoed_in_body
tests/integration/test_p2_verification.py::TestP2Task3RequestIdTracing::test_request_id_echoed_in_header
tests/integration/test_p2_verification.py::TestP2Task4UnwrappedError422::test_422_has_analysis_status_blocked
tests/integration/test_p2_verification.py::TestP2Task4UnwrappedError422::test_422_has_critiques_at_top_level
tests/integration/test_p2_verification.py::TestP2Task4UnwrappedError422::test_422_has_request_id
tests/integration/test_p2_verification.py::TestP2Task4UnwrappedError422::test_422_no_error_wrapper
tests/integration/test_p2_verification.py::TestP2Task4UnwrappedError422::test_422_no_success_field
tests/integration/test_p2_verification.py::TestP2Task4UnwrappedError422::test_422_status_code
tests/integration/test_p2_verification.py::TestP2Task5ProcessingTimeOnBoth::test_processing_time_on_200
tests/integration/test_p2_verification.py::TestP2Task5ProcessingTimeOnBoth::test_processing_time_on_422
tests/integration/test_p2_verification.py::TestSeedTruthfulness::test_computed_seed_is_deterministic
tests/integration/test_p2_verification.py::TestSeedTruthfulness::test_computed_seed_when_no_explicit_seed
tests/integration/test_p2_verification.py::TestSeedTruthfulness::test_different_graphs_produce_different_computed_seeds
tests/integration/test_p2_verification.py::TestSeedTruthfulness::test_explicit_seed_echoed_correctly
tests/integration/test_p2_verification.py::TestSeedTruthfulness::test_reproducibility_with_same_seed
tests/integration/test_p2_verification.py::TestStabilityThresholdsEnvelope::test_stability_thresholds_absent_without_bootstrap
tests/integration/test_p2_verification.py::TestStabilityThresholdsEnvelope::test_stability_thresholds_present_with_bootstrap
```

#### `tests/integration/test_parameter_recommendations.py` (22 tests)

```
tests/integration/test_parameter_recommendations.py::test_all_confidence_levels_are_valid
tests/integration/test_parameter_recommendations.py::test_belief_ranges_are_probabilities
tests/integration/test_parameter_recommendations.py::test_critical_edges_count_matches_graph
tests/integration/test_parameter_recommendations.py::test_critical_edges_ranked_highest
tests/integration/test_parameter_recommendations.py::test_critical_path_edges_get_strong_weight_recommendations
tests/integration/test_parameter_recommendations.py::test_handles_complex_multi_path_graph
tests/integration/test_parameter_recommendations.py::test_handles_disconnected_graphs
tests/integration/test_parameter_recommendations.py::test_handles_single_node_graph
tests/integration/test_parameter_recommendations.py::test_includes_current_values_when_provided
tests/integration/test_parameter_recommendations.py::test_mediator_nodes_get_moderate_high_certainty
tests/integration/test_parameter_recommendations.py::test_outcome_nodes_get_high_certainty_recommendations
tests/integration/test_parameter_recommendations.py::test_peripheral_edges_get_moderate_weight_recommendations
tests/integration/test_parameter_recommendations.py::test_rationale_includes_node_labels
tests/integration/test_parameter_recommendations.py::test_recommendations_include_human_readable_rationale
tests/integration/test_parameter_recommendations.py::test_recommendations_sorted_by_importance
tests/integration/test_parameter_recommendations.py::test_recommended_typical_is_center_of_range
tests/integration/test_parameter_recommendations.py::test_rejects_empty_graph
tests/integration/test_parameter_recommendations.py::test_response_includes_graph_characteristics
tests/integration/test_parameter_recommendations.py::test_response_time_under_2_seconds
tests/integration/test_parameter_recommendations.py::test_risk_nodes_get_moderate_uncertainty_recommendations
tests/integration/test_parameter_recommendations.py::test_treatment_nodes_get_high_certainty_recommendations
tests/integration/test_parameter_recommendations.py::test_weight_ranges_are_valid
```

#### `tests/integration/test_phase1_critical_endpoints.py` (19 tests)

```
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_close_race
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_minimum_options
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_multiple_options
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_negative_top_value
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_recommendations
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_sample_perturbations
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_stability_analysis
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_success
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_with_confidence_intervals
tests/integration/test_phase1_critical_endpoints.py::TestCoherenceEndpoint::test_coherence_endpoint_with_request_id
tests/integration/test_phase1_critical_endpoints.py::TestEndpointValidation::test_coherence_insufficient_options
tests/integration/test_phase1_critical_endpoints.py::TestEndpointValidation::test_feasibility_missing_constraints
tests/integration/test_phase1_critical_endpoints.py::TestFeasibilityEndpoint::test_feasibility_endpoint_all_feasible
tests/integration/test_phase1_critical_endpoints.py::TestFeasibilityEndpoint::test_feasibility_endpoint_all_infeasible
tests/integration/test_phase1_critical_endpoints.py::TestFeasibilityEndpoint::test_feasibility_endpoint_constraint_validation
tests/integration/test_phase1_critical_endpoints.py::TestFeasibilityEndpoint::test_feasibility_endpoint_multiple_constraints
tests/integration/test_phase1_critical_endpoints.py::TestFeasibilityEndpoint::test_feasibility_endpoint_success
tests/integration/test_phase1_critical_endpoints.py::TestFeasibilityEndpoint::test_feasibility_endpoint_violation_details
tests/integration/test_phase1_critical_endpoints.py::TestFeasibilityEndpoint::test_feasibility_endpoint_with_request_id
```

#### `tests/integration/test_phase2_schema_endpoints.py` (32 tests)

```
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_conflicting_correlation_warning
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_diagonal_ones
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_direct_matrix
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_eigenvalue_analysis
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_high_correlation_warning
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_invalid_request_no_input
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_matrix_psd_check
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_matrix_symmetry
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_missing_factor_warning
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_multiple_groups
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_negative_correlation
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_non_factor_node_warning
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_non_psd_matrix
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_perfect_correlation_issue
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_request_id_header
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_single_group
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_skip_psd_check
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_with_graph
tests/integration/test_phase2_schema_endpoints.py::TestCorrelationValidateEndpoint::test_correlations_validate_zero_correlation
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_all_aggregation_methods
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_equal_weights
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_explicit_weights
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_graph_missing_reference_warning
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_invalid_request
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_lexicographic_method
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_lexicographic_no_priority_warning
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_normalised_goals_output
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_request_id_header
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_risk_averse
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_risk_averse_no_coefficient_warning
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_weight_normalization
tests/integration/test_phase2_schema_endpoints.py::TestUtilityValidateEndpoint::test_utility_validate_with_graph_reference
```

#### `tests/integration/test_phase3_e2e.py` (9 tests)

```
tests/integration/test_phase3_e2e.py::TestContrastiveWithBatchValidation::test_contrastive_to_batch
tests/integration/test_phase3_e2e.py::TestDeterminismAcrossFeatures::test_determinism_batch
tests/integration/test_phase3_e2e.py::TestDeterminismAcrossFeatures::test_determinism_contrastive
tests/integration/test_phase3_e2e.py::TestDeterminismAcrossFeatures::test_determinism_transportability
tests/integration/test_phase3_e2e.py::TestInteractionDiscoveryWorkflow::test_interaction_discovery
tests/integration/test_phase3_e2e.py::TestMetadataConsistency::test_batch_metadata
tests/integration/test_phase3_e2e.py::TestMetadataConsistency::test_contrastive_metadata
tests/integration/test_phase3_e2e.py::TestMetadataConsistency::test_transportability_metadata
tests/integration/test_phase3_e2e.py::TestPricingOptimizationWorkflow::test_pricing_workflow
```

#### `tests/integration/test_phase4_endpoints.py` (21 tests)

```
tests/integration/test_phase4_endpoints.py::TestConditionalRecommendEndpoint::test_conditional_recommend_invalid_condition_type
tests/integration/test_phase4_endpoints.py::TestConditionalRecommendEndpoint::test_conditional_recommend_respects_max_conditions
tests/integration/test_phase4_endpoints.py::TestConditionalRecommendEndpoint::test_conditional_recommend_single_option_fails
tests/integration/test_phase4_endpoints.py::TestConditionalRecommendEndpoint::test_conditional_recommend_success
tests/integration/test_phase4_endpoints.py::TestConditionalRecommendEndpoint::test_conditional_recommend_with_risk_profile
tests/integration/test_phase4_endpoints.py::TestConditionalRecommendEndpoint::test_conditional_recommend_with_threshold_only
tests/integration/test_phase4_endpoints.py::TestPhase4ErrorHandling::test_duplicate_node_ids
tests/integration/test_phase4_endpoints.py::TestPhase4ErrorHandling::test_invalid_discount_factor
tests/integration/test_phase4_endpoints.py::TestPhase4ErrorHandling::test_invalid_graph_structure
tests/integration/test_phase4_endpoints.py::TestPhase4Integration::test_conditional_to_sequential_workflow
tests/integration/test_phase4_endpoints.py::TestPhase4Integration::test_full_decision_workflow
tests/integration/test_phase4_endpoints.py::TestPhase4ResponseTimes::test_conditional_recommend_response_time
tests/integration/test_phase4_endpoints.py::TestPhase4ResponseTimes::test_sequential_analysis_response_time
tests/integration/test_phase4_endpoints.py::TestPolicyTreeEndpoint::test_policy_tree_has_children
tests/integration/test_phase4_endpoints.py::TestPolicyTreeEndpoint::test_policy_tree_success
tests/integration/test_phase4_endpoints.py::TestSequentialAnalysisEndpoint::test_sequential_analysis_invalid_risk_tolerance
tests/integration/test_phase4_endpoints.py::TestSequentialAnalysisEndpoint::test_sequential_analysis_risk_averse
tests/integration/test_phase4_endpoints.py::TestSequentialAnalysisEndpoint::test_sequential_analysis_success
tests/integration/test_phase4_endpoints.py::TestSequentialAnalysisEndpoint::test_sequential_analysis_with_discount
tests/integration/test_phase4_endpoints.py::TestStageSensitivityEndpoint::test_stage_sensitivity_success
tests/integration/test_phase4_endpoints.py::TestStageSensitivityEndpoint::test_stage_sensitivity_with_parameters
```

#### `tests/integration/test_plot_workflows.py` (10 tests)

```
tests/integration/test_plot_workflows.py::test_all_responses_include_metadata
tests/integration/test_plot_workflows.py::test_batch_scenarios_with_interactions
tests/integration/test_plot_workflows.py::test_conformal_with_calibration_data
tests/integration/test_plot_workflows.py::test_deterministic_with_seed
tests/integration/test_plot_workflows.py::test_goal_seeking_workflow
tests/integration/test_plot_workflows.py::test_insufficient_calibration_fallback
tests/integration/test_plot_workflows.py::test_invalid_dag_returns_clear_error
tests/integration/test_plot_workflows.py::test_non_identifiable_gets_suggestions
tests/integration/test_plot_workflows.py::test_transportability_check
tests/integration/test_plot_workflows.py::test_validate_analyze_compare_workflow
```

#### `tests/integration/test_production_excellence.py` (4 tests)

```
tests/integration/test_production_excellence.py::TestCompression::test_large_response_compressed
tests/integration/test_production_excellence.py::TestDistributedTracing::test_trace_id_in_response_header
tests/integration/test_production_excellence.py::TestEndToEnd::test_batch_endpoint_with_tracing
tests/integration/test_production_excellence.py::TestEndToEnd::test_causal_validation_with_all_features
```

#### `tests/integration/test_risk_adjustment_endpoint.py` (23 tests)

```
tests/integration/test_risk_adjustment_endpoint.py::test_certainty_equivalent_clamping
tests/integration/test_risk_adjustment_endpoint.py::test_identical_means_different_variances
tests/integration/test_risk_adjustment_endpoint.py::test_interpretation_quality
tests/integration/test_risk_adjustment_endpoint.py::test_mixed_input_formats
tests/integration/test_risk_adjustment_endpoint.py::test_percentile_input_format
tests/integration/test_risk_adjustment_endpoint.py::test_request_id_tracking
tests/integration/test_risk_adjustment_endpoint.py::test_response_metadata
tests/integration/test_risk_adjustment_endpoint.py::test_response_structure_completeness
tests/integration/test_risk_adjustment_endpoint.py::test_risk_averse_basic
tests/integration/test_risk_adjustment_endpoint.py::test_risk_averse_high_coefficient
tests/integration/test_risk_adjustment_endpoint.py::test_risk_averse_rankings_changed
tests/integration/test_risk_adjustment_endpoint.py::test_risk_neutral_no_adjustment
tests/integration/test_risk_adjustment_endpoint.py::test_risk_neutral_sorts_by_mean
tests/integration/test_risk_adjustment_endpoint.py::test_risk_seeking_ranks_aggressive_highest
tests/integration/test_risk_adjustment_endpoint.py::test_risk_seeking_rewards_variance
tests/integration/test_risk_adjustment_endpoint.py::test_risk_type_comparison_same_options
tests/integration/test_risk_adjustment_endpoint.py::test_validation_incomplete_option
tests/integration/test_risk_adjustment_endpoint.py::test_validation_invalid_risk_type
tests/integration/test_risk_adjustment_endpoint.py::test_validation_requires_two_options
tests/integration/test_risk_adjustment_endpoint.py::test_validation_risk_averse_requires_positive_coefficient
tests/integration/test_risk_adjustment_endpoint.py::test_validation_risk_neutral_requires_zero_coefficient
tests/integration/test_risk_adjustment_endpoint.py::test_validation_score_bounds
tests/integration/test_risk_adjustment_endpoint.py::test_zero_variance_options
```

#### `tests/integration/test_tae_workflows.py` (11 tests)

```
tests/integration/test_tae_workflows.py::test_assumption_ranking - htt...
tests/integration/test_tae_workflows.py::test_batch_analysis_performance
tests/integration/test_tae_workflows.py::test_batch_counterfactuals_for_deliberation
tests/integration/test_tae_workflows.py::test_batch_with_explanations
tests/integration/test_tae_workflows.py::test_counterfactual_with_uncertainty
tests/integration/test_tae_workflows.py::test_invalid_proposal_gets_suggestions
tests/integration/test_tae_workflows.py::test_proposal_comparison - ht...
tests/integration/test_tae_workflows.py::test_robustness_filtering_for_proposals
tests/integration/test_tae_workflows.py::test_robustness_with_sensitivity_analysis
tests/integration/test_tae_workflows.py::test_sensitivity_for_disputed_assumption
tests/integration/test_tae_workflows.py::test_validate_team_proposal
```

#### `tests/integration/test_teaching_endpoint.py` (13 tests)

```
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_basic
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_causal
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_confounding
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_deterministic
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_different_concepts
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_invalid_request
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_learning_objectives
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_max_examples
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_optimization
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_privacy
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_ranked
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_time_estimate
tests/integration/test_teaching_endpoint.py::test_generate_teaching_examples_uncertainty
```

#### `tests/integration/test_team_endpoint.py` (4 tests)

```
tests/integration/test_team_endpoint.py::test_team_alignment_basic - T...
tests/integration/test_team_endpoint.py::test_team_alignment_common_ground
tests/integration/test_team_endpoint.py::test_team_alignment_ranked_options
tests/integration/test_team_endpoint.py::test_team_alignment_recommendation
```

#### `tests/integration/test_threshold_endpoint.py` (27 tests)

```
tests/integration/test_threshold_endpoint.py::test_all_options_always_tied
tests/integration/test_threshold_endpoint.py::test_ascending_vs_descending_different_results
tests/integration/test_threshold_endpoint.py::test_baseline_ranking_different_from_actual
tests/integration/test_threshold_endpoint.py::test_baseline_ranking_honored
tests/integration/test_threshold_endpoint.py::test_baseline_ranking_vs_no_baseline
tests/integration/test_threshold_endpoint.py::test_custom_sweep_order_respected
tests/integration/test_threshold_endpoint.py::test_descending_sweep_order
tests/integration/test_threshold_endpoint.py::test_exact_ties_alphabetical_ordering
tests/integration/test_threshold_endpoint.py::test_large_parameter_sweep
tests/integration/test_threshold_endpoint.py::test_low_confidence_threshold_finds_more
tests/integration/test_threshold_endpoint.py::test_monotonic_no_thresholds
tests/integration/test_threshold_endpoint.py::test_multiple_parameters_sensitivity_ranking
tests/integration/test_threshold_endpoint.py::test_multiple_parameters_threshold_aggregation
tests/integration/test_threshold_endpoint.py::test_multiple_thresholds_affected_options
tests/integration/test_threshold_endpoint.py::test_multiple_thresholds_detection
tests/integration/test_threshold_endpoint.py::test_non_monotonic_sweep_order
tests/integration/test_threshold_endpoint.py::test_request_id_tracking
tests/integration/test_threshold_endpoint.py::test_response_metadata
tests/integration/test_threshold_endpoint.py::test_response_structure_completeness
tests/integration/test_threshold_endpoint.py::test_single_option - Typ...
tests/integration/test_threshold_endpoint.py::test_single_threshold_detection
tests/integration/test_threshold_endpoint.py::test_single_threshold_sensitivity
tests/integration/test_threshold_endpoint.py::test_ties_with_confidence_threshold
tests/integration/test_threshold_endpoint.py::test_validation_consistent_options
tests/integration/test_threshold_endpoint.py::test_validation_requires_at_least_one_sweep
tests/integration/test_threshold_endpoint.py::test_validation_requires_at_least_two_values
tests/integration/test_threshold_endpoint.py::test_validation_scores_must_match_values
```

#### `tests/integration/test_transportability_endpoints.py` (17 tests)

```
tests/integration/test_transportability_endpoints.py::TestComplexScenarios::test_larger_dag
tests/integration/test_transportability_endpoints.py::TestComplexScenarios::test_with_data_summaries
tests/integration/test_transportability_endpoints.py::TestDeterminism::test_deterministic_results
tests/integration/test_transportability_endpoints.py::TestDirectTransport::test_direct_transport_explanation
tests/integration/test_transportability_endpoints.py::TestDirectTransport::test_direct_transport_has_assumptions
tests/integration/test_transportability_endpoints.py::TestDirectTransport::test_direct_transport_simple_dag
tests/integration/test_transportability_endpoints.py::TestNonTransportable::test_different_dag_structures
tests/integration/test_transportability_endpoints.py::TestNonTransportable::test_non_transportable_has_suggestions
tests/integration/test_transportability_endpoints.py::TestRequestTracing::test_custom_request_id
tests/integration/test_transportability_endpoints.py::TestRequestValidation::test_missing_outcome
tests/integration/test_transportability_endpoints.py::TestRequestValidation::test_missing_source_domain
tests/integration/test_transportability_endpoints.py::TestRequestValidation::test_missing_treatment
tests/integration/test_transportability_endpoints.py::TestResponseStructure::test_confidence_levels
tests/integration/test_transportability_endpoints.py::TestResponseStructure::test_response_has_metadata
tests/integration/test_transportability_endpoints.py::TestResponseStructure::test_robustness_levels
tests/integration/test_transportability_endpoints.py::TestSelectionDiagramTransport::test_selection_diagram_assumptions
tests/integration/test_transportability_endpoints.py::TestSelectionDiagramTransport::test_selection_diagram_with_explicit_variables
```

#### `tests/integration/test_validation_strategy_endpoints.py` (14 tests)

```
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEdgeCases::test_cyclic_dag
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEdgeCases::test_disconnected_graph
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEdgeCases::test_identical_treatment_outcome
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEdgeCases::test_self_loop_dag
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_already_identifiable
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_backdoor_strategy_generation
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_complex_dag
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_invalid_dag
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_metadata_included
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_missing_nodes
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_multiple_strategies_ranked
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_path_analysis_included
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_response_structure
tests/integration/test_validation_strategy_endpoints.py::TestValidationStrategyEndpoint::test_simple_confounding_case
```

---

## Category 2: Smoke Tests (Production Health)

14 ERROR tests in 1 file. These tests have their own `tests/smoke/pytest.ini` that
injects `--base-url=${BASE_URL}` and `--api-key=${API_KEY}` as default CLI args. When
run from the project root, pytest rejects these unrecognized arguments, causing
collection errors on every test in the file. These tests are designed to run
post-deployment against a live instance, not in local CI.

| File | Tests | Error Type |
|------|------:|------------|
| `tests/smoke/test_production_health.py` | 14 | `pytest.ini` arg conflict (ERROR) |

```
tests/smoke/test_production_health.py::test_causal_validation_basic
tests/smoke/test_production_health.py::test_causal_validation_latency
tests/smoke/test_production_health.py::test_counterfactual_generation_basic
tests/smoke/test_production_health.py::test_counterfactual_latency
tests/smoke/test_production_health.py::test_health_endpoint
tests/smoke/test_production_health.py::test_invalid_dag_rejected
tests/smoke/test_production_health.py::test_llm_endpoints_responding
tests/smoke/test_production_health.py::test_metadata_present_in_all_responses
tests/smoke/test_production_health.py::test_metrics_endpoint
tests/smoke/test_production_health.py::test_missing_api_key_rejected
tests/smoke/test_production_health.py::test_openapi_docs
tests/smoke/test_production_health.py::test_robustness_analysis_basic
tests/smoke/test_production_health.py::test_smoke_test_summary
tests/smoke/test_production_health.py::test_validate_then_counterfactual_workflow
```

---

## Category 3: Unit Test Failures (Service Implementation Drift)

93 tests across 17 files (77 FAILED + 16 ERROR). These are unit tests where the
service implementations have evolved but the tests were not updated. Common patterns
include: `StructuralModel` API changes (e.g. `equations` dict format), mock setup
mismatches, assertion drift on computed values, fixture setup errors, or missing
async client fixtures for endpoint tests embedded in unit test files.

The 16 ERROR tests in `test_conditional_recommender.py` fail during fixture setup
(the module imports correctly but test fixtures reference models or request types
that have since changed).

| File | Tests | Type |
|------|------:|------|
| `tests/unit/test_conditional_recommender.py` | 16 | ERROR (fixture setup) |
| `tests/unit/test_contrastive_explainer.py` | 14 | FAILED |
| `tests/unit/test_identifiability_v2.py` | 12 | FAILED |
| `tests/unit/test_conformal_predictor.py` | 11 | FAILED |
| `tests/unit/test_advanced_discovery.py` | 7 | FAILED |
| `tests/unit/test_causal_discovery_engine.py` | 7 | FAILED |
| `tests/unit/test_batch_counterfactual_engine.py` | 6 | FAILED |
| `tests/unit/test_auth_middleware.py` | 4 | FAILED |
| `tests/unit/test_sensitivity_analyzer.py` | 3 | FAILED |
| `tests/unit/test_sequential_optimizer.py` | 3 | FAILED |
| `tests/unit/test_explanation_generator.py` | 3 | FAILED |
| `tests/unit/test_confounding_sensitivity.py` | 2 | FAILED |
| `tests/unit/test_dag_visualization.py` | 1 | FAILED |
| `tests/unit/test_plot_client.py` | 1 | FAILED |
| `tests/unit/test_preference_elicitor.py` | 1 | FAILED |
| `tests/unit/test_security_config.py` | 1 | FAILED |
| `tests/unit/test_bayesian_teacher.py` | 1 | FAILED |

### Detailed test list

#### `tests/unit/test_advanced_discovery.py` (7 tests)

```
tests/unit/test_advanced_discovery.py::TestAdvancedCausalDiscovery::test_discover_notears
tests/unit/test_advanced_discovery.py::TestCausalDiscoveryEngineIntegration::test_comparison_simple_vs_advanced
tests/unit/test_advanced_discovery.py::TestCausalDiscoveryEngineIntegration::test_discover_advanced_disabled
tests/unit/test_advanced_discovery.py::TestNOTEARSDiscovery::test_basic_discovery
tests/unit/test_advanced_discovery.py::TestPerformance::test_convergence
tests/unit/test_advanced_discovery.py::TestSyntheticData::test_chain_structure
tests/unit/test_advanced_discovery.py::TestSyntheticData::test_fork_structure
```

#### `tests/unit/test_auth_middleware.py` (4 tests)

```
tests/unit/test_auth_middleware.py::TestAPIKeyMiddlewareClientIP::test_get_client_ip_direct
tests/unit/test_auth_middleware.py::TestAPIKeyMiddlewareClientIP::test_get_client_ip_from_forwarded_for
tests/unit/test_auth_middleware.py::TestAPIKeyMiddlewareClientIP::test_get_client_ip_from_real_ip
tests/unit/test_auth_middleware.py::TestAPIKeyMiddlewareClientIP::test_get_client_ip_unknown
```

#### `tests/unit/test_batch_counterfactual_engine.py` (6 tests)

```
tests/unit/test_batch_counterfactual_engine.py::TestEdgeCases::test_complex_multi_variable_scenarios
tests/unit/test_batch_counterfactual_engine.py::TestExplanationGeneration::test_explanation_mentions_interactions
tests/unit/test_batch_counterfactual_engine.py::TestInteractionDetection::test_additive_effects
tests/unit/test_batch_counterfactual_engine.py::TestInteractionDetection::test_antagonistic_interaction
tests/unit/test_batch_counterfactual_engine.py::TestInteractionDetection::test_interaction_missing_scenarios
tests/unit/test_batch_counterfactual_engine.py::TestInteractionDetection::test_synergistic_interaction
```

#### `tests/unit/test_bayesian_teacher.py` (1 tests)

```
tests/unit/test_bayesian_teacher.py::test_generate_teaching_examples_deterministic
```

#### `tests/unit/test_causal_discovery_engine.py` (7 tests)

```
tests/unit/test_causal_discovery_engine.py::TestConfidenceComputation::test_strong_correlation_high_confidence
tests/unit/test_causal_discovery_engine.py::TestDAGValidation::test_acyclicity_check
tests/unit/test_causal_discovery_engine.py::TestDAGValidation::test_self_loop_detection
tests/unit/test_causal_discovery_engine.py::TestDiscoveryFromData::test_confidence_scoring
tests/unit/test_causal_discovery_engine.py::TestDiscoveryFromKnowledge::test_confidence_ranking
tests/unit/test_causal_discovery_engine.py::TestDiscoveryFromKnowledge::test_top_k_parameter
tests/unit/test_causal_discovery_engine.py::TestPriorKnowledgeApplication::test_no_prior_knowledge
```

#### `tests/unit/test_conditional_recommender.py` (16 tests)

```
tests/unit/test_conditional_recommender.py::TestAutoDetectParameters::test_auto_detect_when_none_specified
tests/unit/test_conditional_recommender.py::TestConditionTypes::test_only_specified_types_generated
tests/unit/test_conditional_recommender.py::TestConditionTypes::test_scenario_conditions
tests/unit/test_conditional_recommender.py::TestDominanceDetection::test_dominance_condition_generated
tests/unit/test_conditional_recommender.py::TestEmptyConditions::test_empty_conditions_when_robust
tests/unit/test_conditional_recommender.py::TestMaxConditionsLimit::test_conditions_sorted_by_impact
tests/unit/test_conditional_recommender.py::TestMaxConditionsLimit::test_respects_max_conditions
tests/unit/test_conditional_recommender.py::TestPrimaryRecommendation::test_confidence_levels
tests/unit/test_conditional_recommender.py::TestPrimaryRecommendation::test_primary_is_highest_ev
tests/unit/test_conditional_recommender.py::TestRiskProfileConditions::test_risk_averse_prefers_lower_variance
tests/unit/test_conditional_recommender.py::TestRiskProfileConditions::test_risk_profile_condition
tests/unit/test_conditional_recommender.py::TestRobustnessClassification::test_fragile_when_close_options
tests/unit/test_conditional_recommender.py::TestRobustnessClassification::test_robust_when_far_from_flip
tests/unit/test_conditional_recommender.py::TestRobustnessClassification::test_robustness_summary_has_conditions_count
tests/unit/test_conditional_recommender.py::TestThresholdConditions::test_threshold_condition_generated
tests/unit/test_conditional_recommender.py::TestThresholdConditions::test_threshold_condition_has_probability
```

#### `tests/unit/test_conformal_predictor.py` (11 tests)

```
tests/unit/test_conformal_predictor.py::TestConformityScores::test_conformity_scores_are_non_negative
tests/unit/test_conformal_predictor.py::TestConformityScores::test_conformity_scores_are_residuals
tests/unit/test_conformal_predictor.py::TestConformityScores::test_conformity_scores_length
tests/unit/test_conformal_predictor.py::TestCoverageGuarantee::test_coverage_guarantee_always_below_nominal
tests/unit/test_conformal_predictor.py::TestEdgeCases::test_minimum_calibration_data_requirement
tests/unit/test_conformal_predictor.py::TestEdgeCases::test_none_calibration_data
tests/unit/test_conformal_predictor.py::TestEdgeCases::test_split_with_barely_enough_data
tests/unit/test_conformal_predictor.py::TestResponseStructure::test_coverage_guarantee_is_finite_sample_valid
tests/unit/test_conformal_predictor.py::TestSplitConformal::test_split_conformal_basic
tests/unit/test_conformal_predictor.py::TestSplitConformal::test_split_creates_calibration_split
tests/unit/test_conformal_predictor.py::TestSplitConformal::test_split_uses_seed_for_determinism
```

#### `tests/unit/test_confounding_sensitivity.py` (2 tests)

```
tests/unit/test_confounding_sensitivity.py::TestEndpointIntegration::test_bidirected_graph_includes_confounding_sensitivity
tests/unit/test_confounding_sensitivity.py::TestEndpointIntegration::test_identifiable_graph_no_confounding_sensitivity
```

#### `tests/unit/test_contrastive_explainer.py` (14 tests)

```
tests/unit/test_contrastive_explainer.py::TestComparison::test_comparison_multiple_interventions
tests/unit/test_contrastive_explainer.py::TestComparison::test_comparison_single_intervention
tests/unit/test_contrastive_explainer.py::TestContrastiveExplainerBasic::test_deterministic_with_seed
tests/unit/test_contrastive_explainer.py::TestContrastiveExplainerBasic::test_multi_variable_combination
tests/unit/test_contrastive_explainer.py::TestContrastiveExplainerBasic::test_multiple_feasible_variables
tests/unit/test_contrastive_explainer.py::TestContrastiveExplainerBasic::test_no_solution_returns_empty
tests/unit/test_contrastive_explainer.py::TestContrastiveExplainerBasic::test_respects_fixed_constraints
tests/unit/test_contrastive_explainer.py::TestContrastiveExplainerBasic::test_single_variable_minimal_intervention
tests/unit/test_contrastive_explainer.py::TestEdgeCases::test_very_tight_target_range
tests/unit/test_contrastive_explainer.py::TestExplanationGeneration::test_explanation_with_interventions
tests/unit/test_contrastive_explainer.py::TestRankingAlgorithms::test_rank_by_change_magnitude
tests/unit/test_contrastive_explainer.py::TestRankingAlgorithms::test_rank_by_cost
tests/unit/test_contrastive_explainer.py::TestRankingAlgorithms::test_rank_by_feasibility
tests/unit/test_contrastive_explainer.py::TestRobustnessIntegration::test_robustness_evaluated
```

#### `tests/unit/test_dag_visualization.py` (1 tests)

```
tests/unit/test_dag_visualization.py::TestEdgeCases::test_cyclic_graph
```

#### `tests/unit/test_explanation_generator.py` (3 tests)

```
tests/unit/test_explanation_generator.py::TestReadabilityScoring::test_readability_sentence_count
tests/unit/test_explanation_generator.py::TestReadabilityScoring::test_readability_words_per_sentence
tests/unit/test_explanation_generator.py::TestSyllableCounting::test_count_syllables_two_syllables
```

#### `tests/unit/test_identifiability_v2.py` (12 tests)

```
tests/unit/test_identifiability_v2.py::TestEndpointFrontdoor::test_frontdoor_eligible_identifiable
tests/unit/test_identifiability_v2.py::TestEndpointIdentifiable::test_estimand_is_null_not_prose
tests/unit/test_identifiability_v2.py::TestEndpointIdentifiable::test_identifiable_graph_returns_200
tests/unit/test_identifiability_v2.py::TestEndpointLatency::test_endpoint_latency_12_node
tests/unit/test_identifiability_v2.py::TestEndpointMalformedInput::test_empty_interventions_returns_400
tests/unit/test_identifiability_v2.py::TestEndpointMalformedInput::test_empty_options_returns_422
tests/unit/test_identifiability_v2.py::TestEndpointMalformedInput::test_missing_graph_returns_422
tests/unit/test_identifiability_v2.py::TestEndpointMalformedInput::test_missing_outcome_node_returns_400
tests/unit/test_identifiability_v2.py::TestEndpointMalformedInput::test_missing_treatment_node_returns_400
tests/unit/test_identifiability_v2.py::TestEndpointNonIdentifiable::test_bidirected_returns_non_identifiable
tests/unit/test_identifiability_v2.py::TestEndpointTreatmentExpansion::test_multiple_options_deduplicate_treatments
tests/unit/test_identifiability_v2.py::TestEndpointTreatmentExpansion::test_multiple_treatment_factors
```

#### `tests/unit/test_plot_client.py` (1 tests)

```
tests/unit/test_plot_client.py::test_retries_network_errors_with_backoff
```

#### `tests/unit/test_preference_elicitor.py` (1 tests)

```
tests/unit/test_preference_elicitor.py::test_compute_information_gain_deterministic
```

#### `tests/unit/test_security_config.py` (1 tests)

```
tests/unit/test_security_config.py::TestProductionConfigValidation::test_production_valid_config
```

#### `tests/unit/test_sensitivity_analyzer.py` (3 tests)

```
tests/unit/test_sensitivity_analyzer.py::TestEdgeCases::test_empty_violation_levels
tests/unit/test_sensitivity_analyzer.py::TestEdgeCases::test_extreme_violation_levels
tests/unit/test_sensitivity_analyzer.py::TestPredictOutcome::test_predict_different_interventions
```

#### `tests/unit/test_sequential_optimizer.py` (3 tests)

```
tests/unit/test_sequential_optimizer.py::TestParameterSampling::test_multiple_parameter_sampling
tests/unit/test_sequential_optimizer.py::TestParameterSampling::test_normal_sampling
tests/unit/test_sequential_optimizer.py::TestParameterSampling::test_uniform_sampling
```

---

## Category 4: Property Tests (Hypothesis)

4 FAILED tests in 1 file. These are Hypothesis-based property tests for weight
normalization. The service API for weight normalization has drifted from what the
tests expect.

| File | Tests |
|------|------:|
| `tests/property/test_weight_normalization.py` | 4 |

```
tests/property/test_weight_normalization.py::TestWeightNormalizationProperties::test_all_normalized_weights_positive
tests/property/test_weight_normalization.py::TestWeightNormalizationProperties::test_normalized_weights_preserve_order
tests/property/test_weight_normalization.py::TestWeightNormalizationProperties::test_normalized_weights_sum_to_one
tests/property/test_weight_normalization.py::TestWeightNormalizationProperties::test_uniform_weights_equal_after_normalization
```

---

## Category 5: Duplicate File Tests (Accidental Copies)

32 FAILED tests across 6 files. These are files with spaces in their names
(e.g., `"test_identifiability_v2 2.py"`) that appear to be accidental copies
created by macOS or editor save conflicts. They are duplicates of the original
files and should be deleted from the repository.

| File | Tests | Original |
|------|------:|----------|
| `tests/unit/test_identifiability_v2 2.py` | 12 | `tests/unit/test_identifiability_v2.py` |
| `tests/unit/test_identifiability_v2 3.py` | 12 | `tests/unit/test_identifiability_v2.py` |
| `tests/unit/test_confounding_sensitivity 2.py` | 2 | `tests/unit/test_confounding_sensitivity.py` |
| `tests/unit/test_confounding_sensitivity 3.py` | 2 | `tests/unit/test_confounding_sensitivity.py` |
| `tests/unit/test_constraint_analysis 2.py` | 2 | `tests/unit/test_constraint_analysis.py` |
| `tests/unit/test_constraint_analysis 3.py` | 2 | `tests/unit/test_constraint_analysis.py` |

### Detailed test list

#### `tests/unit/test_identifiability_v2 2.py` (12 tests)

```
tests/unit/test_identifiability_v2 2.py::TestEndpointFrontdoor::test_frontdoor_eligible_identifiable
tests/unit/test_identifiability_v2 2.py::TestEndpointIdentifiable::test_estimand_is_null_not_prose
tests/unit/test_identifiability_v2 2.py::TestEndpointIdentifiable::test_identifiable_graph_returns_200
tests/unit/test_identifiability_v2 2.py::TestEndpointLatency::test_endpoint_latency_12_node
tests/unit/test_identifiability_v2 2.py::TestEndpointMalformedInput::test_empty_interventions_returns_400
tests/unit/test_identifiability_v2 2.py::TestEndpointMalformedInput::test_empty_options_returns_422
tests/unit/test_identifiability_v2 2.py::TestEndpointMalformedInput::test_missing_graph_returns_422
tests/unit/test_identifiability_v2 2.py::TestEndpointMalformedInput::test_missing_outcome_node_returns_400
tests/unit/test_identifiability_v2 2.py::TestEndpointMalformedInput::test_missing_treatment_node_returns_400
tests/unit/test_identifiability_v2 2.py::TestEndpointNonIdentifiable::test_bidirected_returns_non_identifiable
tests/unit/test_identifiability_v2 2.py::TestEndpointTreatmentExpansion::test_multiple_options_deduplicate_treatments
tests/unit/test_identifiability_v2 2.py::TestEndpointTreatmentExpansion::test_multiple_treatment_factors
```

#### `tests/unit/test_identifiability_v2 3.py` (12 tests)

```
tests/unit/test_identifiability_v2 3.py::TestEndpointFrontdoor::test_frontdoor_eligible_identifiable
tests/unit/test_identifiability_v2 3.py::TestEndpointIdentifiable::test_estimand_is_null_not_prose
tests/unit/test_identifiability_v2 3.py::TestEndpointIdentifiable::test_identifiable_graph_returns_200
tests/unit/test_identifiability_v2 3.py::TestEndpointLatency::test_endpoint_latency_12_node
tests/unit/test_identifiability_v2 3.py::TestEndpointMalformedInput::test_empty_interventions_returns_400
tests/unit/test_identifiability_v2 3.py::TestEndpointMalformedInput::test_empty_options_returns_422
tests/unit/test_identifiability_v2 3.py::TestEndpointMalformedInput::test_missing_graph_returns_422
tests/unit/test_identifiability_v2 3.py::TestEndpointMalformedInput::test_missing_outcome_node_returns_400
tests/unit/test_identifiability_v2 3.py::TestEndpointMalformedInput::test_missing_treatment_node_returns_400
tests/unit/test_identifiability_v2 3.py::TestEndpointNonIdentifiable::test_bidirected_returns_non_identifiable
tests/unit/test_identifiability_v2 3.py::TestEndpointTreatmentExpansion::test_multiple_options_deduplicate_treatments
tests/unit/test_identifiability_v2 3.py::TestEndpointTreatmentExpansion::test_multiple_treatment_factors
```

#### `tests/unit/test_confounding_sensitivity 2.py` (2 tests)

```
tests/unit/test_confounding_sensitivity 2.py::TestEndpointIntegration::test_bidirected_graph_includes_confounding_sensitivity
tests/unit/test_confounding_sensitivity 2.py::TestEndpointIntegration::test_identifiable_graph_no_confounding_sensitivity
```

#### `tests/unit/test_confounding_sensitivity 3.py` (2 tests)

```
tests/unit/test_confounding_sensitivity 3.py::TestEndpointIntegration::test_bidirected_graph_includes_confounding_sensitivity
tests/unit/test_confounding_sensitivity 3.py::TestEndpointIntegration::test_identifiable_graph_no_confounding_sensitivity
```

#### `tests/unit/test_constraint_analysis 2.py` (2 tests)

```
tests/unit/test_constraint_analysis 2.py::TestInferenceWarningsDefaultBase::test_warning_when_nonroot_constraint_node_lacks_uncertainty
tests/unit/test_constraint_analysis 2.py::TestInferenceWarningsDefaultBase::test_warning_with_multiple_constraint_nodes
```

#### `tests/unit/test_constraint_analysis 3.py` (2 tests)

```
tests/unit/test_constraint_analysis 3.py::TestInferenceWarningsDefaultBase::test_warning_when_nonroot_constraint_node_lacks_uncertainty
tests/unit/test_constraint_analysis 3.py::TestInferenceWarningsDefaultBase::test_warning_with_multiple_constraint_nodes
```

---

## Totals

| Source | Count |
|--------|------:|
| FAILED (from original baseline run) | 524 |
| ERROR (from original baseline run) | 30 |
| **Total quarantined** | **554** |

Breakdown by test directory:

| Directory | Count |
|-----------|------:|
| `tests/integration/` | 411 |
| `tests/unit/` (non-duplicate) | 93 |
| `tests/unit/` (duplicate files) | 32 |
| `tests/smoke/` | 14 |
| `tests/property/` | 4 |
| **Total** | **554** |
