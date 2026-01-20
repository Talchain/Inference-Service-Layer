# P2-ISL Design Documentation

## Overview

This document consolidates design decisions made during the P2-ISL workstream, which focused on numerical stability, determinism, and response format improvements for the Inference Service Layer.

---

## P2-ISL-1: Determinism and Seed Tracking

**Location:** `src/utils/rng.py`, `src/utils/determinism.py`, `src/constants/__init__.py`

**Problem:** Requests with the same inputs should produce identical results for debugging and reproducibility.

**Solution:**
- Compute deterministic seed from graph hash when `seed` not provided
- Use `SeededRNG` class with separate streams for edge/factor/noise sampling
- Deprecated `set_global_seed()` in favor of `make_deterministic()` for request isolation

**Key Code:**
```python
# src/utils/rng.py
seed = compute_seed_from_graph(request.graph)
rng_edge = SeededRNG(seed)
rng_factor = SeededRNG(seed + 1)
rng_noise = SeededRNG(seed + 2)
```

---

## P2-ISL-2: Request ID Handling and Security

**Location:** `src/api/robustness.py`, `src/utils/tracing.py`

**Problem:** Request IDs need to be unique, traceable, and sanitized for security.

**Solution:**
- Generate request ID if not provided (`req_<uuid>`)
- Sanitize incoming request IDs to prevent log injection
- Support both `X-Request-Id` (platform standard) and `X-Trace-Id` (ISL legacy)

**Headers:**
| Header | Priority | Status |
|--------|----------|--------|
| `X-Request-Id` | 1 | Platform standard |
| `X-Trace-Id` | 2 | ISL legacy (deprecated) |
| Generated | 3 | Fallback |

---

## P2-ISL-3: 422 Error Unwrapping

**Location:** `src/models/response_v2.py`, `src/api/robustness.py`

**Problem:** Validation errors need to return consistent, unwrapped 422 responses.

**Solution:**
- `ISLV2Error422` model for unwrapped error responses
- Critiques with severity "blocker" trigger 422 status
- Clear error messages with recovery hints

**Response Format:**
```json
{
  "analysis_status": "blocked",
  "critiques": [
    {
      "code": "GRAPH_NO_PATH_TO_GOAL",
      "severity": "blocker",
      "message": "No causal path exists..."
    }
  ]
}
```

---

## P2-ISL-4: Graph Structure Validation

**Location:** `src/models/critique.py`, `src/validation/request_validator.py`

**Problem:** Invalid graph structures should be caught early with actionable critiques.

**Validation Checks:**
| Check | Critique Code | Severity |
|-------|---------------|----------|
| No path to goal | `GRAPH_NO_PATH_TO_GOAL` | blocker |
| Identical options | `IDENTICAL_OPTIONS` | blocker |
| Self-loops | Pydantic validation error | blocker |
| Non-existent node refs | Pydantic validation error | blocker |
| Non-inference nodes | `robustness_v2_filtered_non_inference_nodes` (warning) | warning |

---

## P2-ISL-5: Epsilon Guards and Baseline Protection

**Location:** `src/constants/__init__.py`, `src/utils/numerical_stability.py`

**Problem:** Division by near-zero values causes NaN/Inf in sensitivity calculations.

**Constants:**
```python
BASELINE_EPSILON = 1e-8          # Min baseline for division
ZERO_VARIANCE_TOLERANCE = 1e-10  # Variance detection threshold
MIN_VALID_RATIO = 0.8            # Min valid samples for "computed"
```

**Protection Functions:**
- `safe_sensitivity()`: Guards elasticity calculation
- `safe_percent_change()`: Guards percentage calculations
- `validate_mc_samples()`: Cleans NaN/Inf from MC samples

---

## Schema Constraints (v2.6)

**Strength Distribution:**
- `std` must be > 0.001 (prevents near-zero variance)
- `mean` can be any float (positive or negative)

**Graph Limits:**
- Max nodes: 100
- Max edges: 300
- Max samples: 10,000

**Non-Inference Kinds:**
Nodes with these kinds are filtered before analysis:
- `decision`
- `option`
- `constraint`

---

## Response Versions

| Version | Key Differences |
|---------|-----------------|
| V1 (legacy) | `results[]`, `option_id`, `ci_lower/ci_upper` |
| V2 (enhanced) | `options[]`, `id`, `p10/p50/p90`, validity metrics |

Default: V1 (for backward compatibility)
Query param: `?response_version=2` for enhanced format

---

## Deprecation Notes

| Item | Status | Target Removal |
|------|--------|----------------|
| `X-Trace-Id` header | Deprecated | v3.0 |
| `set_global_seed()` | Deprecated | v3.0 |
| `legacy_suggestions` field | Deprecated | v3.0 |
| V1 response format | Active | TBD (see timeline) |

---

## Testing Guidelines

1. **Numerical edge cases**: Test with values near BASELINE_EPSILON
2. **Determinism**: Verify same seed produces identical results
3. **Validation**: Test all blocker critique triggers
4. **Filtering**: Test non-inference node removal with warning
