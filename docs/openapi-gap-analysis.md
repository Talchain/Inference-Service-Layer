# OpenAPI Specification Gap Analysis

**Date:** 2025-12-17
**Status:** Completed

## Executive Summary

The committed `openapi.json` documents only **12 API paths**, but the codebase actually implements **51 endpoints** (excluding metrics which is explicitly excluded from the schema). This represents an **76% documentation gap** (39 undocumented endpoints).

Additionally, **4 endpoints** in the current spec are stale (archived routers that are no longer registered).

## Current State

### Documented Paths in `openapi.json` (12 paths)

| # | Path | Status | Notes |
|---|------|--------|-------|
| 1 | `/health` | **Valid** | |
| 2 | `/api/v1/causal/validate` | **Valid** | |
| 3 | `/api/v1/causal/counterfactual` | **Valid** | |
| 4 | `/api/v1/preferences/elicit` | **STALE** | Router archived (TAE PoC v02) |
| 5 | `/api/v1/preferences/update` | **STALE** | Router archived (TAE PoC v02) |
| 6 | `/api/v1/teaching/teach` | **Valid** | |
| 7 | `/api/v1/validation/validate` | **Valid** | |
| 8 | `/api/v1/team/align` | **Valid** | |
| 9 | `/api/v1/analysis/sensitivity` | **Valid** | |
| 10 | `/api/v1/robustness/analyze` | **Valid** | |
| 11 | `/api/v1/deliberation/deliberate` | **STALE** | Router archived (TAE PoC v02) |
| 12 | `/api/v1/deliberation/session/{session_id}` | **STALE** | Router archived (TAE PoC v02) |

**Summary:** 8 valid, 4 stale

### Registered Routers in `src/api/main.py` (20 active routers)

| # | Router | Prefix | Tag | Endpoints |
|---|--------|--------|-----|-----------|
| 1 | health_router | (root) | Health | 5 |
| 2 | metrics_router | (root) | Monitoring | 1* |
| 3 | causal_router | /api/v1/causal | Causal Inference | 12 |
| 4 | batch_router | /api/v1/batch | Batch Processing | 2 |
| 5 | teaching_router | /api/v1/teaching | Bayesian Teaching | 1 |
| 6 | validation_router | /api/v1/validation | Advanced Validation | 4 |
| 7 | utility_router | /api/v1/utility | Utility Functions | 1 |
| 8 | team_router | /api/v1/team | Team Alignment | 1 |
| 9 | analysis_router | /api/v1/analysis | Sensitivity Analysis | 2 |
| 10 | dominance_router | /api/v1/analysis | Multi-Criteria Analysis | 2 |
| 11 | risk_router | /api/v1/analysis | Multi-Criteria Analysis | 1 |
| 12 | threshold_router | /api/v1/analysis | Multi-Criteria Analysis | 1 |
| 13 | aggregation_router | /api/v1/aggregation | Multi-Criteria Analysis | 1 |
| 14 | robustness_router | /api/v1/robustness | FACET Robustness | 3 |
| 15 | explain_router | /api/v1/explain | Contrastive Explanations | 1 |
| 16 | cee_router | /api/v1 | CEE Enhancement | 4 |
| 17 | phase4_router | /api/v1/analysis | Phase 4: Sequential Decisions | 4 |
| 18 | identifiability_router | /api/v1/analysis | Y₀ Identifiability | 2 |
| 19 | decision_robustness_router | /api/v1/analysis | Decision Robustness Suite | 1 |
| 20 | outcomes_router | /api/v1/outcomes | Outcome Logging | 4 |

*Note: metrics_router has `include_in_schema=False`, so it won't appear in OpenAPI spec.

**Total Active Endpoints:** 51 (excluding metrics)

### Archived/Commented Routers (NOT registered)

| Router | Reason |
|--------|--------|
| preferences_router | Archived - deferred to TAE PoC v02 |
| deliberation_router | Archived - deferred to TAE PoC v02 |

## Gap Analysis

### Stale Endpoints (in spec but not in code): 4

These must be **removed** from the spec:

1. `POST /api/v1/preferences/elicit`
2. `POST /api/v1/preferences/update`
3. `POST /api/v1/deliberation/deliberate`
4. `GET /api/v1/deliberation/session/{session_id}`

### Undocumented Endpoints (in code but not in spec): 43

These must be **added** to the spec:

#### Health Router (4 missing)
- `GET /ready`
- `GET /cache/stats`
- `GET /health/services`
- `GET /health/circuit-breakers`

#### Causal Router (10 missing)
- `POST /api/v1/causal/counterfactual/batch`
- `POST /api/v1/causal/counterfactual/conformal`
- `POST /api/v1/causal/transport`
- `POST /api/v1/causal/validate/strategies`
- `POST /api/v1/causal/discover/from-data`
- `POST /api/v1/causal/discover/from-knowledge`
- `POST /api/v1/causal/experiment/recommend`
- `POST /api/v1/causal/parameter-recommendations`
- `POST /api/v1/causal/sensitivity/detailed`
- `POST /api/v1/causal/extract-factors`

#### Batch Router (2 missing - entire router)
- `POST /api/v1/batch/validate`
- `POST /api/v1/batch/counterfactual`

#### Validation Router (3 missing)
- `POST /api/v1/validation/feasibility`
- `POST /api/v1/validation/coherence`
- `POST /api/v1/validation/correlations`

#### Utility Router (1 missing - entire router)
- `POST /api/v1/utility/validate`

#### Analysis Router (1 missing)
- `POST /api/v1/analysis/optimise`

#### Dominance Router (2 missing - entire router)
- `POST /api/v1/analysis/dominance`
- `POST /api/v1/analysis/pareto`

#### Risk Router (1 missing - entire router)
- `POST /api/v1/analysis/risk-adjust`

#### Threshold Router (1 missing - entire router)
- `POST /api/v1/analysis/thresholds`

#### Aggregation Router (1 missing - entire router)
- `POST /api/v1/aggregation/multi-criteria`

#### Robustness Router (2 missing)
- `POST /api/v1/robustness/analyze/v2`
- `POST /api/v1/robustness/analyze/unified`

#### Explain Router (1 missing - entire router)
- `POST /api/v1/explain/contrastive`

#### CEE Router (4 missing - entire router)
- `POST /api/v1/sensitivity/detailed`
- `POST /api/v1/contrastive`
- `POST /api/v1/conformal`
- `POST /api/v1/validation/strategies`

#### Phase4 Router (4 missing - entire router)
- `POST /api/v1/analysis/conditional-recommend`
- `POST /api/v1/analysis/sequential`
- `POST /api/v1/analysis/policy-tree`
- `POST /api/v1/analysis/stage-sensitivity`

#### Identifiability Router (2 missing - entire router)
- `POST /api/v1/analysis/identifiability`
- `POST /api/v1/analysis/identifiability/dag`

#### Decision Robustness Router (1 missing - entire router)
- `POST /api/v1/analysis/robustness`

#### Outcomes Router (4 missing - entire router)
- `GET /api/v1/outcomes/summary`
- `POST /api/v1/outcomes/log`
- `PATCH /api/v1/outcomes/{log_id}`
- `GET /api/v1/outcomes/{log_id}`

## Impact Assessment

| Metric | Value |
|--------|-------|
| Documented endpoints | 8 (valid) |
| Stale endpoints | 4 |
| Undocumented endpoints | 43 |
| **Total actual endpoints** | **51** |
| **Documentation gap** | **76%** |

### Integration Impact

- **API Client Generation:** Generated clients are missing 84% of functionality
- **Contract Testing:** Cannot test 43 endpoints
- **Cross-workstream Integration:** Teams unaware of available endpoints
- **New Developer Onboarding:** Misleading about available functionality

## Remediation Plan

1. **Create generation script** (`scripts/generate_openapi.py`) to extract spec from FastAPI
2. **Regenerate `openapi.json`** with all 51 endpoints
3. **Add CI validation** to prevent future drift
4. **Verify quality** of generated spec

## Expected Result After Remediation

| Metric | Before | After |
|--------|--------|-------|
| Documented paths | 12 | 51 |
| Stale paths | 4 | 0 |
| Missing paths | 43 | 0 |
| Documentation accuracy | 24% | 100% |

---

## Remediation Status

### Completed

- [x] Gap analysis documented (this file)
- [x] Generation script enhanced (`scripts/generate_openapi.py`)
- [x] CI validation workflow created (`.github/workflows/openapi-validation.yml`)
- [x] README updated with OpenAPI maintenance instructions

### Action Required: Regenerate OpenAPI Spec

**To complete the regeneration, run:**

```bash
# Option 1: Using Poetry (requires Python 3.11+)
poetry run python scripts/generate_openapi.py

# Option 2: Using Docker
docker build -f Dockerfile.dev -t isl-dev . && \
docker run --rm -v $(pwd):/app isl-dev python scripts/generate_openapi.py

# Option 3: Let CI do it
# Push changes, CI will fail with instructions to regenerate
```

After regenerating, verify with:
```bash
# Check the spec is current
poetry run python scripts/generate_openapi.py --check

# Verify path count (~51 expected)
python3 -c "import json; print(len(json.load(open('openapi.json'))['paths']))"
```

---

## CI Enforcement

The `.github/workflows/openapi-validation.yml` workflow will:

1. **Fail if `openapi.json` is stale** - blocking merge until regenerated
2. **Validate OpenAPI 3.x compliance** - ensuring the spec is valid
3. **Report statistics** - showing path counts and coverage

This ensures the spec stays current going forward.

---

## Key Endpoints Verification Checklist

After regenerating, verify these critical endpoints are documented with descriptions:

| Endpoint | Tag | Verified |
|----------|-----|----------|
| `/api/v1/robustness/analyze/v2` | FACET Robustness | ☐ |
| `/api/v1/causal/validate` | Causal Inference | ☐ |
| `/api/v1/causal/sensitivity/detailed` | Causal Inference | ☐ |
| `/health` | Health | ☐ |
| `/api/v1/analysis/identifiability` | Y₀ Identifiability | ☐ |
| `/api/v1/analysis/robustness` | Decision Robustness Suite | ☐ |

These endpoints should have:
- Summary and description fields populated
- Request/response models with Pydantic schemas
- Appropriate HTTP status codes documented
- Logical tag grouping

### Expected Tags After Regeneration

Based on router registrations:
- Health
- Monitoring (excluded from schema)
- Causal Inference
- Batch Processing
- Bayesian Teaching
- Advanced Validation
- Utility Functions
- Team Alignment
- Sensitivity Analysis
- Multi-Criteria Analysis
- FACET Robustness
- Contrastive Explanations
- CEE Enhancement
- Phase 4: Sequential Decisions
- Y₀ Identifiability
- Decision Robustness Suite
- Outcome Logging
