# PLoT + ISL Integration Guide

## Overview

**PLoT (Program-Oriented Learning and Teaching)** is the **SOLE** consumer of the Inference Service Layer (ISL) in PoC v01. This guide explains how PLoT integrates with ISL to add rigorous causal reasoning to agent-driven workflows.

## Architecture: PLoT-Only Consumer

### PoC v01 Architecture

```
┌─────────────────────────────────────────────┐
│          PLoT Engine (Sole Consumer)        │
│                                             │
│  - Agentic workflow orchestration           │
│  - Natural language program execution       │
│  - Causal reasoning delegation              │
└──────────────────┬──────────────────────────┘
                   │
                   │ REST API
                   │ (HTTP/JSON)
                   │
        ┌──────────▼──────────┐
        │       ISL API       │
        │                     │
        │ /causal/validate    │
        │ /causal/counterfactual │
        │ /robustness/analyze │
        └─────────────────────┘
```

**Key Architectural Points:**
- **PLoT is the ONLY caller of ISL**
- CEE (Causal Exploration Environment) does NOT call ISL directly
- UI does NOT call ISL directly
- All causal operations flow: UI → PLoT → ISL

## ISL Capabilities for PLoT

ISL provides three core capabilities for PLoT programs:

### 1. Causal Validation
**When to use:** Before making any causal claim or intervention recommendation

**What it does:**
- Determines if causal effect can be identified from observational data
- Applies do-calculus and graphical criteria
- Returns adjustment sets and formulas

**PLoT Use Cases:**
- Validate that proposed interventions have identifiable effects
- Check if sufficient data exists to estimate causal relationships
- Provide theoretical backing for recommendations

### 2. Counterfactual Prediction
**When to use:** To answer "what-if" questions about interventions

**What it does:**
- Simulates outcomes under hypothetical interventions
- Runs structural equation models
- Provides point estimates and uncertainty

**PLoT Use Cases:**
- Predict outcomes of proposed interventions
- Compare multiple intervention scenarios
- Generate quantitative recommendations

### 3. Robustness Analysis
**When to use:** For high-stakes decisions requiring confidence

**What it does:**
- Tests sensitivity to model assumptions
- Identifies fragile vs. robust recommendations
- Maps stable outcome regions

**PLoT Use Cases:**
- Validate that recommendations hold under perturbations
- Assess confidence in intervention proposals
- Identify when more data/analysis is needed

## Integration Patterns

### Pattern 1: Validate → Counterfactual → Recommend

**Scenario:** PLoT program recommends a business intervention

```typescript
async function analyzeIntervention(
  program: PLoTProgram,
  intervention: Intervention
): Promise<Recommendation> {

  // Step 1: Validate causal model
  const validation = await isl.validateCausal({
    dag: program.causalGraph,
    treatment: intervention.variable,
    outcome: program.targetOutcome
  });

  if (validation.status !== "identifiable") {
    return {
      recommendation: "INSUFFICIENT_DATA",
      reason: validation.explanation.why_not_identifiable,
      suggestedActions: [
        "Collect data on confounders",
        "Revise causal model",
        "Consider randomized experiment"
      ]
    };
  }

  // Step 2: Generate counterfactual predictions
  const counterfactual = await isl.generateCounterfactual({
    causal_model: {
      nodes: program.causalGraph.nodes,
      edges: program.causalGraph.edges,
      structural_equations: program.structuralModel
    },
    intervention: intervention.assignment,
    outcome_variables: [program.targetOutcome],
    samples: 1000
  });

  // Step 3: Return recommendation with evidence
  return {
    recommendation: "PROCEED",
    intervention: intervention,
    predictedOutcome: counterfactual.prediction.point_estimate,
    confidence: counterfactual.prediction.confidence_interval,
    causalJustification: validation.formula,
    assumptions: validation.required_assumptions
  };
}
```

**Example PLoT Program:**
```
PROGRAM PricingRecommendation:
  INPUT: current_price, historical_data

  CAUSAL_MODEL:
    nodes: [price, demand, revenue, satisfaction]
    edges: [(price → demand), (demand → revenue), (price → satisfaction)]

  ANALYZE:
    ISL.validate(treatment=price, outcome=revenue)
    IF identifiable:
      ISL.counterfactual(intervention={price: 55}, outcome=revenue)
      RECOMMEND based on predicted revenue
    ELSE:
      FLAG "Need more data or different analysis approach"
```

### Pattern 2: Robustness Check for High-Stakes Decisions

**Scenario:** PLoT validates recommendation robustness before presenting to user

```typescript
async function validateRecommendationRobustness(
  intervention: Intervention,
  prediction: CounterfactualPrediction
): Promise<RobustnessReport> {

  const robustness = await isl.analyzeRobustness({
    causal_model: {
      nodes: intervention.causalModel.nodes,
      edges: intervention.causalModel.edges,
      structural_equations: intervention.causalModel.equations
    },
    intervention_proposal: intervention.assignment,
    target_outcome: {
      [prediction.outcome]: [
        prediction.point_estimate * 0.95,  // Lower bound (-5%)
        prediction.point_estimate * 1.05   // Upper bound (+5%)
      ]
    },
    perturbation_radius: 0.1,  // ±10% model parameter changes
    min_samples: 500
  });

  return {
    isRobust: !robustness.analysis.is_fragile,
    robustnessScore: robustness.analysis.robustness_score,
    stableRegions: robustness.analysis.robust_regions,
    warnings: robustness.analysis.is_fragile
      ? ["Recommendation sensitive to assumptions - exercise caution"]
      : [],
    confidence: robustness.analysis.robustness_score > 0.8 ? "HIGH" : "MODERATE"
  };
}
```

**Example PLoT Program:**
```
PROGRAM RobustPricingRecommendation:
  INPUT: current_price, historical_data

  RECOMMEND_INTERVENTION: price=55

  VALIDATE_ROBUSTNESS:
    ISL.robustness(
      intervention={price: 55},
      target={revenue: [95000, 105000]},
      perturbation=0.1
    )

    IF robustness_score > 0.8:
      PRESENT_TO_USER with HIGH confidence
    ELSE:
      PRESENT_TO_USER with MODERATE confidence + caveats
```

### Pattern 3: Multi-Option Comparison

**Scenario:** PLoT evaluates multiple intervention options

```typescript
async function compareInterventionOptions(
  options: Intervention[],
  causalModel: CausalModel,
  targetOutcome: string
): Promise<InterventionComparison> {

  // Validate once (same for all options)
  const validation = await isl.validateCausal({
    dag: causalModel.graph,
    treatment: causalModel.interventionVariable,
    outcome: targetOutcome
  });

  if (validation.status !== "identifiable") {
    throw new Error("Cannot compare options - causal effect not identifiable");
  }

  // Generate counterfactuals for each option in parallel
  const predictions = await Promise.all(
    options.map(async (option) => {
      const cf = await isl.generateCounterfactual({
        causal_model: causalModel.toISLFormat(),
        intervention: option.assignment,
        outcome_variables: [targetOutcome],
        samples: 1000
      });

      return {
        option: option,
        prediction: cf.prediction.point_estimate,
        confidence: cf.prediction.confidence_interval
      };
    })
  );

  // Rank by predicted outcome
  predictions.sort((a, b) => b.prediction - a.prediction);

  return {
    bestOption: predictions[0],
    allOptions: predictions,
    causalJustification: validation.formula
  };
}
```

**Example PLoT Program:**
```
PROGRAM MultiOptionPricing:
  INPUT: current_price, historical_data

  OPTIONS: [
    {price: 50},
    {price: 55},
    {price: 60},
    {price: 65}
  ]

  FOR EACH option IN OPTIONS:
    ISL.counterfactual(intervention=option, outcome=revenue)

  RANK by predicted_revenue DESC
  RECOMMEND top_option WITH justification
```

## Error Handling

### ISL Error Responses

PLoT should handle ISL errors gracefully:

```typescript
async function callISLSafely<T>(
  operation: () => Promise<T>,
  fallback: T
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (error.status === 400) {
      // Invalid request - PLoT program has bug
      logError("Invalid ISL request", error);
      throw new ProgramError("Causal analysis failed - check model specification");
    } else if (error.status === 429) {
      // Rate limited - retry after delay
      await sleep(error.retryAfter * 1000);
      return await operation();
    } else if (error.status === 503) {
      // Service unavailable - degrade gracefully
      logWarning("ISL unavailable - using fallback");
      return fallback;
    } else {
      // Unknown error
      logError("ISL error", error);
      throw error;
    }
  }
}
```

### Graceful Degradation

When ISL is unavailable, PLoT should:

1. **Cache previous results:** Return cached counterfactuals if available
2. **Use heuristics:** Fall back to rule-based recommendations
3. **Inform user:** Clearly communicate that causal analysis unavailable
4. **Log for retry:** Queue request for retry when ISL recovers

```typescript
async function generateRecommendation(
  program: PLoTProgram
): Promise<Recommendation> {

  // Try ISL first
  try {
    const cf = await isl.generateCounterfactual({...});
    return createRecommendationFromISL(cf);
  } catch (error) {
    if (error.status === 503) {
      // ISL unavailable - check cache
      const cached = await cache.get(program.cacheKey);
      if (cached) {
        return {
          ...cached,
          warning: "Using cached causal analysis (ISL unavailable)"
        };
      }

      // No cache - use heuristic fallback
      return {
        recommendation: "DEFER",
        reason: "Causal analysis service temporarily unavailable",
        suggestedActions: ["Retry in a few minutes", "Proceed with caution"]
      };
    }
    throw error;
  }
}
```

## Performance Optimization

### Caching Strategies

ISL responses are deterministic for the same inputs - cache aggressively:

```typescript
class ISLCache {
  private cache = new Map<string, CachedResponse>();

  async validateCausal(request: CausalValidationRequest): Promise<ValidationResponse> {
    const cacheKey = this.generateKey("validate", request);

    const cached = this.cache.get(cacheKey);
    if (cached && !cached.isExpired()) {
      return cached.response;
    }

    const response = await isl.validateCausal(request);

    // Cache validation results for 24 hours (rarely change)
    this.cache.set(cacheKey, {
      response,
      expiresAt: Date.now() + 24 * 60 * 60 * 1000
    });

    return response;
  }

  private generateKey(endpoint: string, request: any): string {
    return `${endpoint}:${JSON.stringify(request, Object.keys(request).sort())}`;
  }
}
```

### Batch Requests

When comparing multiple options, batch ISL calls:

```typescript
// ✓ Good: Parallel requests
const predictions = await Promise.all(
  options.map(opt => isl.generateCounterfactual({intervention: opt}))
);

// ✗ Bad: Sequential requests
for (const opt of options) {
  const pred = await isl.generateCounterfactual({intervention: opt});
  // ...
}
```

### Request Prioritization

For latency-sensitive PLoT programs:

1. **Validate first (fast, ~100-300ms):** Always validate before counterfactual
2. **Counterfactual with fewer samples:** Use 100-500 samples for quick estimates
3. **Robustness only when needed:** Skip for low-stakes decisions
4. **Async robustness:** Run robustness analysis async after showing initial recommendation

```typescript
async function quickRecommendation(program: PLoTProgram): Promise<Recommendation> {
  // Fast path: validate + quick counterfactual
  const [validation, counterfactual] = await Promise.all([
    isl.validateCausal({...}),
    isl.generateCounterfactual({..., samples: 100})  // Quick estimate
  ]);

  const recommendation = createRecommendation(validation, counterfactual);

  // Robustness check runs async - update recommendation when complete
  isl.analyzeRobustness({...})
    .then(robustness => {
      recommendation.confidence = robustness.analysis.robustness_score;
      notifyUser(recommendation);
    });

  return recommendation;
}
```

## Testing PLoT + ISL Integration

### Unit Tests

Mock ISL responses for PLoT unit tests:

```typescript
describe("PLoT Intervention Analysis", () => {
  it("should recommend intervention when effect is identifiable", async () => {
    const mockISL = {
      validateCausal: jest.fn().mockResolvedValue({
        status: "identifiable",
        formula: "P(Y|do(X)) = Σ_Z P(Y|X,Z)P(Z)",
        adjustment_set: ["Z"]
      }),
      generateCounterfactual: jest.fn().mockResolvedValue({
        prediction: {
          point_estimate: 50000,
          confidence_interval: [48000, 52000]
        }
      })
    };

    const result = await analyzewithPLoT(program, mockISL);

    expect(result.recommendation).toBe("PROCEED");
    expect(result.predictedOutcome).toBe(50000);
  });
});
```

### Integration Tests

Test against real ISL (staging environment):

```typescript
describe("PLoT + ISL Integration", () => {
  const isl = new ISLClient({ baseURL: "https://isl-staging.onrender.com" });

  it("should complete full validation → counterfactual workflow", async () => {
    const validation = await isl.validateCausal({
      dag: { nodes: ["X", "Y"], edges: [["X", "Y"]] },
      treatment: "X",
      outcome: "Y"
    });

    expect(validation.status).toBe("identifiable");

    const counterfactual = await isl.generateCounterfactual({
      causal_model: {
        nodes: ["X", "Y"],
        edges: [["X", "Y"]],
        structural_equations: { Y: "2 * X + 5" }
      },
      intervention: { X: 10 },
      outcome_variables: ["Y"],
      samples: 100
    });

    expect(counterfactual.prediction.point_estimate).toBeCloseTo(25, 1);
  });
});
```

## Authentication

All ISL requests require API key:

```typescript
const isl = new ISLClient({
  baseURL: "https://isl.olumi.com",
  apiKey: process.env.ISL_API_KEY
});
```

**API Key Management:**
- Store in environment variable (never hardcode)
- Rotate keys quarterly
- Monitor usage via ISL metrics dashboard
- Request production key from isl-team@olumi.com

## Monitoring & Debugging

### Logging ISL Calls

Log all ISL interactions for debugging:

```typescript
async function loggedISLCall<T>(
  endpoint: string,
  request: any,
  response: Promise<T>
): Promise<T> {
  const requestId = generateRequestId();

  logger.info(`ISL ${endpoint} request`, {
    requestId,
    endpoint,
    request: sanitize(request)
  });

  const start = Date.now();

  try {
    const result = await response;
    const duration = Date.now() - start;

    logger.info(`ISL ${endpoint} success`, {
      requestId,
      endpoint,
      duration,
      status: "success"
    });

    return result;
  } catch (error) {
    const duration = Date.now() - start;

    logger.error(`ISL ${endpoint} failed`, {
      requestId,
      endpoint,
      duration,
      error: error.message,
      status: error.status
    });

    throw error;
  }
}
```

### Metrics to Track

Monitor these metrics in PLoT:

1. **ISL call latency:** P50, P95, P99
2. **ISL error rate:** % of failed requests
3. **Cache hit rate:** % of requests served from cache
4. **ISL cost:** If using paid tier (not applicable for PoC)

```typescript
// Track ISL metrics
metrics.recordISLCall({
  endpoint: "validateCausal",
  latency: duration,
  success: true,
  cached: false
});
```

## Support

- **ISL Documentation:** [API_QUICK_REFERENCE.md](../API_QUICK_REFERENCE.md)
- **PLoT Team Slack:** #plot-team
- **ISL Team Slack:** #isl-integration
- **Integration Issues:** isl-team@olumi.com

## Next Steps

1. **Set up ISL client** in PLoT codebase
2. **Implement caching** for ISL responses
3. **Add error handling** for graceful degradation
4. **Write integration tests** against ISL staging
5. **Monitor performance** in production

**Ready to integrate? See [PLOT_WORKSHOP_1HR.md](./PLOT_WORKSHOP_1HR.md) for hands-on exercises.**
