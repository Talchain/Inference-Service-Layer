# Olumi Decision Model Schema v2.6

**Status:** Canonical Reference  
**Version:** 2.6  
**Date:** 15 January 2026  
**Supersedes:** v1.1 (Boundary Contract), v2.2 (Inference Architecture), v2.3, v2.3.1, v2.3.2, v2.4, v2.5

---

## Executive Summary

This document is the **single source of truth** for Olumi's decision model schema. It consolidates:
- API boundary contracts (what crosses service boundaries)
- Inference semantics (how parameters affect computation)
- Translation rules (CEE → Canonical format mapping)
- Normalisation architecture (range-normalised coefficients)

**Key Changes from v2.3:**
- Corrected coefficient semantics: "range-normalised" not "standardised" (SD-to-SD)
- Added normalisation architecture (B.6)
- Added root vs non-root node rules (B.7)
- Added range derivation and consistency constraints (B.8)
- Added robustness threshold disclaimer
- Clarified ci_lower/ci_upper as percentiles
- Expanded std derivation formula

---

## Document Structure

| Part | Purpose | Audience |
|------|---------|----------|
| **A. Boundary Contract** | API shapes between services | All workstreams |
| **B. Parameter Semantics** | How parameters affect inference | CEE, ISL, Jinghui |
| **C. Translation Rules** | CEE → Canonical mapping | PLoT, CEE |
| **D. Validation & Limits** | Constraints and defaults | All workstreams |
| **E. Structural Invariants** | Graph requirements | CEE, PLoT |

---

# Part A: Boundary Contract

## A.1 Graph Schema

### Graph

```typescript
interface Graph {
  nodes: Node[];
  edges: Edge[];
}
```

### Node

```typescript
interface Node {
  id: string;                              // Pattern: ^[a-z0-9_:-]+$ (max 100 chars)
  kind: NodeKind;
  label: string;                           // Human-readable (max 200 chars)
  body?: string;                           // Extended description (max 2000 chars)
  
  // Factor type (NEW in v2.3)
  type?: 'numeric' | 'ordinal' | 'nominal' | 'boolean';
  categories?: string[];                   // For ordinal/nominal only
  
  // Quantitative state
  observed_state?: ObservedState;
  
  // Range specification (for normalisation)
  state_space?: StateSpace;
  
  // Goal nodes only
  goal_threshold?: number;
}

interface StateSpace {
  range?: { min: number; max: number };    // User-confirmed bounds for normalisation
}

type NodeKind = 
  | 'goal'           // Target outcome (optimisation objective)
  | 'factor'         // Causal variable (can be intervened on)
  | 'outcome'        // Observable result
  | 'risk'           // Potential negative outcome
  | 'action'         // Concrete step
  | 'decision'       // Choice point (organisational, non-causal)
  | 'option'         // Alternative within decision (organisational, non-causal)
  | 'constraint';    // Boundary condition

interface ObservedState {
  value: number;                           // Current observed value (required)
  std?: number;                            // Uncertainty in observed value (PoC: not consumed)
  baseline?: number;                       // Reference value for comparison
  unit?: string;                           // Display unit ('£', '%', 'users')
  source?: string;                         // Data provenance (free text, e.g., "Q3 Analytics Report")
}
```

**NodeKind Clarification:**

| Kind | Role | Example | In Inference? |
|------|------|---------|---------------|
| `decision` | The choice point itself | "Pricing Strategy" | ❌ No |
| `option` | An alternative within a decision | "High Price", "Low Price" | ❌ No |

Both are **organisational nodes** for UI structure. They don't participate in causal inference — the `options[]` array in `AnalysisRequest` carries intervention values to ISL.

**Node Classification for Inference:**

| Kind | Participates in Inference | Value Source |
|------|---------------------------|--------------|
| `goal` | Yes | Computed from parents |
| `factor` | Yes | Given (`observed_state.value`) or computed |
| `outcome` | Yes | Computed from parents |
| `risk` | Yes | Computed from parents |
| `action` | Yes | Computed from parents |
| `decision` | **No** — filtered out | N/A |
| `option` | **No** — filtered out | N/A |
| `constraint` | **No** — filtered out | N/A |

### Edge (Canonical Format)

This is the format used by **ISL** and the **canonical reference**.

```typescript
interface Edge {
  from: string;                            // Source node ID
  to: string;                              // Target node ID
  exists_probability: number;              // Structural uncertainty [0, 1]
  strength: StrengthDistribution;          // Parametric uncertainty
  label?: string;                          // Human-readable description
}

interface StrengthDistribution {
  mean: number;                            // Effect size [-1, +1], signed
  std: number;                             // Uncertainty, must be > 0.001
}
```

**Constraints:**
- `exists_probability`: [0.0, 1.0]
- `strength.mean`: [-1.0, +1.0] — range-normalised effect coefficient
- `strength.std`: > 0.001 (strictly positive, floor enforced by PLoT)

---

## A.2 Options & Interventions Schema

### Option

```typescript
interface Option {
  id: string;                              // Unique identifier
  label: string;                           // Human-readable name (max 500 chars)
  description?: string;                    // Extended description
  interventions: Record<string, number>;   // node_id → intervention value
  status?: OptionStatus;
}

type OptionStatus = 'ready' | 'needs_user_mapping' | 'needs_encoding';
```

**PoC Constraint:** Interventions must be numeric. Nominal categorical interventions are not supported (see Part D.3).

---

## A.3 Analysis Request Schema

### AnalysisRequest (UI → PLoT)

```typescript
interface AnalysisRequest {
  graph: Graph;
  options: Option[];
  goal_node_id: string;                    // Must exist in graph.nodes
  goal_threshold?: number;                 // Success threshold (user-provided, same units as factor values)
  seed?: string;                           // Default: "42"
  detail_level?: 'quick' | 'standard' | 'deep';
  include_voi?: boolean;                   // Request Value of Information (default: false)
  request_id?: string;                     // For tracing
}
```

**`goal_threshold` clarification:** User-provided target value (e.g., "I want revenue above £100,000"). Must be in the **same units** as factor values. ISL computes `probability_of_goal` = P(outcome ≥ threshold).

**`include_voi` clarification:** When `true`, ISL computes Value of Information analysis — tells users whether gathering more data is worth it before deciding.

---

## A.4 Analysis Response Schema

### AnalysisResponse (PLoT → UI)

```typescript
interface AnalysisResponse {
  // Status
  analysis_status: AnalysisStatus;
  status_reason?: string;
  
  // Per-feature status
  option_comparison_status: FeatureStatus;
  robustness_status: FeatureStatus;
  drivers_status: FeatureStatus;
  
  // Validation issues
  critiques: Critique[];
  
  // Results
  option_results?: OptionResult[];
  robustness?: RobustnessResult;
  factor_sensitivity?: FactorSensitivityResult[];
  value_of_information?: ValueOfInformationResult[];
  
  // CEE synthesis
  robustness_synthesis?: RobustnessSynthesis;
  
  // Metadata
  meta: ResponseMeta;
}

type AnalysisStatus = 'computed' | 'partial' | 'blocked' | 'failed';
type FeatureStatus = 'computed' | 'unavailable' | 'skipped' | 'error';
```

### OptionResult

```typescript
interface OptionResult {
  option_id: string;
  option_label: string;
  outcome: OutcomeDistribution;
  win_probability: number;                 // [0, 1]
  probability_of_goal?: number;            // [0, 1] when threshold provided
}

interface OutcomeDistribution {
  mean: number;
  std: number;
  median: number;
  ci_lower: number;                        // 5th percentile of outcome distribution
  ci_upper: number;                        // 95th percentile of outcome distribution
  p10?: number;
  p90?: number;
}
```

**Clarification:** `ci_lower`/`ci_upper` are the 5th/95th percentiles of the **outcome distribution** — the range of expected outcome values. These are NOT probability bounds. For "probability of achieving target X", see `probability_of_goal` (requires `goal_threshold` in request).

### RobustnessResult

**Enrichment Note:** ISL returns ID-only computational results. PLoT enriches with labels (e.g., `from_label`, `to_label`, `alternative_winner_label`) via graph lookup before returning to UI. This is the intended architecture — ISL is an inference engine; labels are presentation metadata.

```typescript
interface RobustnessResult {
  recommendation_stability: number;        // [0, 1] P(recommended option stays best)
  confidence: number;                      // [0, 1] MC sampling confidence
  level: RobustnessLevel;
  is_robust: boolean;                      // recommendation_stability >= 0.7
  fragile_edges: FragileEdge[];
  robust_edges: string[];
  recommended_option_id: string;
  recommended_option_label: string;
}

type RobustnessLevel = 'high' | 'moderate' | 'low' | 'very_low';

interface FragileEdge {
  edge_id: string;                         // Format: "{from_id}->{to_id}"
  from_id: string;
  to_id: string;
  from_label: string;
  to_label: string;
  alternative_winner_id: string | null;
  alternative_winner_label: string | null;
  switch_probability: number | null;
}
```

**Note:** Robustness thresholds (`is_robust: recommendation_stability >= 0.7`) are **operational defaults** for PoC usability, not research-validated. Scientific validation pending Neil consultation.

### FactorSensitivityResult

```typescript
interface FactorSensitivityResult {
  node_id: string;
  node_label: string;
  elasticity: number;
  importance_rank: number;
  observed_value?: number;
  direction: 'positive' | 'negative';
  interpretation: string;
}
```

### ResponseMeta

```typescript
interface ResponseMeta {
  seed_used: string;
  n_samples: number;
  response_hash: string;
  request_id?: string;
  processing_time_ms: number;
  computed_at: string;                     // ISO 8601
  analysis_mode?: 'isl_only' | 'hybrid_local_primary';
  goal_threshold_used?: number;
}
```

---

# Part B: Parameter Semantics for Inference

This section explains how each parameter is used in Monte Carlo simulation. **CEE must generate reasonable values for accurate results.**

## B.1 Monte Carlo Algorithm

**Note:** All inference occurs in **normalised [0,1] space** (see B.6). Factor values are normalised before computation, results are denormalised after.

```
For each sample (1 of n_samples, default 1000):
  For each edge:
    1. edge_active = Bernoulli(exists_probability)
    2. IF edge_active:
         β_sample = Normal(strength.mean, strength.std)
       ELSE:
         β_sample = 0
    3. contribution = β_sample × normalised_parent_value
    4. child_value += contribution
  
  For each node (topological order):
    value = intercept + sum(contributions from parents)
```

## B.2 Edge Parameters

| Parameter | Range | Mathematical Role | Effect on Results |
|-----------|-------|-------------------|-------------------|
| `strength.mean` | **[-1, +1]** | Coefficient β in structural equation | Determines magnitude and direction of causal effect |
| `strength.std` | > 0.001 | Epistemic uncertainty about effect size | Higher → wider outcome distributions |
| `exists_probability` | [0, 1] | Structural uncertainty: Bernoulli draw per sample | Lower → edge sometimes excluded |

### Strength Mean: Range-Normalised Effect Coefficient

`strength.mean` is a **range-normalised effect coefficient**, not a statistically standardised (SD-to-SD) coefficient.

**Definition:**

> Moving the parent from its minimum plausible value to its maximum plausible value produces an expected change of `mean` × (child's range) in the child outcome.

| Value | Interpretation | Example |
|-------|----------------|---------|
| +1.0 | Strong positive: parent's full range → child's full range | "Price directly determines revenue" |
| +0.6 | Moderate positive: parent's full range → 60% of child's range | "Marketing noticeably increases awareness" |
| +0.3 | Weak positive | "Weather slightly affects foot traffic" |
| 0.0 | No effect | — |
| -0.3 | Weak negative | "Complexity slightly reduces adoption" |
| -0.6 | Moderate negative | "Price increase reduces demand" |
| -1.0 | Strong negative: parent's full range → child's full range (inverse) | "Competitor launch directly hurts market share" |

**Sign convention:** Direction encoded in sign. Negative mean = inverse relationship.

**Important:** This is NOT a statistically standardised (SD-to-SD) coefficient. True SD standardisation would require population standard deviations, which are typically unavailable in decision-making contexts.

### Strength Std: Epistemic Uncertainty

`strength.std` represents **epistemic uncertainty** about the effect size — our confidence in the coefficient estimate — not a statistical population parameter.

| Value | Interpretation | When to Use |
|-------|----------------|-------------|
| 0.05–0.10 | High confidence in the effect estimate | Direct mechanical relationships, strong evidence |
| 0.10–0.20 | Moderate confidence | Empirically observed relationships |
| 0.20–0.30 | Low confidence — effect could vary significantly | Hypothesised or variable effects |
| > 0.30 | Very uncertain — consider gathering more evidence | Speculative or context-dependent |

### Exists Probability: Structural Uncertainty

| Value | Interpretation | When to Use |
|-------|----------------|-------------|
| 1.0 | Certain | Definitional edges (decision→option) |
| 0.90–0.99 | Near-certain | Well-documented causal relationships |
| 0.70–0.90 | Likely | Observed but not guaranteed |
| 0.50–0.70 | Uncertain | Hypothesised relationships |
| 0.30–0.50 | Speculative | "Might exist" relationships |

## B.3 Node Parameters

| Parameter | Default | Mathematical Role |
|-----------|---------|-------------------|
| `intercept` | 0.0 | Constant in structural equation: Y = intercept + Σ(β × X) |
| `observed_state.value` | — | Factor's current value in user units |
| `observed_state.baseline` | — | Reference for "change from baseline" calculations |

## B.4 What Makes Values "Reasonable"

| Parameter | Unreasonable | Why It's a Problem |
|-----------|--------------|-------------------|
| `strength.mean` | All edges = 0.5 | No differentiation → identical option outcomes |
| `strength.mean` | \|mean\| > 1.0 | Implies amplification (rare in business contexts) |
| `strength.std` | std > \|mean\| | Sign might flip (extremely uncertain) |
| `strength.std` | All edges = same value | Ignores evidence quality differences |
| `exists_probability` | All = 1.0 | Ignores structural uncertainty entirely |
| `exists_probability` | < 0.3 | Why include edge if you doubt it exists? |

**Warning:** If all edges have `strength.mean = 0.5`, this indicates LLM output was lost in the pipeline.

## B.5 Goal Threshold and Units

`goal_threshold` must be in the **same units** as your factor values.

**ISL comparison (no internal conversion):**
```python
probability_of_goal = count(samples >= goal_threshold) / n_samples
```

**Example:**
- Factor values in £k (e.g., `observed_state.value: 59`)
- Goal threshold in £k (e.g., `goal_threshold: 100`)
- ISL computes: P(outcome ≥ 100)

**Note:** ISL is unit-agnostic. It computes with whatever values it receives. PLoT handles normalisation/denormalisation (see B.6).

## B.6 Normalisation Architecture

All inference occurs in **normalised [0,1] space** for coefficient comparability. Users interact in **their own units** for usability.

### Data Flow

```
User (£59k) → PLoT normalises (0.43) → ISL computes → PLoT denormalises (£86k) → UI
```

### Why Normalisation Matters

Without normalisation, factors with larger absolute values dominate:
```
Price (£50,000) × 0.5 = 25,000 contribution
Satisfaction (4.5) × 0.5 = 2.25 contribution
```
Same coefficient, but Price dominates 10,000×.

With normalisation, coefficients become comparable:
```
Price (0.43 normalised) × 0.5 = 0.215 contribution
Satisfaction (0.75 normalised) × 0.5 = 0.375 contribution
```

### What This Enables

- Coefficients represent "effect per unit of decision-relevant variation"
- Comparable within a single analysis (same derivation method)
- User can see and correct ranges if CEE extraction is wrong

### What This Does NOT Provide

- Statistical standardisation (SD-to-SD)
- Universal comparability across different decisions

## B.7 Root vs Non-Root Nodes

| Node Type | Has Intercept? | Has Noise? |
|-----------|---------------|------------|
| **Root** (no incoming edges) | ❌ No | ✅ Yes (`node_noise` distribution) |
| **Non-root** (has parents) | ✅ Yes (default: 0.0) | ❌ No |

**Root nodes** sample from their noise distribution. **Non-root nodes** compute value from parents via:
```
value = intercept + Σ(β × parent_value)
```

**PoC behaviour:** Root nodes use `observed_state.value` directly as their value. Full stochastic sampling from `node_noise` distributions is deferred to post-PoC.

## B.8 Range Derivation and Consistency

### Range Derivation Priority

PLoT derives normalisation range from available information:

| Priority | Source | Derivation |
|----------|--------|------------|
| 1 | User-confirmed range | `state_space.range: { min, max }` |
| 2 | Intervention spread | min/max of option interventions + 20% padding |
| 3 | Baseline + value | Implied range from change context |
| 4 | Heuristic fallback | Value ± 50% |

```typescript
function deriveRange(factor: Factor, options: Option[]): Range {
  // 1. Explicit range
  if (factor.state_space?.range) return factor.state_space.range;
  
  // 2. Intervention spread
  const interventions = getInterventionsForFactor(factor.id, options);
  if (interventions.length >= 2) {
    const min = Math.min(...interventions);
    const max = Math.max(...interventions);
    const padding = (max - min) * 0.2;
    return { min: min - padding, max: max + padding };
  }
  
  // 3. Baseline + value
  if (factor.observed_state?.baseline !== undefined) {
    const value = factor.observed_state.value;
    const baseline = factor.observed_state.baseline;
    const delta = Math.abs(value - baseline);
    return { 
      min: Math.min(baseline, value) - delta * 0.5,
      max: Math.max(baseline, value) + delta * 0.5 
    };
  }
  
  // 4. Heuristic: value ± 50%
  const value = factor.observed_state?.value ?? 1;
  return { min: value * 0.5, max: value * 1.5 };
}
```

### Edge Cases

| Input | Problem | Handling |
|-------|---------|----------|
| `value = 0` | Zero range → division by zero | Require explicit `state_space.range` or use `{ min: -1, max: 1 }` |
| `value` is percentage (unit='%') | Heuristic may exceed [0, 100] | Clamp to natural bounds `{ min: 0, max: 100 }` |
| Intervention outside range | Range doesn't contain all options | Extend range to include intervention + emit warning |

### Intervention Range Validation

When an intervention value falls outside the derived range:

```typescript
if (interventionValue < range.min || interventionValue > range.max) {
  // Extend range to accommodate
  range.min = Math.min(range.min, interventionValue);
  range.max = Math.max(range.max, interventionValue);
  
  // Emit warning
  warnings.push({
    code: 'INTERVENTION_EXTENDS_RANGE',
    severity: 'warning',
    message: `Intervention value ${interventionValue} for ${factorId} extends derived range.`
  });
}
```

### Range Consistency Constraint

For coefficients to be comparable across factors, ranges must be derived **consistently**.

**Invariant:** All factors in a single analysis should use the same derivation priority chain. Mixed derivation methods produce a validation warning:

```typescript
{
  code: 'INCONSISTENT_RANGE_DERIVATION',
  severity: 'warning',
  message: 'Factors use different range derivation methods. Coefficients may not be directly comparable.'
}
```

### Range Transparency

Derived ranges should be:
- **Explicit** — derived from known sources
- **Reproducible** — deterministic derivation algorithm  
- **Inspectable** — surfaced in UI for user confirmation

This enables users to understand and challenge range assumptions.

---

# Part C: Translation Rules

CEE outputs **flat format** for LLM reliability. PLoT normalises to **canonical nested format** for ISL.

## C.1 CEE Output Format (Flat)

```typescript
interface CEEEdge {
  from: string;
  to: string;
  strength_mean: number;                   // [-1, +1]
  strength_std: number;                    // > 0
  belief_exists: number;                   // [0, 1]
  effect_direction: 'positive' | 'negative';
  provenance?: StructuredProvenance;
  
  // Legacy (deprecated, may still appear)
  weight?: number;
  belief?: number;
}
```

## C.2 Translation Mapping

| CEE Field | Canonical Field | Transformation |
|-----------|-----------------|----------------|
| `from` | `from` | Pass-through |
| `to` | `to` | Pass-through |
| `belief_exists` | `exists_probability` | Rename, clamp [0, 1] |
| `strength_mean` | `strength.mean` | Nest, validate [-1, +1] |
| `strength_std` | `strength.std` | Nest, floor at 0.001 |
| `effect_direction` | (encoded in sign) | Applied to `strength.mean` if needed |

**Legacy fallback chain:**
```typescript
exists_probability = edge.exists_probability 
                  ?? edge.belief_exists 
                  ?? edge.belief 
                  ?? 0.8

strength.mean = edge.strength?.mean 
             ?? edge.strength_mean 
             ?? applyDirection(edge.weight, edge.effect_direction)
             ?? 0.5  // WARNING: indicates lost data
```

## C.3 PLoT Normaliser Location

File: `graph-normaliser.ts`
- Lines 201-289: Edge normalisation
- Lines 171-184: Direction inference
- Line 51: Minimum std enforcement (0.001)

---

# Part D: Validation & Limits

## D.1 PoC Limits (Enforced)

| Limit | Value | Enforced By |
|-------|-------|-------------|
| Max nodes | 50 | PLoT |
| Max edges | 100 | PLoT |
| Max options | 10 | PLoT |
| Min options | 2 | PLoT |
| n_samples range | 100–10,000 | PLoT |
| Default n_samples | 1,000 | PLoT |
| Min strength.std | 0.001 | PLoT normaliser |
| Request body size | 10 MB | PLoT |

## D.2 Validation Rules

### Edge Validation

| Rule | Code | Severity |
|------|------|----------|
| `exists_probability` in [0, 1] | `INVALID_EXISTS_PROBABILITY` | error |
| `strength.std` > 0 | `INVALID_STRENGTH_STD` | error |
| `strength.mean` is finite | `INVALID_STRENGTH_MEAN` | error |
| `strength.mean` in [-1, +1] | `STRENGTH_OUT_OF_RANGE` | warning |
| All edges same `strength.mean` | `NO_COEFFICIENT_VARIATION` | warning |

### Node Validation

| Rule | Code | Severity |
|------|------|----------|
| Node ID matches pattern `^[a-z0-9_:-]+$` | `INVALID_NODE_ID` | error |
| All edge references resolve | `REFERENCE_INTEGRITY` | error |
| Factor has `observed_state.value` for inference | `MISSING_FACTOR_VALUE` | warning |

### Graph Validation

| Rule | Code | Severity |
|------|------|----------|
| Graph is acyclic | `GRAPH_HAS_CYCLE` | error |
| At least one outcome or risk node | `MISSING_OUTCOME_OR_RISK` | error |
| Goal node exists | `MISSING_GOAL_NODE` | error |
| Path exists from factors to goal | `NO_PATH_TO_GOAL` | error |
| Max nodes not exceeded | `POC_NODE_LIMIT` | error |
| Max edges not exceeded | `POC_EDGE_LIMIT` | error |

### Robustness Threshold Note

Robustness thresholds (`is_robust: recommendation_stability >= 0.7`, `level` categorisation) are **operational defaults** for PoC usability. These are not research-validated thresholds. Scientific validation is pending Neil consultation.

## D.3 Categorical Type Handling (PoC)

**PoC Constraint:** Nominal categorical interventions are **not supported**.

### Type Definitions

```typescript
type FactorType = 'numeric' | 'ordinal' | 'nominal' | 'boolean';
```

| Type | Description | Numeric Intervention Valid? |
|------|-------------|----------------------------|
| `numeric` | Continuous values (price, budget) | ✅ Yes |
| `ordinal` | Ordered categories (Basic/Pro/Enterprise) | ✅ Yes (rank encoding) |
| `nominal` | Unordered alternatives (UK/US/EU) | ❌ **No** |
| `boolean` | Binary (true/false) | ✅ Yes (0/1) |

### Validation Rule

```typescript
if (factor.type === 'nominal' && hasIntervention(factor.id)) {
  throw new ValidationError(
    'NOMINAL_INTERVENTION_NOT_SUPPORTED',
    'PoC does not support interventions on nominal categories. ' +
    'Reframe as numeric factor (e.g., "market attractiveness score").'
  );
}
```

### CEE Guidance

When extracting factors:
- If values have natural progression (Low/Medium/High, Basic/Pro/Enterprise), mark as `ordinal`
- If values are unordered alternatives (UK/US/EU, Hire/Outsource), mark as `nominal`
- Never assign arbitrary numeric codes to nominal categories

## D.4 Default Values

| Field | Default | Trigger | Implication |
|-------|---------|---------|-------------|
| `strength.mean` | 0.5 | LLM output missing | ⚠️ Causes uniform effects — indicates pipeline bug |
| `strength.std` | Derived | Not provided | See derivation formula below |
| `exists_probability` | 0.8 | Not provided | Reasonable default |
| `intercept` | 0.0 | Not provided | Effects are relative to baseline |

### Strength Std Derivation Formula

When `strength.std` is not explicitly provided:

```typescript
const cv = 0.3 * (1 - exists_probability) + 0.1;  // cv ∈ [0.1, 0.4]
const std = Math.max(0.001, cv * Math.abs(strength_mean));
```

| exists_probability | cv | Example std (for mean=0.6) |
|---------------|-----|---------------------------|
| 0.9 (confident) | 0.13 | 0.078 |
| 0.7 (likely) | 0.19 | 0.114 |
| 0.5 (uncertain) | 0.25 | 0.150 |

**Diagnostic:** If you see all edges with `strength.mean = 0.5`, check for Zod schema stripping or LLM output parsing failures.

## D.5 Factor Sensitivity (PoC)

**PoC Mechanism:** Factor sensitivity is derived from **edge uncertainty** parameters (`strength.std`, `exists_probability`), not node-level parameter uncertainties. This is an edge-uncertainty-driven proxy — true factor/parameter uncertainty analysis requires explicit `parameter_uncertainties` input (post-PoC scope).

| Condition | `drivers_status` | UI Treatment |
|-----------|------------------|--------------|
| Edge uncertainties present | `computed` | Show factor sensitivity panel |
| Edge uncertainties absent | `skipped` | "Drivers analysis requires uncertainty estimates" |
| ISL computation fails | `error` | "Factor analysis temporarily unavailable" |

**Note:** `skipped` is not a computation failure — it indicates the preconditions for factor sensitivity were not met. Values of zero are valid when computed; do not treat zeros as failures.

**Node-level uncertainty:** `observed_state.std`, if present, is **not consumed** by inference or sensitivity in PoC. Uncertainty modelling is edge-level only. Future use of node-level uncertainty must apply normalisation consistently.

---

# Part E: Structural Invariants

These invariants **must** be validated at CEE generation and PLoT ingestion.

## E.1 Required Invariants

```typescript
const STRUCTURAL_INVARIANTS = [
  // Topology
  'Graph must be acyclic (DAG)',
  'All edge.from and edge.to must reference existing node IDs',
  
  // Causal structure
  'At least one node with kind = outcome OR kind = risk must exist',
  'At least one causal path must connect factors to goal',
  'Goal node must exist and be reachable',
  
  // Inference exclusions
  'Nodes with kind = decision | option | constraint do not participate in inference',
  
  // Options
  'At least 2 options required for comparison',
  'All intervention targets must reference existing factor nodes',
  'For each Option in options[], id must match a graph.nodes[kind=option].id',
];
```

## E.2 Edge Direction Rules

| Edge Type | Expected Direction | Enforced? |
|-----------|-------------------|-----------|
| factor → outcome | Positive or negative | No (domain-dependent) |
| factor → risk | Typically positive | No |
| risk → goal | Negative | Inferred by PLoT |
| decision → option | N/A (excluded) | N/A |
| option → factor | N/A (sets intervention) | N/A |

---

# Appendix A: Critique Codes

| Code | Severity | Message |
|------|----------|---------|
| `MISSING_GOAL_NODE` | blocker | Goal node not found in graph |
| `NO_OPTIONS` | blocker | At least 2 options required |
| `EMPTY_INTERVENTIONS` | blocker | Option has no interventions |
| `NO_PATH_TO_GOAL` | blocker | No causal path from factors to goal |
| `GRAPH_HAS_CYCLE` | blocker | Graph contains cycles |
| `INVALID_INTERVENTION_TARGET` | blocker | Intervention targets non-existent node |
| `MISSING_OUTCOME_OR_RISK` | blocker | No outcome or risk nodes in graph |
| `NOMINAL_INTERVENTION_NOT_SUPPORTED` | blocker | Cannot intervene on nominal categories |
| `POC_NODE_LIMIT` | blocker | Exceeded 50 node limit |
| `POC_EDGE_LIMIT` | blocker | Exceeded 100 edge limit |
| `OPTION_ID_MISMATCH` | error | Option ID does not match any option node in graph |
| `INVALID_EXISTS_PROBABILITY` | error | exists_probability not in [0, 1] |
| `INVALID_STRENGTH_STD` | error | strength.std must be > 0 |
| `STRENGTH_OUT_OF_RANGE` | warning | strength.mean outside [-1, +1] |
| `NO_COEFFICIENT_VARIATION` | warning | All edges have same strength.mean |
| `MISSING_FACTOR_VALUE` | warning | Factor missing observed_state.value |
| `INCONSISTENT_RANGE_DERIVATION` | warning | Factors use different range derivation methods |
| `INTERVENTION_EXTENDS_RANGE` | warning | Intervention value extends derived range |

---

# Appendix B: Workstream Responsibilities

| Workstream | Emits | Consumes | Key Files |
|------------|-------|----------|-----------|
| **CEE** | Flat edge format, nodes, options | User brief | `draft-graph.ts` |
| **PLoT** | Canonical format, AnalysisResponse | AnalysisRequest, CEE output | `graph-normaliser.ts`, `translator-v3.ts` |
| **ISL** | RobustnessResult, OptionResult | Canonical edge format | `/api/v1/robustness/analyze/v2` |
| **UI** | AnalysisRequest | AnalysisResponse | `useResultsSectionData.ts` |

---

# Appendix C: Migration from v2.2/v1.1

| Old Document | Section | New Location |
|--------------|---------|--------------|
| v1.1 AnalysisRequest/Response | — | Part A.3, A.4 |
| v1.1 Critique codes | — | Appendix A |
| v2.2 Three-layer architecture | — | Conceptual only (not in v2.3.1) |
| v2.2 Mechanism definitions | — | Part B.1 (simplified) |
| v2.2 Seed computation | — | Removed (implementation detail) |
| v2.2 PoC limits (12/20) | — | Corrected to 50/100 in Part D.1 |
| v1.1 strength range (-3/+3) | — | Corrected to [-1, +1] in Part B.2 |

**Note:** PoC Technical Specification v03 incorrectly states strength range as [-3, +3]. This was corrected to [-1, +1] per Neil's guidance (4 Jan 2026).

---

# Appendix D: Document History

| Version | Date | Changes |
|---------|------|---------|
| v1.1 | 6 Jan 2026 | Boundary contract (superseded) |
| v2.2 | Dec 2025 | Inference architecture (superseded) |
| v2.3 | 14 Jan 2026 | Unified schema |
| v2.3.1 | 15 Jan 2026 | Corrected coefficient semantics, added normalisation architecture |
| v2.3.2 | 15 Jan 2026 | Added `state_space` to Node interface, fixed formula inconsistency, added edge case handling for range derivation, added `INTERVENTION_EXTENDS_RANGE` critique code, clarified root node PoC behaviour |
| v2.4 | 15 Jan 2026 | Added Option ID consistency invariant, `OPTION_ID_MISMATCH` critique code, ISL/PLoT enrichment architecture note, and D.5 Factor Sensitivity (PoC) clarification |
| v2.5 | 15 Jan 2026 | Clarified D.5: factor sensitivity is edge-uncertainty-driven proxy (not node-level parameter uncertainty); zeros are valid computed values |
| **v2.6** | **15 Jan 2026** | Added `observed_state.std` to schema (PoC: not consumed); clarified node-level uncertainty is not used in PoC inference/sensitivity |

---

# Appendix E: Outstanding Questions for Neil

1. **Categorical strategy validation:** Is Option A (block nominal, force reframe to numeric proxy) scientifically acceptable for PoC?

2. **Robustness thresholds:** Are the current thresholds (≥0.7 = robust) scientifically grounded or should they be adjusted?

3. **Distribution for bounded variables:** For `strength.mean` bounded to [-1, +1], should we implement scaled Beta distribution (`ScaledBeta = Beta(α, β) × 2 - 1`) to guarantee samples stay bounded, rather than Normal with implicit clamping?

4. **Range-normalised coefficients:** Does the range-normalised approach (rather than SD-standardised) meet scientific requirements for SCM-based decision analysis, given we lack population statistics for true standardisation?

---

*End of Document*
