# Y₀ Transportability Analysis

## Overview

The transportability analysis system determines whether causal effects identified in a source domain (e.g., UK market) can be validly transported to a target domain (e.g., Germany market). Uses Y₀ selection diagrams and transportability algorithms.

## Key Features

- **Cross-Domain Validation**: Assess if effects generalize across contexts
- **Selection Diagram Construction**: Automatic identification of selection variables
- **Assumption Extraction**: Clear documentation of required assumptions
- **Testability Assessment**: Identifies which assumptions can be empirically tested
- **Robustness Scoring**: Evaluates strength of transportability claim
- **Actionable Suggestions**: Guidance when effects are not transportable

---

## API Endpoint

```
POST /api/v1/causal/transport
```

### Request Example

```json
{
  "source_domain": {
    "name": "UK",
    "dag": {
      "nodes": ["Price", "Quality", "MarketSize", "Revenue"],
      "edges": [
        ["Price", "Revenue"],
        ["Quality", "Revenue"],
        ["MarketSize", "Revenue"]
      ]
    },
    "data_summary": {
      "n_samples": 5000,
      "available_variables": ["Price", "Quality", "MarketSize", "Revenue"],
      "notes": ["E-commerce data 2023-2024", "High quality measurement"]
    }
  },
  "target_domain": {
    "name": "Germany",
    "dag": {
      "nodes": ["Price", "Quality", "MarketSize", "Revenue"],
      "edges": [
        ["Price", "Revenue"],
        ["Quality", "Revenue"],
        ["MarketSize", "Revenue"]
      ]
    },
    "data_summary": {
      "n_samples": 3000,
      "available_variables": ["Price", "Quality", "MarketSize", "Revenue"],
      "notes": ["E-commerce data 2024", "Limited historical data"]
    }
  },
  "treatment": "Price",
  "outcome": "Revenue",
  "selection_variables": ["MarketSize"]
}
```

### Response Example (Transportable)

```json
{
  "transportable": true,
  "method": "selection_diagram",
  "formula": "P_target(Revenue|do(Price)) = Σ_{MarketSize} P_source(Revenue|Price, MarketSize) P_target(MarketSize)",
  "required_assumptions": [
    {
      "type": "same_mechanism",
      "description": "The causal mechanism Price→Revenue is the same in UK and Germany",
      "critical": true,
      "testable": false
    },
    {
      "type": "no_selection_bias",
      "description": "Selection into domains doesn't affect the causal mechanism",
      "critical": true,
      "testable": true
    },
    {
      "type": "measured_selection",
      "description": "All relevant selection variables are measured: MarketSize",
      "critical": true,
      "testable": true
    },
    {
      "type": "common_support",
      "description": "The target domain has overlap in Price values with source domain",
      "critical": true,
      "testable": true
    }
  ],
  "robustness": "moderate",
  "confidence": "medium",
  "explanation": {
    "summary": "Effect can be transported from UK to Germany",
    "reasoning": "Causal structures are compatible. Effect is transportable by adjusting for selection variables: MarketSize.",
    "technical_basis": "Y₀ transportability analysis via selection_diagram method",
    "assumptions": [
      "The causal mechanism Price→Revenue is the same in UK and Germany",
      "Selection into domains doesn't affect the causal mechanism",
      "All relevant selection variables are measured: MarketSize",
      "The target domain has overlap in Price values with source domain"
    ]
  }
}
```

### Response Example (Non-Transportable)

```json
{
  "transportable": false,
  "reason": "different_mechanisms",
  "suggestions": [
    "Investigate structural differences between UK and Germany",
    "Consider if Price→Revenue mechanism differs due to context",
    "Collect data in target domain to estimate effect directly",
    "Explore domain-stratified analysis"
  ],
  "robustness": "fragile",
  "confidence": "high",
  "explanation": {
    "summary": "Effect cannot be transported from UK to Germany",
    "reasoning": "Causal mechanisms differ between UK and Germany",
    "technical_basis": "Y₀ transportability analysis - no valid transport formula found",
    "assumptions": [
      "DAG structures correct",
      "Selection variables identified"
    ]
  }
}
```

---

## How Transportability Works

### 1. Selection Diagrams

A selection diagram augments the causal DAG with selection nodes (S) that represent factors determining which domain an observation belongs to.

```
Original DAG:
  MarketSize → Revenue
  Price → Revenue

Selection Diagram:
  S_MarketSize → MarketSize
  MarketSize → Revenue
  Price → Revenue
```

The selection node S_MarketSize indicates that market size distribution differs between UK and Germany.

### 2. Transportability Conditions

Effect is transportable if we can re-express the target domain effect in terms of source domain data:

**Direct Transport** (no selection bias):
```
P_target(Y|do(X)) = P_source(Y|do(X))
```

**Adjustment Transport** (with selection variables):
```
P_target(Y|do(X)) = Σ_S P_source(Y|X,S) P_target(S)
```

### 3. Methods

| Method | When Applicable | Requirements |
|--------|----------------|--------------|
| **direct** | Identical DAGs, no selection bias | Same causal mechanism |
| **selection_diagram** | Compatible DAGs with selection variables | Measured selection variables, no selection bias on mechanism |

---

## Use Cases

### 1. Market Expansion

**Question**: "Will UK pricing strategy work in Germany?"

**Setup**:
- Source domain: UK e-commerce data
- Target domain: Germany market (limited data)
- Treatment: Price changes
- Outcome: Revenue

**Result**: Transportable if market characteristics (e.g., customer demographics) are measured and adjusted for.

### 2. Clinical Trials

**Question**: "Do trial results from young adults apply to elderly patients?"

**Setup**:
- Source domain: 18-40 age group trial data
- Target domain: 65+ age group
- Treatment: Drug dosage
- Outcome: Health outcome

**Result**: May not be transportable if biological mechanisms differ by age.

### 3. Online vs Offline

**Question**: "Will online marketing effects work for in-store campaigns?"

**Setup**:
- Source domain: Online channel data
- Target domain: Physical stores
- Treatment: Marketing spend
- Outcome: Sales

**Result**: Transportable if customer types are similar and channel effects are additive.

---

## Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_domain` | DomainSpec | Yes | - | Source domain specification |
| `target_domain` | DomainSpec | Yes | - | Target domain specification |
| `treatment` | string | Yes | - | Treatment variable name |
| `outcome` | string | Yes | - | Outcome variable name |
| `selection_variables` | List[string] | No | auto-infer | Variables that differ between domains |

### DomainSpec Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Domain name (e.g., "UK", "Germany") |
| `dag` | DAGStructure | Yes | Causal DAG for this domain |
| `data_summary` | DataSummary | No | Summary of available data |

### DataSummary Fields

| Field | Type | Description |
|-------|------|-------------|
| `n_samples` | int | Number of observations |
| `available_variables` | List[string] | Variables measured in this domain |
| `notes` | List[string] | Additional context about the data |

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `transportable` | boolean | Whether effect can be transported |
| `method` | string | Method used (if transportable): direct, selection_diagram |
| `formula` | string | Transport formula (if transportable) |
| `required_assumptions` | List[TransportAssumption] | Assumptions required for valid transport |
| `robustness` | string | Robustness assessment: robust, moderate, fragile |
| `reason` | string | Reason if not transportable |
| `suggestions` | List[string] | Suggestions if not transportable |
| `confidence` | string | Confidence in assessment: high, medium, low |
| `explanation` | ExplanationMetadata | Plain English explanation |

### TransportAssumption Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Assumption type (e.g., same_mechanism, no_selection_bias) |
| `description` | string | Human-readable explanation |
| `critical` | boolean | Whether assumption is critical for transportability |
| `testable` | boolean | Whether assumption can be empirically tested |

---

## Assumptions

All transportability claims depend on key assumptions:

### 1. Same Mechanism (Critical, Non-Testable)

**Description**: The causal mechanism X→Y is identical in source and target domains.

**Example**: Price elasticity in UK equals price elasticity in Germany.

**Risk**: If mechanisms differ (e.g., due to regulations, culture), transport fails.

### 2. No Selection Bias (Critical, Testable)

**Description**: Selection into domains doesn't affect the causal mechanism itself.

**Example**: Being in UK vs Germany doesn't change how price affects revenue.

**Test**: Check if mechanism differs conditional on measured covariates.

### 3. Measured Selection (Critical, Testable)

**Description**: All variables causing domain differences are measured.

**Example**: MarketSize, CustomerDemographics fully explain UK/Germany differences.

**Test**: Balance diagnostics, overlap assessment.

### 4. Common Support (Critical, Testable)

**Description**: Target domain has overlap in treatment values with source.

**Example**: Germany customers see similar price ranges as UK customers.

**Test**: Compare treatment distributions across domains.

---

## Robustness Levels

| Level | Meaning | Typical Scenario |
|-------|---------|------------------|
| **robust** | All testable assumptions, strong empirical support | Identical DAGs, rich data, clear overlap |
| **moderate** | 1 critical untestable assumption, some testable assumptions | Selection diagram with measured covariates |
| **fragile** | 2+ critical untestable assumptions | Different DAG structures, unmeasured selection |

---

## Failure Reasons

When `transportable = false`, the `reason` field indicates why:

| Reason | Meaning | Suggestions |
|--------|---------|-------------|
| `different_mechanisms` | DAG structures differ between domains | Investigate structural differences, collect target data |
| `no_source_path` | No causal path in source domain | Verify source DAG structure |
| `no_target_path` | No causal path in target domain | Verify target DAG structure |
| `unknown` | Transportability conditions not satisfied | Collect more information about domain differences |

---

## Best Practices

1. **Specify Selection Variables**: If you know what differs between domains (e.g., MarketSize, Demographics), specify them explicitly.

2. **Provide Data Summaries**: Include sample sizes and available variables to improve confidence assessment.

3. **Test Assumptions**: For testable assumptions (no_selection_bias, common_support), validate empirically before relying on transported effects.

4. **Start Simple**: Begin with direct transport (identical DAGs). Only use selection diagrams when necessary.

5. **Document Context**: Use domain `name` and data `notes` to document domain characteristics.

---

## Examples

### Example 1: Direct Transport (Identical Markets)

**Scenario**: UK and Ireland e-commerce (very similar markets)

```json
{
  "source_domain": {
    "name": "UK",
    "dag": {
      "nodes": ["Price", "Revenue"],
      "edges": [["Price", "Revenue"]]
    }
  },
  "target_domain": {
    "name": "Ireland",
    "dag": {
      "nodes": ["Price", "Revenue"],
      "edges": [["Price", "Revenue"]]
    }
  },
  "treatment": "Price",
  "outcome": "Revenue"
}
```

**Result**: Direct transport, high confidence, robust.

### Example 2: Selection Diagram (Different Market Sizes)

**Scenario**: UK to Germany (different market sizes)

```json
{
  "source_domain": {
    "name": "UK",
    "dag": {
      "nodes": ["Price", "MarketSize", "Revenue"],
      "edges": [
        ["MarketSize", "Revenue"],
        ["Price", "Revenue"]
      ]
    }
  },
  "target_domain": {
    "name": "Germany",
    "dag": {
      "nodes": ["Price", "MarketSize", "Revenue"],
      "edges": [
        ["MarketSize", "Revenue"],
        ["Price", "Revenue"]
      ]
    }
  },
  "treatment": "Price",
  "outcome": "Revenue",
  "selection_variables": ["MarketSize"]
}
```

**Result**: Selection diagram transport, medium confidence, moderate robustness.

### Example 3: Non-Transportable (Different Regulations)

**Scenario**: UK to Germany with regulatory differences

```json
{
  "source_domain": {
    "name": "UK",
    "dag": {
      "nodes": ["Price", "Revenue"],
      "edges": [["Price", "Revenue"]]
    }
  },
  "target_domain": {
    "name": "Germany",
    "dag": {
      "nodes": ["Price", "Regulation", "Revenue"],
      "edges": [
        ["Price", "Revenue"],
        ["Regulation", "Price"]
      ]
    }
  },
  "treatment": "Price",
  "outcome": "Revenue"
}
```

**Result**: Not transportable (different mechanisms).

**Suggestions**:
- Investigate how German regulations alter price-revenue relationship
- Collect German data to estimate effect directly
- Consider stratified analysis by regulatory regime

---

## Interpretation Guidelines

### High Confidence, Transportable

**Interpretation**: Strong evidence effect will work in target domain.

**Action**: Proceed with confidence, but monitor for unexpected differences.

### Medium Confidence, Transportable

**Interpretation**: Effect likely transportable, but some uncertainty remains.

**Action**: Validate key assumptions before full rollout. Consider pilot test.

### Low Confidence, Transportable

**Interpretation**: Transportability is theoretically possible but empirical support is weak.

**Action**: Treat as hypothesis. Validate extensively before relying on transported effect.

### Not Transportable

**Interpretation**: Effect cannot be reliably transported with current information.

**Action**: Follow suggestions. Most commonly: collect target domain data or investigate structural differences.

---

## Performance

- **Complexity**: O(n_nodes + n_edges) for selection diagram construction
- **Typical Time**: <100ms for graphs with <20 nodes
- **Scaling**: Linear with graph size

---

## Testing

### Unit Tests (28 tests)

- Direct transport scenarios
- Selection diagram transport
- Non-transportable cases
- Assumption extraction
- Robustness assessment
- Edge cases and determinism

### Integration Tests (16 tests)

- API endpoint validation
- Request/response structure
- Error handling
- Determinism verification
- Custom request ID handling

---

## Integration with Other Features

- **Causal Validation**: Validate source and target DAGs before assessing transportability
- **Counterfactual Analysis**: After confirming transportability, use counterfactual engine to estimate effects in target domain
- **Batch Counterfactuals**: Test transportability of multiple intervention scenarios simultaneously

---

## Limitations

1. **Simplified Implementation**: Current version uses structural comparison. Full Y₀ transportability algorithms (do-calculus-based) would provide more general coverage.

2. **No Partial Transportability**: Either fully transportable or not. Doesn't handle cases where part of the effect transports.

3. **Binary Selection**: Assumes discrete domain selection (UK vs Germany). Doesn't handle continuous domain variation.

4. **No Experimental Design**: Doesn't provide optimal data collection strategies for improving transportability.

---

## Future Enhancements

- **Full Y₀ Integration**: Use Y₀'s complete transportability algorithms (Bareinboim & Pearl)
- **Partial Transport**: Handle cases where only part of causal effect transports
- **Meta-Transport**: Transport across multiple domains simultaneously
- **Active Learning**: Suggest which variables to measure to enable transport
- **Sensitivity Analysis**: How robust is transportability to assumption violations?

---

## References

- Bareinboim, E., & Pearl, J. (2016). "Causal Inference and the Data-Fusion Problem"
- Pearl, J., & Bareinboim, E. (2014). "External Validity: From Do-Calculus to Transportability Across Populations"
- Y₀ Library: https://github.com/y0-causal-inference/y0

---

## Related Documentation

- [Causal Validation (Y₀)](./causal-validation.md)
- [Counterfactual Analysis (FACET)](./counterfactuals.md)
- [Batch Counterfactuals](./batch-counterfactuals.md)
