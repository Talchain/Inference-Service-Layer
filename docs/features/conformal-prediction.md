# Conformal Prediction for Counterfactuals

## Overview

Conformal prediction provides **finite-sample valid prediction intervals** with provable coverage guarantees for counterfactual analysis. Unlike traditional Monte Carlo methods that offer asymptotic coverage, conformal intervals guarantee coverage for any sample size.

## Key Value Proposition

**Traditional Monte Carlo:** "This interval has ~95% coverage (based on asymptotic theory and parametric assumptions)"

**Conformal Prediction:** "This interval has provable ≥94.7% coverage (finite-sample guarantee, distribution-free)"

## Core Properties

- **Distribution-Free**: No parametric assumptions about the underlying distribution
- **Finite-Sample Valid**: Guarantees hold for any sample size, not just asymptotically
- **Adaptive**: Intervals automatically adapt to local uncertainty
- **Honest**: Often wider than Monte Carlo intervals, providing more realistic uncertainty quantification

## API Endpoint

```
POST /api/v1/causal/counterfactual/conformal
```

### Request Example

```json
{
  "model": {
    "variables": ["Price", "Quality", "Revenue"],
    "equations": {
      "Revenue": "10000 + 500*Price + 200*Quality + 50*Price*Quality"
    },
    "distributions": {
      "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
    }
  },
  "intervention": {"Price": 50, "Quality": 8},
  "calibration_data": [
    {"inputs": {"Price": 40, "Quality": 7}, "outcome": {"Revenue": 30000}},
    {"inputs": {"Price": 45, "Quality": 7.5}, "outcome": {"Revenue": 32500}},
    {"inputs": {"Price": 50, "Quality": 8}, "outcome": {"Revenue": 35000}},
    // ... (at least 10 calibration points required)
  ],
  "confidence_level": 0.95,
  "method": "split",
  "samples": 1000,
  "seed": 42
}
```

### Response Example

```json
{
  "prediction_interval": {
    "lower_bound": {"Revenue": 48000},
    "upper_bound": {"Revenue": 56000},
    "point_estimate": {"Revenue": 52000},
    "interval_width": {"Revenue": 8000}
  },
  "coverage_guarantee": {
    "nominal_coverage": 0.95,
    "guaranteed_coverage": 0.9474,
    "finite_sample_valid": true,
    "assumptions": [
      "Exchangeability of calibration and test points"
    ]
  },
  "calibration_quality": {
    "calibration_size": 50,
    "residual_statistics": {
      "mean": 2500,
      "std": 1200,
      "median": 2300,
      "iqr": 1500
    },
    "interval_adaptivity": 0.48
  },
  "comparison_to_standard": {
    "monte_carlo_interval": {
      "Revenue": {"lower": 49000, "upper": 55000, "confidence_level": 0.95}
    },
    "conformal_interval": {
      "Revenue": [48000, 56000]
    },
    "width_ratio": {
      "Revenue": 1.33
    },
    "interpretation": "Conformal interval is 33% wider, providing more honest uncertainty quantification with finite-sample guarantees"
  },
  "explanation": {
    "summary": "Conformal prediction provides 94.7% guaranteed coverage: Revenue interval [48000, 56000] around point estimate 52000",
    "reasoning": "Using split conformal prediction with 50 calibration points. Conformal interval is 33% wider.",
    "technical_basis": "Finite-sample conformal prediction (Vovk et al. 2005)",
    "assumptions": ["Exchangeability of calibration and test points"]
  }
}
```

---

## How Conformal Prediction Works

### Split Conformal Algorithm

1. **Split Calibration Data** (50/50)
   - Training set: Used implicitly (not required for split conformal)
   - Calibration set: Used to compute conformity scores

2. **Compute Conformity Scores**
   - For each calibration point: `R_i = |Y_i - Ŷ_i|`
   - These are absolute residuals (how far predictions miss)

3. **Find Quantile**
   - For coverage level 1-α, compute quantile:
   - `q = quantile(R, ⌈(n+1)(1-α)⌉/n)`
   - Finite-sample correction ensures coverage guarantee

4. **Form Interval**
   - For new prediction Ŷ_new:
   - Interval = `[Ŷ_new - q, Ŷ_new + q]`

### Coverage Guarantee

**Theorem**: Under exchangeability, the conformal interval satisfies:

```
P(Y_new ∈ [Ŷ_new - q, Ŷ_new + q]) ≥ (⌈(n+1)(1-α)⌉ - 1) / n
```

**Example**: With n=19 calibration points and α=0.05:
- Guaranteed coverage = (⌈20 × 0.95⌉ - 1) / 19 = 18/19 ≈ 0.947 = 94.7%

---

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | StructuralModel | Yes | - | Structural causal model |
| `intervention` | Dict[str, float] | Yes | - | Intervention to apply |
| `calibration_data` | List[ObservationPoint] | Yes | - | Historical observations (≥10 required) |
| `confidence_level` | float | No | 0.95 | Target coverage (1-α) |
| `method` | string | No | "split" | Conformal method: split, cv+, jackknife+ |
| `samples` | int | No | 1000 | Monte Carlo samples for comparison |
| `seed` | int | No | None | Random seed for reproducibility |

### ObservationPoint Structure

```json
{
  "inputs": {"Price": 45, "Quality": 8},
  "outcome": {"Revenue": 32500}
}
```

---

## Response Fields

### Prediction Interval

```json
{
  "lower_bound": {"Revenue": 48000},
  "upper_bound": {"Revenue": 56000},
  "point_estimate": {"Revenue": 52000},
  "interval_width": {"Revenue": 8000}
}
```

### Coverage Guarantee

```json
{
  "nominal_coverage": 0.95,           // Requested coverage
  "guaranteed_coverage": 0.9474,      // Provable coverage
  "finite_sample_valid": true,        // Always true for conformal
  "assumptions": ["Exchangeability"]  // Required assumption
}
```

### Calibration Quality

```json
{
  "calibration_size": 50,
  "residual_statistics": {
    "mean": 2500,    // Average residual
    "std": 1200,     // Residual variability
    "median": 2300,  // Median residual
    "iqr": 1500      // Interquartile range
  },
  "interval_adaptivity": 0.48  // Coefficient of variation (higher = more adaptive)
}
```

### Comparison Metrics

Compares conformal interval to standard Monte Carlo approach:

```json
{
  "monte_carlo_interval": {
    "Revenue": {"lower": 49000, "upper": 55000, "confidence_level": 0.95}
  },
  "conformal_interval": {
    "Revenue": [48000, 56000]
  },
  "width_ratio": {"Revenue": 1.33},
  "interpretation": "Conformal interval is 33% wider..."
}
```

---

## Use Cases

### 1. Safety-Critical Decisions

**Scenario**: Medical device dosage recommendation

**Why Conformal**: Need provable bounds on outcomes, not just "~95% coverage"

```json
{
  "intervention": {"Dosage": 5.0},
  "calibration_data": [/* historical patient data */],
  "confidence_level": 0.99
}
```

**Outcome**: "Provably ≥99% confidence that outcome is within bounds"

### 2. Regulatory Compliance

**Scenario**: Financial model validation for stress testing

**Why Conformal**: Regulators require formal uncertainty guarantees

**Benefit**: Can document finite-sample validity without distributional assumptions

### 3. Model Calibration Assessment

**Scenario**: Validating a causal model's uncertainty quantification

**Why Conformal**: Compare conformal (honest) vs Monte Carlo (potentially overconfident)

**Insight**: If conformal intervals are much wider (>50%), Monte Carlo underestimates uncertainty

### 4. Small Sample Situations

**Scenario**: New product launch with limited historical data (n=20)

**Why Conformal**: Finite-sample guarantee works even with small calibration sets

**Traditional Methods**: Asymptotic guarantees unreliable with small samples

---

## Calibration Data Requirements

### Minimum Size

- **Absolute Minimum**: 10 observations
- **Recommended**: ≥50 for stable intervals
- **Ideal**: ≥100 for narrow intervals and good adaptivity

### Quality Considerations

1. **Exchangeability**
   - Calibration and test points should come from same distribution
   - Violated if: distribution shifts, non-stationarity, selection bias

2. **Representativeness**
   - Calibration data should cover the input space well
   - Sparse coverage → wider intervals

3. **Outcome Measurement**
   - Actual observed outcomes required (not predictions)
   - Must be for same outcome variable as prediction target

### Example: Good vs Poor Calibration Data

**Good**:
```json
[
  {"inputs": {"Price": 40}, "outcome": {"Revenue": 29800}},  // Actual observation
  {"inputs": {"Price": 45}, "outcome": {"Revenue": 32100}},  // Different price point
  {"inputs": {"Price": 50}, "outcome": {"Revenue": 35300}},  // Covers range
  // ... 7 more diverse observations
]
```

**Poor**:
```json
[
  {"inputs": {"Price": 50}, "outcome": {"Revenue": 35000}},  // All same inputs
  {"inputs": {"Price": 50}, "outcome": {"Revenue": 35100}},
  // ... all Price=50 (not representative)
]
```

---

## Conformal Methods

### Split Conformal (Default)

**Algorithm**: Split calibration data 50/50, use half for conformity scores

**Pros**:
- Simple and fast
- Easy to understand
- Good finite-sample guarantees

**Cons**:
- Wastes half the calibration data
- Can be unstable with small samples

**Best For**: n ≥ 20 calibration points

### CV+ Conformal (Future)

**Algorithm**: K-fold cross-validation for conformity scores

**Pros**:
- Uses all calibration data efficiently
- More stable than split conformal
- Better for small samples

**Cons**:
- Computationally expensive (K times slower)
- More complex to implement

**Best For**: 10 ≤ n < 50 calibration points

### Jackknife+ Conformal (Future)

**Algorithm**: Leave-one-out for each calibration point

**Pros**:
- Most stable
- Smallest intervals for given coverage
- Uses all data optimally

**Cons**:
- Very expensive (n times slower than split)
- Infeasible for large calibration sets

**Best For**: n < 20 (when every data point matters)

---

## Assumptions

### Exchangeability (Required)

**Definition**: Calibration points and test point are interchangeable

**Means**: They come from the same distribution

**Violated When**:
- Distribution shift between calibration and test
- Non-stationarity over time
- Selection bias

**Testing**:
- Plot calibration residuals vs. time → should be stable
- Compare calibration vs. test input distributions → should be similar

**Example Violation**:
```
Calibration: Historical data from 2020
Test: Prediction for 2024
Problem: Market conditions changed (distribution shift)
```

### No Other Assumptions!

Conformal prediction does **not** assume:
- Normality
- Homoscedasticity
- Specific distributional family
- Asymptotic regime
- Independent observations (exchangeability is weaker)

---

## Interpretation Guidelines

### Width Ratio Interpretation

| Width Ratio | Interpretation | Action |
|-------------|----------------|--------|
| 0.8 - 1.2 | Conformal and MC similar | MC uncertainty is reasonable |
| 1.2 - 1.5 | Conformal moderately wider | MC slightly overconfident |
| 1.5 - 2.0 | Conformal much wider | MC significantly underestimates uncertainty |
| > 2.0 | Conformal very much wider | MC unreliable, use conformal |

### Adaptivity Interpretation

| Adaptivity | Meaning | Implication |
|------------|---------|-------------|
| < 0.2 | Low variability in residuals | Intervals similar width everywhere |
| 0.2 - 0.5 | Moderate variability | Some local adaptation |
| > 0.5 | High variability | Strong local adaptation (good!) |

High adaptivity means conformal intervals are wider in high-uncertainty regions and narrower in low-uncertainty regions - exactly what you want.

---

## Performance Characteristics

### Computational Complexity

- **Time**: O(n log n) for sorting conformity scores
- **Space**: O(n) for storing calibration data
- **Typical**: <100ms for n=100 calibration points

### Scalability

| Calibration Size | Expected Time | Interval Quality |
|------------------|---------------|------------------|
| 10-20 | <50ms | Basic coverage |
| 20-50 | <100ms | Good coverage, moderate width |
| 50-100 | <200ms | Excellent coverage, narrow width |
| 100-500 | <500ms | Optimal coverage, very narrow |
| 500+ | 500ms-2s | Diminishing returns |

### When to Use vs Standard Counterfactual

**Use Conformal When**:
- Need provable coverage guarantees
- Have historical calibration data (≥10 points)
- High-stakes decisions requiring rigorous uncertainty
- Suspect Monte Carlo underestimates uncertainty
- Regulatory/scientific rigor required

**Use Standard When**:
- No calibration data available
- Quick exploratory analysis
- Low-stakes decisions
- Computational constraints
- Prediction focus (not uncertainty quantification)

---

## Examples

### Example 1: Basic Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/causal/counterfactual/conformal",
    json={
        "model": {
            "variables": ["Price", "Revenue"],
            "equations": {"Revenue": "10000 + 500*Price"},
            "distributions": {
                "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
            },
        },
        "intervention": {"Price": 50},
        "calibration_data": [
            {"inputs": {"Price": 40}, "outcome": {"Revenue": 30000}},
            {"inputs": {"Price": 45}, "outcome": {"Revenue": 32500}},
            # ... (at least 8 more points)
        ],
        "confidence_level": 0.95,
        "seed": 42,
    }
)

result = response.json()
print(f"Guaranteed coverage: {result['coverage_guarantee']['guaranteed_coverage']:.1%}")
print(f"Interval: [{result['prediction_interval']['lower_bound']['Revenue']:.0f}, "
      f"{result['prediction_interval']['upper_bound']['Revenue']:.0f}]")
```

### Example 2: Comparing Confidence Levels

```python
for confidence in [0.90, 0.95, 0.99]:
    response = requests.post(url, json={
        "model": model,
        "intervention": intervention,
        "calibration_data": calib_data,
        "confidence_level": confidence,
        "seed": 42,
    })

    result = response.json()
    width = result['prediction_interval']['interval_width']['Revenue']
    print(f"{confidence:.0%} confidence: width = {width:.0f}")

# Output:
# 90% confidence: width = 6500
# 95% confidence: width = 8000
# 99% confidence: width = 11000
```

### Example 3: Model Calibration Check

```python
response = requests.post(url, json=request_data)
result = response.json()

width_ratio = result['comparison_to_standard']['width_ratio']['Revenue']

if width_ratio > 1.5:
    print("⚠️ Model uncertainty is underestimated!")
    print(f"Monte Carlo intervals are {(width_ratio - 1) * 100:.0f}% too narrow")
    print("Recommendation: Use conformal intervals for decisions")
else:
    print("✓ Model uncertainty calibration is reasonable")
```

---

## Troubleshooting

### Problem: "Insufficient calibration data" Error

**Cause**: Fewer than 10 calibration points

**Solution**: Collect more historical observations, or use standard counterfactual analysis

### Problem: Very Wide Intervals

**Possible Causes**:
1. Small calibration set (n < 30)
2. High residual variability
3. Poor model fit

**Diagnostics**:
```python
result = response.json()
calib = result['calibration_quality']

print(f"Calibration size: {calib['calibration_size']}")
print(f"Residual std: {calib['residual_statistics']['std']:.0f}")
print(f"Adaptivity: {calib['interval_adaptivity']:.2f}")
```

**Solutions**:
- Collect more calibration data
- Improve model fit (reduce residuals)
- Use lower confidence level if acceptable

### Problem: Coverage Seems Too Conservative

**Observation**: Guaranteed coverage much below nominal (e.g., 90% vs 95%)

**Cause**: Finite-sample correction with small calibration set

**Solution**: This is correct behavior! Conformal prediction is honest about finite-sample limitations. To get closer to nominal:
- Increase calibration set size
- Accept that small samples can't provide 95% coverage guarantees

---

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). "Distribution-Free Predictive Inference for Regression". *Journal of the American Statistical Association*.
- Angelopoulos, A. N., & Bates, S. (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification". arXiv:2107.07511.

---

## Related Features

- [Standard Counterfactual Analysis](./counterfactuals.md) - For cases without calibration data
- [Batch Counterfactuals](./batch-counterfactuals.md) - Can be combined with conformal prediction
- [Sensitivity Analysis](./sensitivity-analysis.md) - Complementary uncertainty quantification
