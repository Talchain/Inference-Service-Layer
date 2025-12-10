# Multi-Criteria Aggregation Methods

Guide to selecting and using aggregation methods in ISL.

## Table of Contents

- [Overview](#overview)
- [Method Comparison](#method-comparison)
- [Weighted Sum](#weighted-sum)
- [Weighted Product](#weighted-product)
- [Lexicographic](#lexicographic)
- [Decision Tree](#decision-tree)
- [Examples](#examples)

---

## Overview

ISL supports three multi-criteria aggregation methods for combining scores across different criteria:

1. **Weighted Sum**: Linear combination of normalized scores
2. **Weighted Product**: Geometric aggregation (multiplicative)
3. **Lexicographic**: Strict priority ordering

Each method has different mathematical properties and is suited for different decision contexts.

---

## Method Comparison

| Property | Weighted Sum | Weighted Product | Lexicographic |
|----------|--------------|------------------|---------------|
| **Compensatory** | ✅ Yes | ✅ Yes | ❌ No |
| **Allows trade-offs** | ✅ Yes | ✅ Partial | ❌ No |
| **Zero tolerance** | ✅ Allows zeros | ❌ Penalizes zeros heavily | ✅ Allows zeros |
| **Scale sensitivity** | Low | High | None |
| **Computational complexity** | O(n) | O(n) | O(n log n) |
| **Interpretability** | High | Medium | High |
| **Common use** | Most decisions | Balanced requirements | Safety-critical |

---

## Weighted Sum

### Formula

```
Score(option) = Σ (weight_i × normalized_score_i)
```

Where:
- Weights sum to 1.0
- Scores normalized to 0-1 per criterion

### Characteristics

**✅ Use when:**
- Criteria are compensatory (strength on one can offset weakness on another)
- All criteria important but not strict requirements
- Need intuitive, easy-to-explain results
- Making most business decisions

**❌ Avoid when:**
- Any criterion is a hard constraint
- Cannot tolerate poor performance on any criterion

### Properties

1. **Linear aggregation**: Doubling a score doubles its contribution
2. **Fully compensatory**: High revenue can fully compensate for high cost
3. **Symmetric**: Swapping two options' scores swaps their rankings
4. **Scale-independent**: Normalization ensures fair comparison

### Example

Hiring decision with criteria: [skills: 0.5, culture_fit: 0.3, cost: 0.2]

| Candidate | Skills | Culture | Cost | **Weighted Sum** | Rank |
|-----------|--------|---------|------|------------------|------|
| Alice     | 1.0    | 0.6     | 0.2  | 0.5×1.0 + 0.3×0.6 + 0.2×0.2 = **0.72** | 1 |
| Bob       | 0.8    | 0.9     | 0.8  | 0.5×0.8 + 0.3×0.9 + 0.2×0.8 = **0.83** | **1** |
| Charlie   | 0.6    | 1.0     | 1.0  | 0.5×0.6 + 0.3×1.0 + 0.2×1.0 = **0.80** | 2 |

**Result**: Bob wins despite lower skills due to strong culture fit and low cost.

### When to Choose

```
if (criteria are compensatory AND
    no hard constraints AND
    need interpretable results):
    use weighted_sum
```

---

## Weighted Product

### Formula

```
Score(option) = ∏ (normalized_score_i ^ weight_i)
```

Where:
- Weights sum to 1.0
- Scores normalized to 0-1 per criterion

### Characteristics

**✅ Use when:**
- All criteria must be reasonably good (no zeros tolerated)
- Need balanced performance across criteria
- Prefer geometric mean over arithmetic mean
- Avoiding dominance by single criterion

**❌ Avoid when:**
- Criteria can validly be zero
- Very different scales (causes numerical instability)
- Need simple explanation

### Properties

1. **Multiplicative**: Zero on any criterion → zero overall score
2. **Partially compensatory**: High score on one helps, but can't fully offset low score on another
3. **Diminishing returns**: Improving from 0.9 to 1.0 less valuable than 0.1 to 0.2
4. **Shape of trade-off**: Encourages balanced performance

### Example

Product quality metrics: [reliability: 0.4, usability: 0.3, performance: 0.3]

| Product | Reliability | Usability | Performance | **Weighted Product** | Rank |
|---------|-------------|-----------|-------------|----------------------|------|
| A       | 1.0         | 0.5       | 0.5         | 1.0^0.4 × 0.5^0.3 × 0.5^0.3 = **0.63** | 2 |
| B       | 0.8         | 0.8       | 0.8         | 0.8^0.4 × 0.8^0.3 × 0.8^0.3 = **0.80** | **1** |
| C       | 1.0         | 0.1       | 1.0         | 1.0^0.4 × 0.1^0.3 × 1.0^0.3 = **0.29** | 3 |

**Result**: B wins with balanced scores. C penalized heavily for poor usability (0.1^0.3 = 0.46).

**Comparison with Weighted Sum**:
- Weighted sum: A would score 0.75, C would score 0.73 (closer race)
- Weighted product: C penalized more severely (0.29 vs 0.73)

### When to Choose

```
if (all criteria must be decent AND
    prefer balanced performance AND
    zero scores are invalid):
    use weighted_product
```

---

## Lexicographic

### Formula

```
Sort by criterion_1 (highest weight)
If tied, sort by criterion_2 (next highest weight)
If tied, sort by criterion_3
...
```

### Characteristics

**✅ Use when:**
- Clear hierarchy of importance
- First criterion is non-negotiable
- Strict priority ordering required
- Safety or compliance decisions

**❌ Avoid when:**
- Criteria have similar importance
- Need compensatory trade-offs
- Small differences in top criterion matter less than large differences in lower criteria

### Properties

1. **Non-compensatory**: No amount of criterion_2 can overcome deficit in criterion_1
2. **Discontinuous**: Tiny improvement in criterion_1 outweighs huge improvement in criterion_2
3. **Strict hierarchy**: Weights define pure priority order
4. **Ties handled by next criterion**: Natural tiebreaker mechanism

### Example

Safety compliance decision: [regulatory_compliance: 0.6, cost: 0.3, speed: 0.1]

| Option | Compliance | Cost | Speed | **Lexicographic** | Rank |
|--------|------------|------|-------|-------------------|------|
| A      | 1.0        | 0.2  | 0.1   | (1.0, 0.2, 0.1)   | **1** |
| B      | 0.9        | 1.0  | 1.0   | (0.9, 1.0, 1.0)   | 2 |
| C      | 1.0        | 0.8  | 0.5   | (1.0, 0.8, 0.5)   | 1 |

**Result**:
1. A and C tie on compliance (1.0)
2. Tiebreaker: C has higher cost score (0.8 vs 0.2)
3. **Winner: C** (despite B being perfect on cost and speed, it fails on compliance)

**Comparison with Weighted Sum**:
- Weighted sum: B would score 0.94, possibly winning despite compliance deficit
- Lexicographic: B cannot win without perfect compliance

### When to Choose

```
if (criterion_1 is non-negotiable OR
    strict priority hierarchy exists OR
    safety/compliance decision):
    use lexicographic
```

---

## Decision Tree

### Step 1: Is there a strict hierarchy?

```
YES → Use lexicographic
NO  → Continue to Step 2
```

**Examples of strict hierarchy:**
- Safety > Cost > Speed
- Compliance > Budget > Timeline
- Must-have feature > Nice-to-have features

---

### Step 2: Can criteria be zero?

```
YES → Use weighted_sum
NO  → Continue to Step 3
```

**Examples where zero is valid:**
- Cost can be zero (free option)
- Risk can be zero (no risk)
- Improvement can be zero (no change)

---

### Step 3: Need balanced performance?

```
YES → Use weighted_product
NO  → Use weighted_sum
```

**Examples needing balance:**
- Product quality (all aspects must be decent)
- Team member evaluation (no major weaknesses)
- Multi-objective optimization (Pareto-style thinking)

---

### Decision Tree Diagram

```
                    Start
                      |
                      v
            Is there strict hierarchy?
                   /     \
                 YES      NO
                  |        |
           Lexicographic   |
                           v
                Can criteria be zero?
                       /     \
                     YES      NO
                      |        |
               Weighted Sum    |
                               v
                Need balanced performance?
                           /     \
                         YES      NO
                          |        |
                Weighted Product  Weighted Sum
```

---

## Examples

### Example 1: Product Launch Decision

**Criteria:** [revenue_potential: 0.5, development_cost: 0.3, time_to_market: 0.2]

**Considerations:**
- All criteria important but compensatory
- High revenue can justify high cost
- No hard constraints
- Need clear explanation for stakeholders

**→ Use weighted_sum** (most common for business decisions)

---

### Example 2: Supplier Selection

**Criteria:** [quality: 0.4, reliability: 0.3, price: 0.2, sustainability: 0.1]

**Considerations:**
- All criteria must be reasonably good
- Cannot tolerate very poor quality or reliability
- Prefer balanced suppliers
- Zero quality/reliability is unacceptable

**→ Use weighted_product** (ensures minimum standards on all criteria)

---

### Example 3: Medical Treatment Choice

**Criteria:** [safety: 0.7, efficacy: 0.2, cost: 0.1]

**Considerations:**
- Safety is absolutely primary
- No amount of efficacy or cost savings justifies unsafe treatment
- Strict regulatory requirements
- Non-compensatory

**→ Use lexicographic** (safety cannot be traded off)

---

### Example 4: Hiring Decision

**Criteria:** [skills: 0.5, culture_fit: 0.3, compensation: 0.2]

**Considerations:**
- High skills can compensate for higher compensation
- All criteria important but flexible
- Zero on any criterion is possible (entry-level = low skills, volunteer = zero comp)
- Need to explain to hiring committee

**→ Use weighted_sum** (compensatory, intuitive)

---

## Implementation Notes

### Normalization

All methods use 0-1 normalization per criterion:

```python
normalized_score = (score - min_score) / (max_score - min_score)

# For minimize criteria:
normalized_score = 1.0 - normalized_score
```

### Edge Cases

#### Weighted Sum
- **Uniform scores**: All options get 0.5 (neutral)
- **Missing scores**: Option gets 0.0 (worst)

#### Weighted Product
- **Zero scores**: Result is 0.0 (heavily penalized)
- **Near-zero scores**: Replaced with 0.001 to avoid numerical issues

#### Lexicographic
- **Ties**: Broken by next criterion
- **All tied**: Arbitrary order (stable sort)

### API Usage

```python
# Request
POST /api/v1/aggregation/multi-criteria
{
  "criterion_results": [...],
  "aggregation_method": "weighted_sum"  # or "weighted_product" or "lexicographic"
}

# Response
{
  "aggregated_rankings": [
    {"option_id": "A", "rank": 1, "aggregated_score": 0.85},
    ...
  ],
  "trade_offs": [...]
}
```

---

## Mathematical Properties

### Weighted Sum

- **Linearity**: f(ax + by) = af(x) + bf(y)
- **Range**: [0, 1]
- **Monotonicity**: Improving any criterion improves score
- **Pareto-compliance**: Pareto-superior options score higher

### Weighted Product

- **Homogeneity**: f(kx) = k^w × f(x) where w = sum of weights = 1
- **Range**: [0, 1]
- **Monotonicity**: Improving any criterion improves score
- **Pareto-compliance**: Pareto-superior options score higher
- **Concavity**: Encourages balanced scores (diminishing returns)

### Lexicographic

- **Non-compensatory**: Criterion i+1 only matters when tied on criterion i
- **Transitive**: If A > B and B > C, then A > C
- **Not continuous**: Small change in criterion 1 can dominate large change in criterion 2
- **Pareto-compliance**: May violate in edge cases (rare)

---

## Recommendations Summary

| Decision Context | Recommended Method | Why |
|------------------|-------------------|-----|
| Business strategy | Weighted Sum | Compensatory, intuitive |
| Product development | Weighted Sum | Balanced needs, trade-offs expected |
| Supplier selection | Weighted Product | All criteria must be decent |
| Quality assurance | Weighted Product | No tolerance for poor performance |
| Safety/compliance | Lexicographic | Non-negotiable priorities |
| Regulatory | Lexicographic | Strict hierarchy required |
| Hiring (senior) | Weighted Sum | Skills can offset cost |
| Hiring (must-haves) | Lexicographic | Required qualifications first |
| Investment choice | Weighted Sum | Risk-return trade-offs |
| Medical treatment | Lexicographic | Safety is paramount |

**Default recommendation**: Start with **weighted_sum** unless you have specific reason to use others.

---

## References

### Academic Background

- **Weighted Sum**: Multi-Attribute Utility Theory (MAUT)
- **Weighted Product**: Geometric Mean Method
- **Lexicographic**: Lexicographic Preference Ordering

### Further Reading

- Keeney, R. L., & Raiffa, H. (1993). *Decisions with Multiple Objectives*
- Saaty, T. L. (1980). *The Analytic Hierarchy Process*
- Fishburn, P. C. (1974). *Lexicographic Orders, Utilities and Decision Rules*

---

## Testing Your Choice

After selecting a method, validate with these tests:

### 1. Extreme Case Test

Create option with perfect score on criterion 1, zero on all others.
- **Weighted sum**: Should score = weight_1
- **Weighted product**: Should score = 0
- **Lexicographic**: Should rank first

### 2. Balance Test

Create two options:
- Option A: (1.0, 0.2, 0.2)
- Option B: (0.6, 0.6, 0.6)

Which should win depends on method and your preferences.

### 3. Sensitivity Test

Slightly change criterion 1 weight. Rankings should change smoothly (not discontinuously).

---

## Need Help?

If unsure which method to use:

1. **Start with weighted_sum** (works for 80% of cases)
2. **Run all three** and compare results
3. **Check which aligns with your intuition** about the decision
4. **Ask stakeholders** to validate the ranking makes sense

Remember: The "right" method is the one that produces rankings that align with your actual decision preferences.
