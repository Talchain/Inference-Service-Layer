# ISL Cross-Reference Schema for Assumption Traceability

## Overview

ISL provides stable, deterministic IDs for assumptions, uncertainty sources, and sensitivity drivers. This enables UI to create clickable navigation: **uncertainty → assumption → sensitivity**.

**User Journey:**
1. User sees outcome range (e.g., "£50k-£70k")
2. Clicks "Why so uncertain?" → Breakdown shows "58% from customer sensitivity"
3. Clicks "customer sensitivity" → Navigates to assumption "elasticity = 0.5 [weak evidence]"
4. Clicks "Show impact" → Navigates to sensitivity analysis showing tornado chart

---

## Stable ID Generation

All IDs are deterministic: **same input → same ID across requests**

### **Algorithm**

```python
def generate_stable_id(id_type: str, context: Dict) -> str:
    """
    Generate deterministic ID for cross-referencing.

    Args:
        id_type: "assumption", "uncertainty_source", "sensitivity_driver"
        context: Identifying information (name, type, etc.)

    Returns:
        Stable ID like "assumption_param_elasticity"
    """
    # Normalize context
    normalized = json.dumps(context, sort_keys=True, separators=(',', ':'))

    # Hash (first 8 chars for readability)
    hash_value = hashlib.sha256(normalized.encode()).hexdigest()[:8]

    # Construct ID
    if id_type == "assumption":
        return f"assumption_{context['parameter_name']}_{hash_value}"
    elif id_type == "uncertainty_source":
        return f"uncertainty_{context['source_type']}_{hash_value}"
    elif id_type == "sensitivity_driver":
        return f"sensitivity_{context['variable_name']}_{hash_value}"
```

**Properties:**
- Deterministic: Same parameter → same ID
- Readable: Includes human-readable prefix
- Unique: Hash prevents collisions
- Persistent: Stable across ISL versions

---

## Response Structure with Cross-References

### **Example: Counterfactual Analysis Response**

```json
{
  "outcome_range": {
    "lower": 50000,
    "upper": 70000,
    "confidence_level": 0.95
  },

  "uncertainty_breakdown": [
    {
      "id": "uncertainty_param_customer_sensitivity_a1b2c3d4",
      "source_type": "parameter_uncertainty",
      "contribution": 0.58,
      "description": "Uncertainty in customer price sensitivity",
      "linked_assumption": "assumption_param_elasticity_e5f6g7h8",
      "linked_sensitivity": "sensitivity_driver_elasticity_e5f6g7h8"
    },
    {
      "id": "uncertainty_structural_confounding_i9j0k1l2",
      "source_type": "structural_uncertainty",
      "contribution": 0.23,
      "description": "Potential confounding variable",
      "linked_assumption": "assumption_structure_no_confounder_m3n4o5p6",
      "linked_sensitivity": null
    }
  ],

  "assumptions": [
    {
      "id": "assumption_param_elasticity_e5f6g7h8",
      "type": "parameter_value",
      "parameter_name": "elasticity",
      "value": 0.5,
      "evidence_quality": "weak",
      "description": "Price elasticity of demand",
      "source": "User estimate (no data)",
      "linked_uncertainty": "uncertainty_param_customer_sensitivity_a1b2c3d4",
      "linked_sensitivity": "sensitivity_driver_elasticity_e5f6g7h8"
    },
    {
      "id": "assumption_structure_no_confounder_m3n4o5p6",
      "type": "structural",
      "description": "No confounding between price and demand",
      "evidence_quality": "moderate",
      "source": "Causal identification algorithm",
      "linked_uncertainty": "uncertainty_structural_confounding_i9j0k1l2",
      "linked_sensitivity": null
    }
  ],

  "sensitivity_analysis": {
    "drivers": [
      {
        "id": "sensitivity_driver_elasticity_e5f6g7h8",
        "variable_name": "elasticity",
        "contribution": 0.58,
        "range_if_changed": {
          "elasticity_0.3": [45000, 65000],
          "elasticity_0.7": [52000, 75000]
        },
        "linked_assumption": "assumption_param_elasticity_e5f6g7h8",
        "linked_uncertainty": "uncertainty_param_customer_sensitivity_a1b2c3d4"
      }
    ]
  }
}
```

---

## UI Navigation Flows

### **Flow 1: Uncertainty → Assumption**

**User Action:** Clicks "58% from customer sensitivity" in uncertainty breakdown

**UI Lookup:**
```javascript
// 1. Get uncertainty source
const uncertaintySource = response.uncertainty_breakdown.find(
  u => u.contribution === 0.58
);

// 2. Get linked assumption ID
const assumptionId = uncertaintySource.linked_assumption;
// "assumption_param_elasticity_e5f6g7h8"

// 3. Find assumption details
const assumption = response.assumptions.find(
  a => a.id === assumptionId
);

// 4. Navigate to assumption panel
showAssumptionDetails(assumption);
```

**UI Display:**
```
┌─────────────────────────────────────┐
│ Assumption: elasticity              │
├─────────────────────────────────────┤
│ Value: 0.5                          │
│ Evidence: Weak (user estimate)      │
│ Description: Price elasticity of    │
│              demand                 │
│                                     │
│ [Edit Value] [See Impact]           │
└─────────────────────────────────────┘
```

---

### **Flow 2: Assumption → Sensitivity**

**User Action:** Clicks "See Impact" on assumption panel

**UI Lookup:**
```javascript
// 1. Get assumption
const assumption = response.assumptions.find(
  a => a.id === "assumption_param_elasticity_e5f6g7h8"
);

// 2. Get linked sensitivity ID
const sensitivityId = assumption.linked_sensitivity;
// "sensitivity_driver_elasticity_e5f6g7h8"

// 3. Find sensitivity driver
const sensitivityDriver = response.sensitivity_analysis.drivers.find(
  d => d.id === sensitivityId
);

// 4. Navigate to sensitivity panel
showSensitivityAnalysis(sensitivityDriver);
```

**UI Display:**
```
┌─────────────────────────────────────┐
│ Sensitivity: elasticity             │
├─────────────────────────────────────┤
│ Impact: 58% of total uncertainty    │
│                                     │
│ If elasticity = 0.3: £45k-£65k     │
│ If elasticity = 0.5: £50k-£70k ⚫   │
│ If elasticity = 0.7: £52k-£75k     │
│                                     │
│ [Tornado Chart] [What-If Analyser] │
└─────────────────────────────────────┘
```

---

### **Flow 3: Sensitivity → Uncertainty**

**User Action:** Clicks "Why does this matter?" on sensitivity driver

**UI Lookup:**
```javascript
// 1. Get sensitivity driver
const sensitivityDriver = response.sensitivity_analysis.drivers.find(
  d => d.variable_name === "elasticity"
);

// 2. Get linked uncertainty ID
const uncertaintyId = sensitivityDriver.linked_uncertainty;
// "uncertainty_param_customer_sensitivity_a1b2c3d4"

// 3. Find uncertainty source
const uncertaintySource = response.uncertainty_breakdown.find(
  u => u.id === uncertaintyId
);

// 4. Navigate back to uncertainty breakdown
showUncertaintyBreakdown(uncertaintySource);
```

---

## ID Persistence Guarantees

### **Within Session (Same Request)**

**Guarantee:** Identical IDs across all sections of response

**Example:**
```json
{
  "assumptions": [
    {"id": "assumption_param_elasticity_e5f6g7h8", ...}
  ],
  "uncertainty_breakdown": [
    {"linked_assumption": "assumption_param_elasticity_e5f6g7h8", ...}
  ],
  "sensitivity_analysis": {
    "drivers": [
      {"linked_assumption": "assumption_param_elasticity_e5f6g7h8", ...}
    ]
  }
}
```

---

### **Across Requests (Same Model)**

**Guarantee:** Same parameter → same ID, even in different requests

**Request 1:**
```json
{
  "model": {"parameters": {"elasticity": 0.5}},
  "response": {
    "assumptions": [
      {"id": "assumption_param_elasticity_e5f6g7h8", "value": 0.5}
    ]
  }
}
```

**Request 2 (hours later, same model):**
```json
{
  "model": {"parameters": {"elasticity": 0.5}},
  "response": {
    "assumptions": [
      {"id": "assumption_param_elasticity_e5f6g7h8", "value": 0.5}
    ]
  }
}
```

**Use Case:** UI can bookmark/link to specific assumptions

---

### **Across ISL Versions**

**Guarantee:** IDs stable across ISL deployments (same hashing algorithm)

**Exception:** If ISL changes ID generation algorithm, IDs may change (documented in release notes)

---

## UI Implementation Guide

### **Step 1: Parse Response**

```typescript
interface ISLResponse {
  outcome_range: OutcomeRange;
  uncertainty_breakdown: UncertaintySource[];
  assumptions: Assumption[];
  sensitivity_analysis: SensitivityAnalysis;
}

interface UncertaintySource {
  id: string;
  source_type: string;
  contribution: number;
  description: string;
  linked_assumption: string | null;
  linked_sensitivity: string | null;
}

interface Assumption {
  id: string;
  type: string;
  parameter_name?: string;
  value?: number;
  evidence_quality: string;
  description: string;
  source: string;
  linked_uncertainty: string | null;
  linked_sensitivity: string | null;
}

interface SensitivityDriver {
  id: string;
  variable_name: string;
  contribution: number;
  range_if_changed: Record<string, [number, number]>;
  linked_assumption: string | null;
  linked_uncertainty: string | null;
}
```

---

### **Step 2: Build Cross-Reference Index**

```typescript
class CrossReferenceIndex {
  private uncertaintyById = new Map<string, UncertaintySource>();
  private assumptionById = new Map<string, Assumption>();
  private sensitivityById = new Map<string, SensitivityDriver>();

  constructor(response: ISLResponse) {
    // Index all entities by ID
    response.uncertainty_breakdown.forEach(u => {
      this.uncertaintyById.set(u.id, u);
    });

    response.assumptions.forEach(a => {
      this.assumptionById.set(a.id, a);
    });

    response.sensitivity_analysis.drivers.forEach(d => {
      this.sensitivityById.set(d.id, d);
    });
  }

  getAssumptionForUncertainty(uncertaintyId: string): Assumption | null {
    const uncertainty = this.uncertaintyById.get(uncertaintyId);
    if (!uncertainty?.linked_assumption) return null;
    return this.assumptionById.get(uncertainty.linked_assumption) || null;
  }

  getSensitivityForAssumption(assumptionId: string): SensitivityDriver | null {
    const assumption = this.assumptionById.get(assumptionId);
    if (!assumption?.linked_sensitivity) return null;
    return this.sensitivityById.get(assumption.linked_sensitivity) || null;
  }

  getUncertaintyForSensitivity(sensitivityId: string): UncertaintySource | null {
    const sensitivity = this.sensitivityById.get(sensitivityId);
    if (!sensitivity?.linked_uncertainty) return null;
    return this.uncertaintyById.get(sensitivity.linked_uncertainty) || null;
  }
}
```

---

### **Step 3: Implement Navigation**

```typescript
class AssumptionTraceabilityPanel {
  private index: CrossReferenceIndex;

  onUncertaintyClick(uncertaintyId: string) {
    const assumption = this.index.getAssumptionForUncertainty(uncertaintyId);
    if (assumption) {
      this.navigateToAssumption(assumption);
    }
  }

  onShowImpactClick(assumptionId: string) {
    const sensitivity = this.index.getSensitivityForAssumption(assumptionId);
    if (sensitivity) {
      this.navigateToSensitivity(sensitivity);
    }
  }

  onWhyMatterClick(sensitivityId: string) {
    const uncertainty = this.index.getUncertaintyForSensitivity(sensitivityId);
    if (uncertainty) {
      this.navigateToUncertainty(uncertainty);
    }
  }
}
```

---

### **Step 4: Handle Missing Links**

**Some links may be `null`:**

- **Uncertainty without assumption:** Structural uncertainty (e.g., model misspecification) may not link to specific parameter
- **Assumption without sensitivity:** Some assumptions (e.g., "no confounders") don't have sensitivity analysis
- **Sensitivity without uncertainty:** Edge case (shouldn't happen, but handle gracefully)

**UI Behaviour:**

```typescript
onUncertaintyClick(uncertaintyId: string) {
  const assumption = this.index.getAssumptionForUncertainty(uncertaintyId);

  if (assumption) {
    this.navigateToAssumption(assumption);
  } else {
    this.showMessage("This uncertainty source doesn't link to a specific assumption");
  }
}
```

---

## Testing Cross-Reference Links

### **Test 1: Round-Trip Navigation**

```typescript
test("uncertainty → assumption → sensitivity → uncertainty", () => {
  const response = getMockISLResponse();
  const index = new CrossReferenceIndex(response);

  // Start with uncertainty
  const uncertainty = response.uncertainty_breakdown[0];

  // Navigate to assumption
  const assumption = index.getAssumptionForUncertainty(uncertainty.id);
  expect(assumption).not.toBeNull();
  expect(assumption!.linked_uncertainty).toBe(uncertainty.id);

  // Navigate to sensitivity
  const sensitivity = index.getSensitivityForAssumption(assumption!.id);
  expect(sensitivity).not.toBeNull();
  expect(sensitivity!.linked_assumption).toBe(assumption!.id);

  // Navigate back to uncertainty
  const backToUncertainty = index.getUncertaintyForSensitivity(sensitivity!.id);
  expect(backToUncertainty).not.toBeNull();
  expect(backToUncertainty!.id).toBe(uncertainty.id);
});
```

---

### **Test 2: ID Persistence Across Requests**

```typescript
test("same model → same IDs", async () => {
  const model = {
    parameters: {
      elasticity: 0.5,
      market_size: 1000000
    }
  };

  // First request
  const response1 = await islClient.post("/api/v1/causal/counterfactual", {
    json: { model }
  });

  // Second request (same model)
  const response2 = await islClient.post("/api/v1/causal/counterfactual", {
    json: { model }
  });

  // IDs should be identical
  const assumption1 = response1.assumptions.find(a => a.parameter_name === "elasticity");
  const assumption2 = response2.assumptions.find(a => a.parameter_name === "elasticity");

  expect(assumption1.id).toBe(assumption2.id);
});
```

---

## Error Handling

### **Invalid ID References**

**Scenario:** ISL returns ID that doesn't exist in response

**Cause:** ISL bug (should never happen)

**UI Handling:**
```typescript
const assumption = this.index.getAssumptionForUncertainty(uncertaintyId);

if (!assumption) {
  logger.error("Invalid cross-reference", { uncertaintyId });
  this.showError("Unable to find linked assumption. Please refresh.");
  return;
}
```

---

### **Circular References**

**Scenario:** A → B → C → A

**Cause:** ISL bug (should never happen)

**Prevention:**
```typescript
class CrossReferenceIndex {
  private visitedIds = new Set<string>();

  getAssumptionForUncertainty(uncertaintyId: string): Assumption | null {
    if (this.visitedIds.has(uncertaintyId)) {
      throw new Error("Circular reference detected");
    }

    this.visitedIds.add(uncertaintyId);
    // ... rest of logic
  }
}
```

---

## Appendix: Example Full Response

```json
{
  "outcome_range": {
    "lower": 50000,
    "upper": 70000,
    "p10": 52000,
    "p50": 60000,
    "p90": 68000,
    "confidence_level": 0.95
  },

  "uncertainty_breakdown": [
    {
      "id": "uncertainty_param_customer_sensitivity_a1b2c3d4",
      "source_type": "parameter_uncertainty",
      "contribution": 0.58,
      "description": "Uncertainty in customer price sensitivity",
      "linked_assumption": "assumption_param_elasticity_e5f6g7h8",
      "linked_sensitivity": "sensitivity_driver_elasticity_e5f6g7h8"
    },
    {
      "id": "uncertainty_structural_confounding_i9j0k1l2",
      "source_type": "structural_uncertainty",
      "contribution": 0.23,
      "description": "Potential confounding variable",
      "linked_assumption": "assumption_structure_no_confounder_m3n4o5p6",
      "linked_sensitivity": null
    },
    {
      "id": "uncertainty_model_specification_q7r8s9t0",
      "source_type": "model_uncertainty",
      "contribution": 0.19,
      "description": "Functional form assumptions",
      "linked_assumption": "assumption_model_linear_u1v2w3x4",
      "linked_sensitivity": "sensitivity_driver_functional_form_u1v2w3x4"
    }
  ],

  "assumptions": [
    {
      "id": "assumption_param_elasticity_e5f6g7h8",
      "type": "parameter_value",
      "parameter_name": "elasticity",
      "value": 0.5,
      "evidence_quality": "weak",
      "description": "Price elasticity of demand",
      "source": "User estimate (no data)",
      "linked_uncertainty": "uncertainty_param_customer_sensitivity_a1b2c3d4",
      "linked_sensitivity": "sensitivity_driver_elasticity_e5f6g7h8"
    },
    {
      "id": "assumption_structure_no_confounder_m3n4o5p6",
      "type": "structural",
      "description": "No confounding between price and demand",
      "evidence_quality": "moderate",
      "source": "Causal identification algorithm (Y₀)",
      "linked_uncertainty": "uncertainty_structural_confounding_i9j0k1l2",
      "linked_sensitivity": null
    },
    {
      "id": "assumption_model_linear_u1v2w3x4",
      "type": "functional_form",
      "description": "Linear relationship between price and demand",
      "evidence_quality": "weak",
      "source": "User specification",
      "linked_uncertainty": "uncertainty_model_specification_q7r8s9t0",
      "linked_sensitivity": "sensitivity_driver_functional_form_u1v2w3x4"
    }
  ],

  "sensitivity_analysis": {
    "drivers": [
      {
        "id": "sensitivity_driver_elasticity_e5f6g7h8",
        "variable_name": "elasticity",
        "contribution": 0.58,
        "range_if_changed": {
          "elasticity_0.3": [45000, 65000],
          "elasticity_0.5": [50000, 70000],
          "elasticity_0.7": [52000, 75000]
        },
        "linked_assumption": "assumption_param_elasticity_e5f6g7h8",
        "linked_uncertainty": "uncertainty_param_customer_sensitivity_a1b2c3d4"
      },
      {
        "id": "sensitivity_driver_functional_form_u1v2w3x4",
        "variable_name": "functional_form",
        "contribution": 0.19,
        "range_if_changed": {
          "linear": [50000, 70000],
          "log_linear": [48000, 72000],
          "quadratic": [46000, 74000]
        },
        "linked_assumption": "assumption_model_linear_u1v2w3x4",
        "linked_uncertainty": "uncertainty_model_specification_q7r8s9t0"
      }
    ]
  },

  "_metadata": {
    "isl_version": "1.0.0",
    "config_fingerprint": "a1b2c3d4e5f6",
    "request_id": "req_y7z8a9b0c1d2",
    "computed_at": "2025-11-20T10:30:00Z"
  }
}
```

---

**For questions or clarifications, contact: #isl-ui-integration**
