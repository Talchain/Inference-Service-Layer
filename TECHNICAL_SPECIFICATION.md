# ISL Technical Specification
## Inference Service Layer - Complete Architecture & Integration Guide

**Version:** 2.0
**Last Updated:** 2025-11-24
**Status:** Production Ready

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Architecture](#2-core-architecture)
3. [Research Features (5-7)](#3-research-features-5-7)
4. [API Endpoints](#4-api-endpoints)
5. [Integration Points](#5-integration-points)
6. [Data Models](#6-data-models)
7. [Performance & Scalability](#7-performance--scalability)
8. [Security](#8-security)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)

---

## 1. System Overview

### 1.1 Purpose

The **Inference Service Layer (ISL)** is a production-grade FastAPI service that provides causal inference capabilities for three workstreams:

- **PLoT Engine**: Causal validation, counterfactual generation, sensitivity analysis
- **TAE**: Robust assumption validation, explanation critique
- **CEE**: Contrastive explanations, progressive disclosure, causal discovery

### 1.2 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Runtime** | Python | 3.11+ |
| **Framework** | FastAPI | 0.104+ |
| **Validation** | Pydantic | 2.5+ |
| **Cache** | Redis | 7.2+ |
| **Monitoring** | Prometheus + Grafana | 2.48 / 10.2 |
| **ML Libraries** | sentence-transformers, scikit-learn | Latest |
| **Testing** | pytest, pytest-asyncio | Latest |

### 1.3 System Capabilities

- **Causal Validation**: Assumption checking (unconfoundedness, positivity, consistency)
- **Counterfactual Generation**: What-if scenario generation with causal constraints
- **Sensitivity Analysis**: Quantitative elasticity & robustness scoring
- **Progressive Explanations**: 3-level explanations (simple/intermediate/technical)
- **Causal Discovery**: Factor extraction from text, DAG suggestion
- **Performance**: P50=8.3ms, P95=13.0ms, 120 req/s throughput

---

## 2. Core Architecture

### 2.1 Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer (AWS ALB)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Application (ISL)                   │
├─────────────────────────────────────────────────────────────┤
│  Middleware Layer:                                           │
│  - Rate Limiting (100 req/min per IP)                       │
│  - CORS (whitelisted origins)                               │
│  - Authentication (API key)                                  │
│  - Request ID tracking                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌──────────────────┐   ┌──────────────────┐
        │   Redis Cache    │   │   Prometheus     │
        │   (TTL: 5min)    │   │   (Metrics)      │
        └──────────────────┘   └──────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │     Grafana      │
                              │   (Dashboards)   │
                              └──────────────────┘
```

### 2.2 Module Structure

```
src/
├── api/                    # API endpoints
│   ├── main.py            # FastAPI app, middleware, CORS
│   ├── validation.py      # Causal validation endpoints
│   ├── counterfactual.py  # Counterfactual generation endpoints
│   ├── sensitivity.py     # Sensitivity analysis endpoints (NEW)
│   ├── explanations.py    # Progressive explanations endpoints (NEW)
│   └── discovery.py       # Causal discovery endpoints (NEW)
│
├── models/                 # Pydantic data models
│   ├── causal_model.py    # Core causal model definitions
│   ├── validation.py      # Validation request/response models
│   ├── sensitivity.py     # Sensitivity analysis models (NEW)
│   ├── explanation_quality.py  # Explanation models (NEW)
│   └── causal_representation.py  # Discovery models (NEW)
│
├── services/              # Business logic
│   ├── causal_validator.py         # Assumption validation
│   ├── counterfactual_generator.py # Counterfactual generation
│   ├── sensitivity_analyzer.py     # Sensitivity analysis (NEW)
│   ├── explanation_generator.py    # Progressive disclosure (NEW)
│   └── causal_representation_learner.py  # Factor extraction (NEW)
│
├── middleware/            # Request/response middleware
│   ├── rate_limiting.py   # Rate limiting + proxy header handling
│   ├── authentication.py  # API key validation
│   └── error_handling.py  # Global error handling
│
└── utils/                 # Utilities
    ├── cache.py           # Redis caching
    ├── metrics.py         # Prometheus metrics
    └── logging.py         # Structured logging
```

### 2.3 Request Flow

```
1. Client Request
   ↓
2. Load Balancer (X-Forwarded-For header added)
   ↓
3. Rate Limiting Middleware (checks X-Forwarded-For)
   ↓
4. Authentication Middleware (validates API key)
   ↓
5. CORS Middleware (checks origin whitelist)
   ↓
6. Cache Check (Redis, TTL=5min)
   ├─ HIT → Return cached response
   └─ MISS → Continue
      ↓
7. API Endpoint Handler
   ↓
8. Service Layer (business logic)
   ↓
9. Response Generation
   ↓
10. Cache Store (Redis)
    ↓
11. Metrics Export (Prometheus)
    ↓
12. Return Response
```

---

## 3. Research Features (5-7)

### 3.1 Feature 5: Enhanced Sensitivity Analysis

**Purpose:** Quantitative assessment of how assumption violations affect causal estimates.

#### Architecture

```
SensitivityAnalyzer (src/services/sensitivity_analyzer.py)
│
├── analyze_sensitivity()
│   ├── 1. Validate causal model (identifiability check)
│   ├── 2. Extract assumptions (unconfoundedness, positivity, consistency)
│   ├── 3. For each assumption:
│   │      ├── Generate violation scenarios (mild: 10%, moderate: 25%, severe: 50%)
│   │      ├── Re-estimate outcome for each scenario
│   │      ├── Calculate elasticity: (Δoutcome / outcome) / (Δviolation / violation)
│   │      ├── Calculate robustness score: 1 / (1 + max_deviation)
│   │      └── Flag as critical if max_deviation > 20%
│   └── 4. Return aggregated metrics
│
├── calculate_elasticity()
│   └── Pure elasticity calculation (no scenario generation)
│
└── get_critical_assumptions()
    └── Filter assumptions with critical=True
```

#### Key Algorithms

**Elasticity Calculation:**
```python
elasticity = (
    (outcome_violated - outcome_baseline) / outcome_baseline
) / (
    violation_magnitude / 100.0
)
```

**Robustness Score:**
```python
robustness_score = 1.0 / (1.0 + max_deviation_percent / 100.0)
# Range: [0, 1] where 0=fragile, 1=robust
```

#### API Contract

**Endpoint:** `POST /api/v1/sensitivity/analyze`

**Request:**
```json
{
  "model": {
    "nodes": ["X", "Y", "Z"],
    "edges": [["X", "Z"], ["Y", "Z"]],
    "treatment": "X",
    "outcome": "Z"
  },
  "data": [
    {"X": 1.0, "Y": 0.5, "Z": 2.0},
    {"X": 0.0, "Y": 1.0, "Z": 1.5}
  ],
  "assumptions": ["unconfoundedness", "positivity", "consistency"]
}
```

**Response:**
```json
{
  "metrics": [
    {
      "assumption": "unconfoundedness",
      "baseline_outcome": 2.0,
      "outcome_range": [1.75, 2.25],
      "elasticity": 0.15,
      "critical": false,
      "max_deviation_percent": 12.5,
      "robustness_score": 0.87,
      "interpretation": "Outcome changes by 15% for each 100% violation of unconfoundedness",
      "violation_details": [
        {
          "severity": "mild",
          "violation_magnitude": 10.0,
          "outcome_estimate": 1.80,
          "deviation_percent": 5.0,
          "severity_score": 0.33
        }
      ]
    }
  ],
  "overall_robustness": 0.85,
  "critical_count": 0,
  "metadata": {
    "isl_version": "2.0.0",
    "request_id": "req_abc123",
    "timestamp": "2025-11-24T10:00:00Z"
  }
}
```

---

### 3.2 Feature 6: Explanation Quality Enhancement

**Purpose:** Progressive disclosure of causal concepts with readability optimization.

#### Architecture

```
ExplanationGenerator (src/services/explanation_generator.py)
│
├── generate_progressive_explanation()
│   ├── 1. Parse causal model
│   ├── 2. Extract key concepts (treatment, outcome, confounders)
│   ├── 3. For each level (simple/intermediate/technical):
│   │      ├── Load concept template (15 templates available)
│   │      ├── Substitute variables (treatment → "X", outcome → "Y")
│   │      ├── Adjust complexity (sentence length, vocabulary)
│   │      └── Calculate readability metrics
│   └── 4. Return multi-level explanation
│
├── assess_quality()
│   ├── Flesch Reading Ease: 206.835 - 1.015(words/sentence) - 84.6(syllables/word)
│   ├── Flesch-Kincaid Grade: 0.39(words/sentence) + 11.8(syllables/word) - 15.59
│   ├── SMOG Index: 1.0430 * sqrt(polysyllables * 30/sentences) + 3.1291
│   └── Return quality scores
│
└── improve_explanation()
    ├── Assess current quality
    ├── If FRE < 60 (difficult): simplify sentences
    ├── If FK Grade > 12 (college): reduce vocabulary
    └── Return improved text
```

#### Concept Templates (15 Templates)

| Concept | Simple | Intermediate | Technical |
|---------|--------|--------------|-----------|
| **Treatment Effect** | "When we apply the treatment, the outcome changes" | "The treatment has a causal effect on the outcome of magnitude δ" | "Under unconfoundedness and consistency, ATE = E[Y(1) - Y(0)]" |
| **Confounding** | "Other factors affect both treatment and outcome" | "Confounders create spurious associations between treatment and outcome" | "Confounding bias arises when E[Y(1)\|X=1] ≠ E[Y(1)\|X=0]" |
| **Identifiability** | "We can estimate the causal effect from data" | "The causal estimand is identifiable given the observed variables" | "P(Y\|do(X)) = Σ P(Y\|X,Z)P(Z) satisfies the backdoor criterion" |

#### API Contract

**Endpoint:** `POST /api/v1/explanations/progressive`

**Request:**
```json
{
  "model": {
    "nodes": ["Treatment", "Confounder", "Outcome"],
    "edges": [["Treatment", "Outcome"], ["Confounder", "Treatment"], ["Confounder", "Outcome"]],
    "treatment": "Treatment",
    "outcome": "Outcome"
  },
  "concepts": ["treatment_effect", "confounding", "identifiability"]
}
```

**Response:**
```json
{
  "explanations": [
    {
      "concept": "treatment_effect",
      "levels": {
        "simple": "When we apply the treatment, the outcome changes",
        "intermediate": "The treatment causes a change in the outcome. We can measure how much the outcome changes.",
        "technical": "Under the assumptions of unconfoundedness and consistency, the Average Treatment Effect (ATE) is E[Y(1) - Y(0)] where Y(1) and Y(0) are potential outcomes."
      },
      "quality_scores": {
        "simple": {
          "flesch_reading_ease": 85.2,
          "flesch_kincaid_grade": 4.1,
          "smog_index": 5.3
        },
        "technical": {
          "flesch_reading_ease": 32.1,
          "flesch_kincaid_grade": 16.8,
          "smog_index": 18.2
        }
      }
    }
  ],
  "metadata": {
    "isl_version": "2.0.0",
    "request_id": "req_xyz789"
  }
}
```

---

### 3.3 Feature 7: Causal Representation Learning

**Purpose:** Extract causal factors from unstructured text using ML embeddings.

#### Architecture

```
CausalRepresentationLearner (src/services/causal_representation_learner.py)
│
├── extract_factors_from_text()
│   ├── 1. Embed texts using sentence-transformers
│   │      Model: all-MiniLM-L6-v2 (384-dim embeddings)
│   │      Fallback: TF-IDF (100-dim vectors)
│   │
│   ├── 2. Cluster embeddings (K-means)
│   │      K: user-specified or auto-detect (3-10 clusters)
│   │      Distance metric: cosine similarity
│   │
│   ├── 3. For each cluster:
│   │      ├── Extract keywords (TF-IDF top-5)
│   │      ├── Name factor (most frequent keywords)
│   │      ├── Calculate strength (silhouette score)
│   │      ├── Calculate prevalence (cluster size / total)
│   │      └── Select representative texts (nearest to centroid)
│   │
│   └── 4. Suggest DAG structure
│          ├── Compute factor correlation matrix
│          ├── Apply threshold (corr > 0.3 → edge)
│          └── Return DAG (nodes=factors, edges=correlations)
│
├── discover_from_data()
│   ├── PC algorithm (constraint-based discovery)
│   ├── Conditional independence tests (χ² or Fisher's Z)
│   └── Return discovered DAG
│
└── suggest_dag_structure()
    ├── Combine text factors + data variables
    ├── Apply structural constraints (acyclicity)
    └── Return integrated DAG
```

#### ML Pipeline

```
Text Input: ["price increase led to demand drop", "quality improved sales", ...]
    ↓
Sentence Embedding (sentence-transformers)
    ↓
384-dim vectors: [[0.12, -0.43, ...], [0.89, 0.21, ...], ...]
    ↓
K-means Clustering (K=5)
    ↓
Clusters: {0: [text1, text5], 1: [text2, text7], ...}
    ↓
Keyword Extraction (TF-IDF per cluster)
    ↓
Factors: [
    {name: "Price Sensitivity", keywords: ["price", "cost"], strength: 0.82},
    {name: "Quality Impact", keywords: ["quality", "satisfaction"], strength: 0.75}
]
    ↓
DAG Suggestion (correlation threshold)
    ↓
Edges: [["Price Sensitivity", "Demand"], ["Quality Impact", "Sales"]]
```

#### API Contract

**Endpoint:** `POST /api/v1/discovery/extract-factors`

**Request:**
```json
{
  "texts": [
    "Price increase led to demand reduction",
    "Quality improvements increased customer satisfaction",
    "Marketing spend correlated with sales growth"
  ],
  "num_factors": 3,
  "min_cluster_size": 1
}
```

**Response:**
```json
{
  "factors": [
    {
      "name": "Price Sensitivity",
      "keywords": ["price", "cost", "demand"],
      "strength": 0.82,
      "prevalence": 0.33,
      "representative_texts": [
        "Price increase led to demand reduction"
      ],
      "cluster_id": 0
    },
    {
      "name": "Quality Impact",
      "keywords": ["quality", "satisfaction", "improvements"],
      "strength": 0.75,
      "prevalence": 0.33,
      "representative_texts": [
        "Quality improvements increased customer satisfaction"
      ],
      "cluster_id": 1
    }
  ],
  "suggested_dag": {
    "nodes": ["Price Sensitivity", "Quality Impact", "Marketing"],
    "edges": [
      ["Price Sensitivity", "Demand"],
      ["Quality Impact", "Satisfaction"],
      ["Marketing", "Sales"]
    ],
    "confidence": 0.78
  },
  "metadata": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "num_texts": 3,
    "num_factors": 3
  }
}
```

---

## 4. API Endpoints

### 4.1 Complete Endpoint Reference

#### Causal Validation

| Endpoint | Method | Purpose | Latency (P95) |
|----------|--------|---------|---------------|
| `/api/v1/validation/assumptions` | POST | Validate causal assumptions | 13.0ms |
| `/api/v1/validation/identifiability` | POST | Check identifiability | 15.2ms |
| `/api/v1/validation/batch` | POST | Batch validation | 45.0ms |

#### Counterfactual Generation

| Endpoint | Method | Purpose | Latency (P95) |
|----------|--------|---------|---------------|
| `/api/v1/counterfactual/generate` | POST | Generate counterfactuals | 245ms |
| `/api/v1/counterfactual/goal-seek` | POST | Goal-seeking scenarios | 380ms |
| `/api/v1/counterfactual/batch` | POST | Batch generation | 1.2s |

#### Sensitivity Analysis (NEW)

| Endpoint | Method | Purpose | Latency (P95) |
|----------|--------|---------|---------------|
| `/api/v1/sensitivity/analyze` | POST | Full sensitivity analysis | 180ms |
| `/api/v1/sensitivity/elasticity` | POST | Elasticity calculation only | 95ms |
| `/api/v1/sensitivity/critical` | GET | Get critical assumptions | 12ms |

#### Progressive Explanations (NEW)

| Endpoint | Method | Purpose | Latency (P95) |
|----------|--------|---------|---------------|
| `/api/v1/explanations/progressive` | POST | Multi-level explanations | 120ms |
| `/api/v1/explanations/quality` | POST | Readability assessment | 45ms |
| `/api/v1/explanations/improve` | POST | Auto-improve explanation | 85ms |

#### Causal Discovery (NEW)

| Endpoint | Method | Purpose | Latency (P95) |
|----------|--------|---------|---------------|
| `/api/v1/discovery/extract-factors` | POST | Extract factors from text | 850ms* |
| `/api/v1/discovery/from-data` | POST | PC algorithm discovery | 320ms |
| `/api/v1/discovery/suggest-dag` | POST | DAG structure suggestion | 180ms |

\* *Depends on text volume and embedding model. Uses caching for repeated requests.*

---

## 5. Integration Points

### 5.1 PLoT Engine Integration

**Purpose:** PLoT Engine validates causal models and generates counterfactuals for policy optimization.

#### Integration Flow

```
┌─────────────────────┐
│   PLoT Engine UI    │
│   (React/Next.js)   │
└─────────────────────┘
          │
          │ 1. User submits causal model (DAG + data)
          ▼
┌─────────────────────┐
│  PLoT Backend API   │
│   (FastAPI/Flask)   │
└─────────────────────┘
          │
          │ 2. Forward to ISL for validation
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/validation/      │
│       assumptions                    │
│                                      │
│  Returns:                            │
│  - unconfoundedness: PASS/FAIL      │
│  - positivity: PASS/FAIL            │
│  - identifiability: PASS/FAIL       │
└─────────────────────────────────────┘
          │
          │ 3. If PASS, request sensitivity analysis
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/sensitivity/     │
│       analyze                        │
│                                      │
│  Returns:                            │
│  - elasticity per assumption         │
│  - robustness scores                 │
│  - critical assumptions flagged      │
└─────────────────────────────────────┘
          │
          │ 4. If acceptable robustness, generate counterfactuals
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/counterfactual/  │
│       goal-seek                      │
│                                      │
│  Input: target_outcome=100.0         │
│  Returns: interventions needed       │
└─────────────────────────────────────┘
          │
          │ 5. Display results to user
          ▼
┌─────────────────────┐
│   PLoT Engine UI    │
│   (Results View)    │
└─────────────────────┘
```

#### API Contract: Validation Request

**PLoT → ISL:**
```javascript
// PLoT Backend sends:
const response = await fetch('https://isl.olumi.ai/api/v1/validation/assumptions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.ISL_API_KEY,
    'X-Request-ID': generateRequestId() // For tracing
  },
  body: JSON.stringify({
    model: {
      nodes: ["Marketing", "Price", "Demand", "Revenue"],
      edges: [
        ["Marketing", "Demand"],
        ["Price", "Demand"],
        ["Demand", "Revenue"]
      ],
      treatment: "Marketing",
      outcome: "Revenue"
    },
    data: plotData, // User's historical data
    assumptions: ["unconfoundedness", "positivity", "consistency"]
  })
});

// ISL responds:
{
  "results": [
    {
      "assumption": "unconfoundedness",
      "status": "pass",
      "confidence": 0.92,
      "details": "No unmeasured confounders detected in backdoor paths"
    },
    {
      "assumption": "positivity",
      "status": "warning",
      "confidence": 0.68,
      "details": "Low propensity scores detected for Marketing=0 (5% of samples)"
    }
  ],
  "overall_status": "pass_with_warnings",
  "metadata": {
    "isl_version": "2.0.0",
    "request_id": "plot_req_123",
    "timestamp": "2025-11-24T10:00:00Z"
  }
}
```

#### API Contract: Sensitivity Analysis

**PLoT → ISL:**
```javascript
const sensitivity = await fetch('https://isl.olumi.ai/api/v1/sensitivity/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.ISL_API_KEY
  },
  body: JSON.stringify({
    model: validatedModel, // From previous validation
    data: plotData,
    assumptions: ["unconfoundedness", "positivity"]
  })
});

// ISL responds with elasticity and robustness
```

#### API Contract: Counterfactual Generation

**PLoT → ISL:**
```javascript
const counterfactuals = await fetch('https://isl.olumi.ai/api/v1/counterfactual/goal-seek', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.ISL_API_KEY
  },
  body: JSON.stringify({
    model: validatedModel,
    data: plotData,
    target_outcome: 150.0, // User wants Revenue = 150
    max_interventions: 3,
    constraints: {
      "Marketing": {"min": 0, "max": 100},
      "Price": {"min": 10, "max": 50}
    }
  })
});

// ISL responds:
{
  "scenarios": [
    {
      "interventions": {
        "Marketing": 75.0,
        "Price": 25.0
      },
      "predicted_outcome": 150.2,
      "feasibility": 0.89,
      "cost_estimate": 45000
    }
  ],
  "metadata": {...}
}
```

---

### 5.2 TAE Integration

**Purpose:** TAE validates treatment effect assumptions and provides robustness assessments.

#### Integration Flow

```
┌─────────────────────┐
│      TAE UI         │
│  (Treatment Effect  │
│   Analyzer)         │
└─────────────────────┘
          │
          │ 1. User specifies treatment, outcome, covariates
          ▼
┌─────────────────────┐
│   TAE Backend       │
└─────────────────────┘
          │
          │ 2. Request assumption validation
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/validation/      │
│       assumptions                    │
│                                      │
│  Focus on:                           │
│  - Unconfoundedness (critical)       │
│  - SUTVA (stable unit treatment)     │
│  - Ignorability                      │
└─────────────────────────────────────┘
          │
          │ 3. Request sensitivity analysis
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/sensitivity/     │
│       analyze                        │
│                                      │
│  Returns:                            │
│  - How sensitive is ATE to hidden   │
│    confounding?                      │
│  - Robustness thresholds             │
└─────────────────────────────────────┘
          │
          │ 4. Filter results by robustness score
          ▼
┌─────────────────────┐
│   TAE UI            │
│   (Results: only    │
│    robust estimates │
│    displayed)       │
└─────────────────────┘
```

#### API Contract: TAE Robustness Filtering

**TAE → ISL:**
```javascript
// TAE sends treatment effect estimation request
const robustness = await fetch('https://isl.olumi.ai/api/v1/sensitivity/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.ISL_API_KEY
  },
  body: JSON.stringify({
    model: {
      nodes: ["Treatment", "Confounder1", "Confounder2", "Outcome"],
      edges: [
        ["Confounder1", "Treatment"],
        ["Confounder2", "Treatment"],
        ["Confounder1", "Outcome"],
        ["Confounder2", "Outcome"],
        ["Treatment", "Outcome"]
      ],
      treatment: "Treatment",
      outcome: "Outcome"
    },
    data: taeData,
    assumptions: ["unconfoundedness", "sutva"]
  })
});

// TAE filters based on robustness_score
const robustResults = robustness.metrics.filter(m => m.robustness_score > 0.75);

// Only display treatment effects with robustness > 0.75
```

---

### 5.3 CEE Integration

**Purpose:** CEE generates contrastive explanations for causal effects.

#### Integration Flow

```
┌─────────────────────┐
│      CEE UI         │
│  (Contrastive       │
│   Explanation       │
│   Engine)           │
└─────────────────────┘
          │
          │ 1. User views a causal effect estimate
          │    "Treatment X increases Outcome Y by 20%"
          ▼
┌─────────────────────┐
│   CEE Backend       │
└─────────────────────┘
          │
          │ 2. Request progressive explanation
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/explanations/    │
│       progressive                    │
│                                      │
│  Returns 3 levels:                   │
│  - Simple: "Treatment causes outcome"│
│  - Intermediate: "Treatment causes   │
│    20% increase in outcome"          │
│  - Technical: "ATE = 0.20 ± 0.05"   │
└─────────────────────────────────────┘
          │
          │ 3. User requests "Why not 30%?" (contrastive)
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/counterfactual/  │
│       generate                       │
│                                      │
│  Input: target_outcome = 30%         │
│  Returns: "Would need to change X    │
│            by 50% to achieve 30%"    │
└─────────────────────────────────────┘
          │
          │ 4. User requests causal factor discovery
          ▼
┌─────────────────────────────────────┐
│  ISL: POST /api/v1/discovery/       │
│       extract-factors                │
│                                      │
│  Input: domain_texts (user docs)     │
│  Returns: discovered factors +       │
│           suggested DAG               │
└─────────────────────────────────────┘
          │
          │ 5. Display integrated explanation
          ▼
┌─────────────────────┐
│      CEE UI         │
│  (Comprehensive     │
│   Explanation)      │
└─────────────────────┘
```

#### API Contract: Progressive Explanation

**CEE → ISL:**
```javascript
// CEE requests explanation for causal effect
const explanation = await fetch('https://isl.olumi.ai/api/v1/explanations/progressive', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.ISL_API_KEY
  },
  body: JSON.stringify({
    model: {
      nodes: ["Treatment", "Outcome"],
      edges: [["Treatment", "Outcome"]],
      treatment: "Treatment",
      outcome: "Outcome"
    },
    concepts: ["treatment_effect", "confounding", "identifiability"],
    effect_size: 0.20 // 20% increase
  })
});

// ISL returns 3-level explanation
// CEE displays appropriate level based on user's expertise
```

#### API Contract: Contrastive Explanation

**CEE → ISL:**
```javascript
// User asks "Why is the effect 20% and not 30%?"
const contrastive = await fetch('https://isl.olumi.ai/api/v1/counterfactual/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.ISL_API_KEY
  },
  body: JSON.stringify({
    model: ceeModel,
    data: ceeData,
    factual_outcome: 0.20, // Current effect
    counterfactual_outcome: 0.30, // Desired effect
    explanation_mode: "contrastive"
  })
});

// ISL responds:
{
  "contrastive_explanation": {
    "factual": "With current treatment level X=10, outcome Y=20%",
    "counterfactual": "To achieve Y=30%, treatment X must increase to 15",
    "delta": {
      "Treatment": 5.0,
      "required_change_percent": 50.0
    },
    "feasibility": 0.75,
    "reason": "Increasing treatment by 50% is feasible given historical data"
  }
}

// CEE displays: "The effect would be 30% if treatment increased by 50%"
```

---

### 5.4 UI Integration (All Workstreams)

**Purpose:** Unified UI components for all ISL-powered features.

#### Shared Components

```typescript
// src/components/ISLClient.tsx
import { ISLClient } from '@olumi/isl-client';

const client = new ISLClient({
  baseURL: process.env.NEXT_PUBLIC_ISL_URL,
  apiKey: process.env.ISL_API_KEY,
  timeout: 30000,
  retries: 3
});

// Validation Component (used by PLoT, TAE, CEE)
export const ValidationPanel = ({ model, data }) => {
  const [validation, setValidation] = useState(null);

  const validate = async () => {
    const result = await client.validation.checkAssumptions({
      model,
      data,
      assumptions: ['unconfoundedness', 'positivity']
    });
    setValidation(result);
  };

  return (
    <div>
      <button onClick={validate}>Validate Model</button>
      {validation && (
        <ValidationResults results={validation.results} />
      )}
    </div>
  );
};

// Sensitivity Analysis Component (used by PLoT, TAE)
export const SensitivityPanel = ({ model, data }) => {
  const [sensitivity, setSensitivity] = useState(null);

  const analyze = async () => {
    const result = await client.sensitivity.analyze({
      model,
      data,
      assumptions: ['unconfoundedness', 'positivity']
    });
    setSensitivity(result);
  };

  return (
    <div>
      <button onClick={analyze}>Analyze Sensitivity</button>
      {sensitivity && (
        <SensitivityChart metrics={sensitivity.metrics} />
      )}
    </div>
  );
};

// Progressive Explanation Component (used by CEE)
export const ExplanationPanel = ({ model, concepts }) => {
  const [level, setLevel] = useState('simple');
  const [explanation, setExplanation] = useState(null);

  const explain = async () => {
    const result = await client.explanations.progressive({
      model,
      concepts
    });
    setExplanation(result);
  };

  return (
    <div>
      <select value={level} onChange={(e) => setLevel(e.target.value)}>
        <option value="simple">Simple</option>
        <option value="intermediate">Intermediate</option>
        <option value="technical">Technical</option>
      </select>
      {explanation && (
        <ExplanationText text={explanation.levels[level]} />
      )}
    </div>
  );
};
```

---

## 6. Data Models

### 6.1 Core Data Models

#### CausalModel

```python
# src/models/causal_model.py

class CausalModel(BaseModel):
    """Directed Acyclic Graph (DAG) representation."""

    nodes: List[str]
    """List of variable names (nodes in the graph)."""

    edges: List[Tuple[str, str]]
    """List of directed edges [(parent, child), ...]."""

    treatment: str
    """Treatment variable name (must be in nodes)."""

    outcome: str
    """Outcome variable name (must be in nodes)."""

    confounders: Optional[List[str]] = None
    """Optional list of known confounders."""

    instrumental_variables: Optional[List[str]] = None
    """Optional list of instrumental variables."""

    @validator('treatment', 'outcome')
    def validate_node_exists(cls, v, values):
        if 'nodes' in values and v not in values['nodes']:
            raise ValueError(f"{v} must be in nodes list")
        return v

    @validator('edges')
    def validate_acyclic(cls, v, values):
        """Ensure graph is acyclic (DAG)."""
        # Topological sort check
        # Raises ValueError if cycle detected
        return v
```

#### SensitivityMetric

```python
# src/models/sensitivity.py

class ViolationDetail(BaseModel):
    """Details of a single violation scenario."""

    severity: str  # 'mild', 'moderate', 'severe'
    violation_magnitude: float  # Percentage (10.0, 25.0, 50.0)
    outcome_estimate: float
    deviation_percent: float
    severity_score: float  # 0.33, 0.67, 1.0

class SensitivityMetric(BaseModel):
    """Quantitative sensitivity analysis result for one assumption."""

    assumption: str
    baseline_outcome: float
    outcome_range: Tuple[float, float]
    elasticity: float
    critical: bool
    max_deviation_percent: float
    robustness_score: float
    interpretation: str
    violation_details: List[ViolationDetail]

class SensitivityAnalysisResponse(BaseModel):
    """Complete sensitivity analysis response."""

    metrics: List[SensitivityMetric]
    overall_robustness: float
    critical_count: int
    metadata: ResponseMetadata
```

#### ExplanationLevel

```python
# src/models/explanation_quality.py

class QualityScores(BaseModel):
    """Readability metrics for explanation text."""

    flesch_reading_ease: float  # 0-100 (higher = easier)
    flesch_kincaid_grade: float  # U.S. grade level
    smog_index: float  # Years of education required

class ExplanationLevel(BaseModel):
    """Single explanation level (simple/intermediate/technical)."""

    level: str  # 'simple', 'intermediate', 'technical'
    text: str
    quality_scores: QualityScores

class ProgressiveExplanation(BaseModel):
    """Multi-level explanation for a single concept."""

    concept: str
    levels: Dict[str, str]  # {'simple': '...', 'intermediate': '...', 'technical': '...'}
    quality_scores: Dict[str, QualityScores]

class ProgressiveExplanationResponse(BaseModel):
    """Complete progressive explanation response."""

    explanations: List[ProgressiveExplanation]
    metadata: ResponseMetadata
```

#### CausalFactor

```python
# src/models/causal_representation.py

class CausalFactor(BaseModel):
    """Extracted causal factor from text."""

    name: str
    keywords: List[str]
    strength: float  # Silhouette score [0, 1]
    prevalence: float  # Cluster size / total [0, 1]
    representative_texts: List[str]
    cluster_id: int

class DAGSuggestion(BaseModel):
    """Suggested DAG structure."""

    nodes: List[str]
    edges: List[Tuple[str, str]]
    confidence: float  # Overall confidence [0, 1]

class FactorExtractionResponse(BaseModel):
    """Complete factor extraction response."""

    factors: List[CausalFactor]
    suggested_dag: Optional[DAGSuggestion]
    metadata: ResponseMetadata
```

---

## 7. Performance & Scalability

### 7.1 Performance Benchmarks

**Measured with Apache Bench (ab) and custom profiling script:**

| Endpoint | P50 | P95 | P99 | Throughput |
|----------|-----|-----|-----|------------|
| **Validation** | 8.3ms | 13.0ms | 18.5ms | 120 req/s |
| **Counterfactual** | 125ms | 245ms | 380ms | 8 req/s |
| **Sensitivity** | 95ms | 180ms | 290ms | 11 req/s |
| **Explanations** | 65ms | 120ms | 185ms | 15 req/s |
| **Discovery (cached)** | 450ms | 850ms | 1.2s | 2 req/s |
| **Discovery (uncached)** | 2.1s | 4.5s | 6.8s | 0.5 req/s |

### 7.2 Caching Strategy

**Redis Cache (TTL: 5 minutes):**

```python
# Cache key structure
cache_key = f"isl:{endpoint}:{sha256(json.dumps(request_body))}"

# Example cache keys:
# - isl:validation:abc123def456...
# - isl:sensitivity:789ghi012jkl...
# - isl:discovery:mno345pqr678...

# Cache hit rate: 78% (after warmup)
# Cache size: ~500MB for 10K unique requests
```

**Cache Invalidation:**
- TTL-based: 5 minutes default
- Manual: `DELETE /api/v1/cache/{endpoint}/{hash}`
- Clear all: `DELETE /api/v1/cache/clear`

### 7.3 Scalability

**Horizontal Scaling:**

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isl-api
spec:
  replicas: 3  # Scale to 10+ for production
  selector:
    matchLabels:
      app: isl-api
  template:
    spec:
      containers:
      - name: isl-api
        image: olumi/isl-api:2.0.0
        resources:
          requests:
            cpu: 1000m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 1Gi
        env:
        - name: REDIS_URL
          value: redis://redis-service:6379
        - name: WORKERS
          value: "4"  # Gunicorn workers per pod
```

**Load Balancing:**
- AWS ALB with health checks (`/health`)
- Session affinity: None (stateless)
- Connection draining: 30s

**Auto-scaling:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: isl-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: isl-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 7.4 Rate Limiting

**Per-IP Rate Limits:**
- Default: 100 requests/minute per IP
- Burst: 150 requests/minute (sliding window)
- Whitelisted IPs: 1000 requests/minute

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1700000000
Retry-After: 42
```

**Response on Rate Limit Exceeded:**
```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit of 100 requests/minute exceeded",
    "retry_after": 42
  }
}
```

---

## 8. Security

### 8.1 Authentication

**API Key Authentication:**

```http
POST /api/v1/validation/assumptions
Content-Type: application/json
X-API-Key: isl_prod_<64-char-hex>

{
  "model": {...},
  "data": [...]
}
```

**Key Generation:**
```bash
# Generate production API key
openssl rand -hex 32

# Store in environment
ISL_API_KEY=isl_prod_7k9mP2nX8vQ4rL6wF3jH5tY1cB0zS...
```

**Key Rotation:**
- Recommended: Every 90 days
- Emergency rotation: Revoke old key, issue new key immediately
- Grace period: 24 hours (both keys valid)

### 8.2 CORS Configuration

**Production CORS:**
```python
# src/api/main.py
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")

# Example .env:
# CORS_ORIGINS=https://plot.olumi.ai,https://tae.olumi.ai,https://cee.olumi.ai

# NEVER use wildcard (*) in production
if not settings.RELOAD and "*" in CORS_ORIGINS:
    raise ValueError("Wildcard CORS not allowed in production")
```

**Allowed Origins:**
- https://plot.olumi.ai
- https://tae.olumi.ai
- https://cee.olumi.ai
- http://localhost:3000 (development only)

### 8.3 Data Protection

**Sensitive Data Handling:**
- No PII stored in logs
- No PII cached in Redis
- Data retention: 0 days (no persistence)
- Encryption: TLS 1.3 in transit

**Security Headers:**
```python
# Automatically added by FastAPI + middleware
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
```

### 8.4 Vulnerability Fixes (Recent)

**CRITICAL: Removed Hardcoded API Keys** ✅
- **Files:** `tests/smoke/quick_check.sh`, `PLOT_INTEGRATION_CHECKLIST.md`
- **Fix:** Environment variable support (`ISL_API_KEY`)

**HIGH: Fixed CORS Wildcard** ✅
- **File:** `src/api/main.py`
- **Fix:** Whitelist-only, runtime validation

**HIGH: Fixed Grafana Password** ✅
- **File:** `docker-compose.monitoring.yml`
- **Fix:** Environment variable (`GRAFANA_PASSWORD`)

**MEDIUM: Fixed Rate Limiting** ✅
- **File:** `src/middleware/rate_limiting.py`
- **Fix:** X-Forwarded-For header parsing

**MEDIUM: Pinned Docker Images** ✅
- **File:** `docker-compose.monitoring.yml`
- **Fix:** Specific versions (Prometheus v2.48.0, Grafana 10.2.2)

---

## 9. Deployment

### 9.1 Environment Variables

**Required (Production):**
```bash
# Security
ISL_API_KEY=<generate with: openssl rand -hex 32>
CORS_ORIGINS=https://plot.olumi.ai,https://tae.olumi.ai,https://cee.olumi.ai

# Monitoring
GRAFANA_PASSWORD=<generate with: openssl rand -base64 32>

# Redis
REDIS_URL=redis://redis:6379

# Prometheus
PROMETHEUS_URL=http://prometheus:9090
```

**Optional:**
```bash
# Performance
WORKERS=4  # Gunicorn workers
CACHE_TTL=300  # Redis TTL in seconds

# ML Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 9.2 Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY monitoring/ ./monitoring/

# Non-root user
RUN useradd -m -u 1000 isl && chown -R isl:isl /app
USER isl

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["gunicorn", "src.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "60"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  isl-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - ISL_API_KEY=${ISL_API_KEY}
      - CORS_ORIGINS=${CORS_ORIGINS}
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.48.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.2.2
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/dashboards
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 9.3 Kubernetes Deployment

**Full deployment manifest: `k8s/deployment.yaml`**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: isl

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isl-api
  namespace: isl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: isl-api
  template:
    metadata:
      labels:
        app: isl-api
    spec:
      containers:
      - name: isl-api
        image: olumi/isl-api:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: redis://redis-service:6379
        - name: ISL_API_KEY
          valueFrom:
            secretKeyRef:
              name: isl-secrets
              key: api-key
        - name: CORS_ORIGINS
          valueFrom:
            configMapKeyRef:
              name: isl-config
              key: cors-origins
        resources:
          requests:
            cpu: 1000m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: isl-api-service
  namespace: isl
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: isl-api
```

---

## 10. Testing

### 10.1 Test Coverage Summary

| Test Suite | Tests | Coverage | Status |
|------------|-------|----------|--------|
| **Unit Tests** | 95 | 92% | ✅ |
| **Integration Tests** | 50 | 88% | ✅ |
| **Client Tests** | 50 | 90% | ✅ |
| **E2E Tests** | 20 | 85% | ✅ |
| **TOTAL** | **215+** | **90%** | **✅** |

### 10.2 Running Tests

**All tests:**
```bash
pytest
```

**Unit tests only:**
```bash
pytest tests/test_*.py
```

**Integration tests:**
```bash
pytest tests/integration/
```

**Client library tests:**
```bash
cd isl-python-client && pytest
```

**Performance profiling:**
```bash
python scripts/performance_profiling.py --duration 30
```

### 10.3 Test Examples

**Unit Test: Sensitivity Analysis**
```python
# tests/test_sensitivity_analyzer.py

def test_elasticity_calculation():
    analyzer = SensitivityAnalyzer()

    result = analyzer.calculate_elasticity(
        baseline_outcome=100.0,
        violated_outcome=110.0,
        violation_magnitude=10.0
    )

    assert result.elasticity == 1.0  # 10% outcome change / 10% violation
    assert result.robustness_score > 0.8
```

**Integration Test: PLoT Workflow**
```python
# tests/integration/test_plot_workflows.py

@pytest.mark.asyncio
async def test_plot_standard_workflow(client):
    # 1. Validate model
    validation = await client.post("/api/v1/validation/assumptions", json={
        "model": PLOT_MODEL,
        "data": PLOT_DATA,
        "assumptions": ["unconfoundedness", "positivity"]
    })
    assert validation.json()["overall_status"] == "pass"

    # 2. Sensitivity analysis
    sensitivity = await client.post("/api/v1/sensitivity/analyze", json={
        "model": PLOT_MODEL,
        "data": PLOT_DATA
    })
    assert sensitivity.json()["overall_robustness"] > 0.75

    # 3. Generate counterfactuals
    counterfactuals = await client.post("/api/v1/counterfactual/goal-seek", json={
        "model": PLOT_MODEL,
        "data": PLOT_DATA,
        "target_outcome": 150.0
    })
    assert len(counterfactuals.json()["scenarios"]) > 0
```

---

## Appendix A: Quick Reference

### API Base URLs

| Environment | URL |
|-------------|-----|
| **Production** | https://isl.olumi.ai |
| **Staging** | https://isl-staging.olumi.ai |
| **Development** | http://localhost:8000 |

### Environment Variables Checklist

- [ ] `ISL_API_KEY` - API authentication key
- [ ] `CORS_ORIGINS` - Allowed CORS origins (comma-separated)
- [ ] `REDIS_URL` - Redis connection string
- [ ] `GRAFANA_PASSWORD` - Grafana admin password
- [ ] `PROMETHEUS_URL` - Prometheus endpoint (optional)
- [ ] `EMBEDDING_MODEL` - Sentence transformer model (optional)

### Common cURL Commands

**Validate Model:**
```bash
curl -X POST https://isl.olumi.ai/api/v1/validation/assumptions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{"model": {...}, "data": [...]}'
```

**Sensitivity Analysis:**
```bash
curl -X POST https://isl.olumi.ai/api/v1/sensitivity/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{"model": {...}, "data": [...]}'
```

**Progressive Explanation:**
```bash
curl -X POST https://isl.olumi.ai/api/v1/explanations/progressive \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{"model": {...}, "concepts": [...]}'
```

---

## Appendix B: Troubleshooting

### Common Issues

**Issue: 401 Unauthorized**
- **Cause:** Missing or invalid API key
- **Fix:** Set `X-API-Key` header with valid key

**Issue: 429 Rate Limit Exceeded**
- **Cause:** Exceeded 100 req/min per IP
- **Fix:** Wait for `Retry-After` seconds, implement backoff

**Issue: 503 Service Unavailable**
- **Cause:** Redis connection failed
- **Fix:** Check `REDIS_URL`, ensure Redis is running

**Issue: Slow Discovery Requests (>5s)**
- **Cause:** Uncached request with large text corpus
- **Fix:** Use caching, reduce num_factors, batch requests

---

## Appendix C: Changelog

**Version 2.0.0 (2025-11-24)**
- ✅ Added Research Feature 5: Enhanced Sensitivity Analysis
- ✅ Added Research Feature 6: Explanation Quality Enhancement
- ✅ Added Research Feature 7: Causal Representation Learning
- ✅ Added 50 integration tests (PLoT/TAE/CEE workflows)
- ✅ Added 50 client library tests (error handling, retry logic)
- ✅ Fixed 5 critical security vulnerabilities
- ✅ Added performance profiling and Grafana dashboards
- ✅ Pinned Docker image versions
- ✅ Added CORS environment variable support
- ✅ Added rate limiting proxy header handling

**Version 1.0.0 (2025-10-15)**
- Initial production release
- Causal validation endpoints
- Counterfactual generation endpoints
- Redis caching
- Prometheus metrics
- Basic integration tests

---

## Appendix D: Support & Contact

**Documentation:** https://docs.olumi.ai/isl
**GitHub:** https://github.com/Talchain/Inference-Service-Layer
**Issues:** https://github.com/Talchain/Inference-Service-Layer/issues
**Email:** support@olumi.ai

---

**End of Technical Specification**
