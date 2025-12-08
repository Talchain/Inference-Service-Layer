# Phase 1 Architecture Design

## Overview

Phase 1 adds three major feature areas to the Inference Service Layer:
1. **Adaptive Personalization** - Learn user preferences efficiently (ActiVA)
2. **Intelligent Teaching** - Help users understand concepts (Bayesian Teaching)
3. **Advanced Validation** - Catch model errors proactively

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                      │
├─────────────────┬────────────────────┬──────────────────────────┤
│  Preferences    │   Learning/Teaching │  Advanced Validation     │
│  /preferences/* │   /learning/*       │  /model/validate-advanced│
└────────┬────────┴──────────┬─────────┴────────────┬─────────────┘
         │                   │                      │
         ▼                   ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│ Preference      │  │ Bayesian        │  │ Advanced             │
│ Elicitor        │  │ Teacher         │  │ Model Validator      │
│ (ActiVA)        │  │                 │  │                      │
├─────────────────┤  ├─────────────────┤  ├──────────────────────┤
│ - Query Gen     │  │ - Example Gen   │  │ - Structural Checks  │
│ - Info Gain     │  │ - Teaching Val  │  │ - Statistical Checks │
│ - Entropy Calc  │  │ - KL Divergence │  │ - Domain Checks      │
└────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘
         │                    │                       │
         ▼                    ▼                       │
┌─────────────────┐  ┌─────────────────┐             │
│ Belief          │  │ Concept         │             │
│ Updater         │  │ Generators      │             │
│ (Bayesian)      │  │                 │             │
├─────────────────┤  ├─────────────────┤             │
│ - Bayesian Inf  │  │ - Confounding   │             │
│ - Posterior     │  │ - Trade-offs    │             │
│ - Likelihood    │  │ - Uncertainty   │             │
└────────┬────────┘  └─────────────────┘             │
         │                                            │
         ▼                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage & Caching Layer                       │
├──────────────────────┬──────────────────────┬───────────────────┤
│  User Beliefs Store  │   Response Cache     │  Query Results    │
│  (Redis/Database)    │   (Redis)            │  (Redis)          │
└──────────────────────┴──────────────────────┴───────────────────┘
         │                       │                      │
         └───────────────────────┴──────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Monitoring Layer                            │
├──────────────────────┬──────────────────────┬───────────────────┤
│  Prometheus Metrics  │   Logging            │  Health Checks    │
│  - Latency          │   - Structured JSON  │  - Dependencies   │
│  - Cache Hits       │   - Request Tracing  │  - Service Status │
└──────────────────────┴──────────────────────┴───────────────────┘
```

## Component Details

### 1. Preference Elicitation (ActiVA)

**Flow:**
```
User Request → PreferenceElicitor.generate_queries()
              ↓
         Check user beliefs in storage
              ↓
         Generate candidate scenarios
              ↓
         Compute information gain for each
              ↓
         Rank and select top N queries
              ↓
         Return queries to user
              ↓
User responds → BeliefUpdater.update_beliefs()
              ↓
         Compute likelihood of response
              ↓
         Bayesian update of belief distribution
              ↓
         Store updated beliefs
              ↓
         Generate next queries (if needed)
```

**Key Algorithms:**
- **Information Gain**: H(current) - E[H(posterior)]
- **Monte Carlo Sampling**: 1000 samples for expectation estimation
- **Bayesian Update**: P(θ|D) ∝ P(D|θ) × P(θ)

**Storage:**
- User beliefs: Redis hash with TTL (24 hours)
- Query history: Redis sorted set (for tracking progress)

### 2. Bayesian Teaching

**Flow:**
```
Teaching Request → BayesianTeacher.select_examples()
                   ↓
              Define target belief
                   ↓
              Generate candidate examples
                   ↓
              Compute teaching value (KL divergence reduction)
                   ↓
              Select diverse, high-value examples
                   ↓
              Format with explanations
                   ↓
              Return to user
```

**Key Algorithms:**
- **Teaching Value**: KL(current || target) - KL(posterior || target)
- **Diversity Selection**: Max-min diversity across examples

**Concepts Supported:**
- Confounding
- Trade-offs
- Uncertainty
- Robustness
- Mediators
- Colliders

### 3. Advanced Model Validation

**Flow:**
```
Validation Request → AdvancedValidator.validate()
                     ↓
               Structural Checks
               - Complexity
               - Colliders
               - Mediators
               - Cycles
               - Isolation
                     ↓
               Statistical Checks (if model provided)
               - Distributions
               - Parameters
               - Identifiability
                     ↓
               Domain Checks (if context provided)
               - Pricing patterns
               - Feature patterns
               - Marketing patterns
                     ↓
               Generate Suggestions
               - Missing edges
               - Problematic structures
               - Refinements
                     ↓
               Compute Quality Score
                     ↓
               Return comprehensive results
```

**Validation Checks:**

**Structural:**
- Appropriate complexity (5-20 nodes ideal)
- Collider detection and warnings
- Mediator identification
- No isolated nodes
- Logical temporal flow

**Statistical:**
- Reasonable distribution parameters
- Parameter magnitude checks
- Model identifiability

**Domain-Specific:**
- Pricing: price → revenue → churn chain
- Features: satisfaction → retention → revenue
- Marketing: attribution → conversion → revenue

### 4. Caching Strategy

**Cache Layers:**

1. **Response Cache** (Redis)
   - Key: `canonical_hash(request)`
   - TTL: 24 hours (most endpoints), 1 hour (preferences)
   - Invalidation: On user preference update

2. **Computation Cache**
   - Y₀ identification results
   - Monte Carlo samples
   - Information gain calculations

3. **User Belief Cache**
   - Current belief models
   - Query history
   - Learning progress

**Cache Hit Targets:**
- Causal validation: 60%+ (many repeated DAGs)
- Counterfactual: 40%+ (scenario comparisons)
- Preferences: 20%+ (mostly unique, but sequence matters)

### 5. Monitoring & Observability

**Metrics to Track:**

```python
# Request metrics
- inference_requests_total{endpoint, status}
- inference_request_duration_seconds{endpoint}

# Preference learning metrics
- preference_queries_generated_total
- preference_entropy_reduction{user_id}
- preference_learning_complete{user_id}

# Cache metrics
- cache_hits_total{endpoint}
- cache_misses_total{endpoint}
- cache_hit_rate{endpoint}

# Validation metrics
- validation_issues_found{severity}
- validation_suggestions_generated

# System metrics
- active_users_gauge
- user_beliefs_stored_gauge
```

**Logging Strategy:**

All logs structured JSON with:
- `request_id`: Unique per request
- `user_id`: (hashed for privacy)
- `endpoint`: API endpoint
- `duration_ms`: Request duration
- `status`: success/error
- `deterministic_hash`: For reproducibility

**Critical Logs:**
- Preference query generation (info gain, strategy)
- Belief updates (entropy before/after)
- Validation issues (severity, type)
- Cache performance (hit/miss, latency)

## Data Models

### User Belief Storage

```python
# Redis key structure
user:beliefs:{user_id} → UserBeliefModel (JSON)
user:queries:{user_id} → Sorted set of query IDs (by timestamp)
user:responses:{user_id} → List of responses (for analysis)

# TTL
beliefs: 24 hours (extend on activity)
queries: 7 days
responses: 30 days (anonymized)
```

### Cache Storage

```python
# Redis key structure
cache:validate:{dag_hash} → CausalValidationResponse
cache:counterfactual:{request_hash} → CounterfactualResponse
cache:teaching:{concept}:{context_hash} → TeachingExamples

# TTL
validate: 24 hours
counterfactual: 24 hours
teaching: 12 hours
```

## Performance Targets

**Response Times (P95):**
- Preference elicitation: < 1.5s
- Belief update: < 0.5s
- Teaching examples: < 1.0s
- Advanced validation: < 2.0s

**Throughput:**
- 100+ concurrent users
- 1000+ requests/minute

**Cache Effectiveness:**
- Hit rate: 40%+
- Latency reduction: 50%+ on hits

## Error Handling

**Graceful Degradation:**

1. **Preference Service Unavailable:**
   - Return generic queries
   - Log warning
   - Continue with reduced personalization

2. **Cache Unavailable:**
   - Compute directly
   - Log warning
   - Continue without caching

3. **Storage Unavailable:**
   - Use in-memory fallback (session only)
   - Log error
   - Warn user about temporary storage

**Error Categories:**

- **Retryable:** Network timeouts, temporary service unavailability
- **Non-Retryable:** Invalid input, authentication failure
- **Degraded:** Cache miss (still returns result)

## Security & Privacy

**User Data Protection:**

1. **No PII in logs:**
   - Hash user IDs before logging
   - Never log raw preference responses
   - Aggregate metrics only

2. **Data Retention:**
   - Beliefs: 24 hours (extend on activity)
   - Queries: 7 days
   - Responses: 30 days anonymized, then deleted

3. **Access Control:**
   - User can only access own beliefs
   - Admin endpoints for monitoring only

## Testing Strategy

**Unit Tests:**
- Each service in isolation
- Mock dependencies
- Test edge cases
- Verify determinism

**Integration Tests:**
- Full API request/response cycles
- Cache integration
- Storage integration
- Cross-service flows

**End-to-End Tests:**
- Complete preference learning flow (5-7 queries)
- Teaching concept to mastery
- Advanced validation catching real issues
- Performance under load

**Performance Tests:**
- Load testing (100+ concurrent users)
- Cache effectiveness
- Response time percentiles
- Resource utilization

## Migration Plan

**Phase 0 → Phase 1:**

1. Add new dependencies (Redis, Prometheus client)
2. Deploy cache layer
3. Deploy new services (backward compatible)
4. Enable new endpoints
5. Monitor performance
6. Gradually increase traffic

**Rollback Plan:**
- All Phase 1 endpoints optional
- Phase 0 endpoints unchanged
- Can disable Phase 1 via feature flags

## Success Criteria

✅ Preference learning reaches 80% confidence in 5-7 queries
✅ Teaching improves concept understanding by 30%+
✅ Advanced validation catches 90%+ of model issues
✅ P95 response time < 1.5s (30% improvement)
✅ Cache hit rate > 40%
✅ Support 100+ concurrent users
✅ All Phase 0 functionality preserved

---

**Next Steps:**
1. Implement PreferenceElicitor service
2. Implement BeliefUpdater service
3. Add user storage layer
4. Create API endpoints
5. Comprehensive testing
6. Performance optimization
