# PLoT + ISL Integration Checklist

## Pre-Integration Setup

### Environment Configuration
- [ ] **API Key Configured**
  - Add ISL API key to PLoT environment variables
  - Key: `ISL_API_KEY=<your-production-api-key-here>`
  - **IMPORTANT**: Generate a strong random API key for production
  - Use: `openssl rand -hex 32` to generate a secure key
  - Location: `.env` or Kubernetes secrets
  - Verify key works: `curl -H "X-API-Key: $ISL_API_KEY" https://isl.olumi.com/health`

- [ ] **Base URL Configured**
  - Development: `ISL_BASE_URL=http://localhost:8000`
  - Staging: `ISL_BASE_URL=https://isl-staging.onrender.com`
  - Production: `ISL_BASE_URL=https://isl.olumi.com`

### Client Installation
- [ ] **TypeScript Client Generated**
  ```bash
  # Generate from OpenAPI spec
  curl https://isl.olumi.com/openapi.json > isl-openapi.json
  npx openapi-typescript-codegen --input isl-openapi.json --output ./src/clients/isl
  ```

- [ ] **HTTP Client Library**
  - Install: `npm install axios` (or fetch wrapper)
  - Configure timeout: 30 seconds
  - Configure retry logic: 3 attempts with exponential backoff

### Documentation Review
- [ ] **Read Integration Guide**
  - Location: `docs/integration/PLOT_INTEGRATION_GUIDE.md`
  - Understand 3 integration patterns
  - Review error handling strategies

- [ ] **Read API Quick Reference**
  - Location: `docs/API_QUICK_REFERENCE.md`
  - Familiarize with endpoints
  - Understand rate limits

---

## Integration Implementation

### Core ISL Client
- [ ] **Create ISL Client Module**
  ```typescript
  // src/clients/isl-client.ts
  class ISLClient {
    private baseURL: string;
    private apiKey: string;
    private cache: Cache;

    async validateCausal(request: CausalValidationRequest): Promise<ValidationResponse>
    async generateCounterfactual(request: CounterfactualRequest): Promise<CounterfactualResponse>
    async analyzeRobustness(request: RobustnessRequest): Promise<RobustnessResponse>
  }
  ```

- [ ] **Test Basic Connectivity**
  ```typescript
  const isl = new ISLClient({
    baseURL: process.env.ISL_BASE_URL,
    apiKey: process.env.ISL_API_KEY
  });

  const health = await isl.health();
  console.log("ISL version:", health.version);
  ```

### Pattern 1: Validate → Counterfactual
- [ ] **Implement in PLoT Program**
  ```typescript
  async function analyzeIntervention(program: PLoTProgram) {
    // Step 1: Validate
    const validation = await isl.validateCausal({
      dag: program.causalGraph,
      treatment: program.intervention.variable,
      outcome: program.targetOutcome
    });

    if (validation.status !== "identifiable") {
      return { error: "Not identifiable", reason: validation.explanation };
    }

    // Step 2: Counterfactual
    const cf = await isl.generateCounterfactual({
      causal_model: program.toISLFormat(),
      intervention: program.intervention.assignment,
      outcome_variables: [program.targetOutcome]
    });

    return { prediction: cf.prediction };
  }
  ```

- [ ] **Test with Sample Program**
  - Create test PLoT program with pricing intervention
  - Verify ISL validation returns identifiable
  - Verify counterfactual prediction is reasonable
  - Check response latency (<2s)

### Pattern 2: Robustness Validation
- [ ] **Implement Robustness Check**
  ```typescript
  async function validateRecommendationRobustness(intervention, prediction) {
    const robustness = await isl.analyzeRobustness({
      causal_model: intervention.causalModel,
      intervention_proposal: intervention.assignment,
      target_outcome: {
        [prediction.outcome]: [
          prediction.point_estimate * 0.95,
          prediction.point_estimate * 1.05
        ]
      },
      perturbation_radius: 0.1
    });

    return {
      isRobust: !robustness.analysis.is_fragile,
      score: robustness.analysis.robustness_score
    };
  }
  ```

- [ ] **Integrate into High-Stakes Decisions**
  - Add robustness check for interventions >$10k impact
  - Display confidence level to users
  - Show warnings for fragile recommendations

### Pattern 3: Multi-Option Comparison
- [ ] **Implement Option Comparison**
  ```typescript
  async function compareOptions(options: Intervention[]) {
    const predictions = await Promise.all(
      options.map(opt =>
        isl.generateCounterfactual({
          causal_model: opt.causalModel,
          intervention: opt.assignment,
          outcome_variables: [opt.targetOutcome]
        })
      )
    );

    return predictions
      .map((pred, i) => ({ option: options[i], prediction: pred }))
      .sort((a, b) => b.prediction.point_estimate - a.prediction.point_estimate);
  }
  ```

- [ ] **Test with Multiple Options**
  - Test with 3-5 pricing options
  - Verify parallel execution (latency <3s for 5 options)
  - Validate ranking is correct

---

## Error Handling

### Graceful Degradation
- [ ] **Implement Fallback Strategy**
  ```typescript
  async function callISLSafely<T>(operation: () => Promise<T>, fallback: T): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      if (error.status === 503) {
        logger.warn("ISL unavailable - using fallback");
        return fallback;
      }
      throw error;
    }
  }
  ```

- [ ] **Test Fallback Behavior**
  - Simulate ISL downtime
  - Verify PLoT continues to operate
  - Confirm user sees appropriate warning message

### Retry Logic
- [ ] **Implement Exponential Backoff**
  ```typescript
  async function retryWithBackoff<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3
  ): Promise<T> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await operation();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        if (error.status === 429 || error.status === 503) {
          await sleep(Math.pow(2, i) * 1000);
        } else {
          throw error;
        }
      }
    }
  }
  ```

- [ ] **Test Retry Logic**
  - Verify retries on 429 (rate limit)
  - Verify retries on 503 (service unavailable)
  - Confirm no retries on 400 (bad request)

### Error Logging
- [ ] **Log All ISL Interactions**
  ```typescript
  logger.info("ISL request", {
    endpoint: "validateCausal",
    requestId: generateRequestId(),
    request: sanitize(request)
  });
  ```

- [ ] **Track Error Metrics**
  - ISL error rate
  - Error types (400, 429, 503, network)
  - Impact on PLoT programs

---

## Caching Strategy

### Implementation
- [ ] **Create ISL Cache Module**
  ```typescript
  class ISLCache {
    private cache = new Map<string, CachedResponse>();

    async get(key: string): Promise<any | null>
    async set(key: string, value: any, ttl: number): Promise<void>

    generateKey(endpoint: string, request: any): string {
      return `${endpoint}:${JSON.stringify(request, Object.keys(request).sort())}`;
    }
  }
  ```

- [ ] **Cache Validation Responses**
  - TTL: 24 hours (validation results rarely change)
  - Key: `validate:${JSON.stringify({dag, treatment, outcome})}`
  - Hit rate target: >60%

- [ ] **Cache Counterfactual Responses**
  - TTL: 1 hour (may change with model updates)
  - Key: `counterfactual:${JSON.stringify(request)}`
  - Hit rate target: >40%

- [ ] **Skip Caching Robustness**
  - Robustness analysis uses randomness (not fully deterministic)
  - Only cache for <5 minutes if needed

### Testing
- [ ] **Validate Cache Behavior**
  - First call: Cache miss, ISL API call
  - Second call (same input): Cache hit, no API call
  - Different input: Cache miss, ISL API call
  - Expired cache: Cache miss, ISL API call

- [ ] **Monitor Cache Performance**
  - Track hit rate
  - Track latency improvement (cached vs. uncached)
  - Adjust TTL based on hit rate

---

## Testing

### Unit Tests
- [ ] **Mock ISL Responses**
  ```typescript
  jest.mock('./isl-client');

  test('should recommend intervention when identifiable', async () => {
    mockISL.validateCausal.mockResolvedValue({
      status: 'identifiable',
      formula: 'P(Y|do(X)) = ...'
    });

    mockISL.generateCounterfactual.mockResolvedValue({
      prediction: { point_estimate: 50000 }
    });

    const result = await analyzewithPLoT(program);
    expect(result.recommendation).toBe('PROCEED');
  });
  ```

- [ ] **Test Error Handling**
  - Mock 400 errors (invalid request)
  - Mock 429 errors (rate limit)
  - Mock 503 errors (service unavailable)
  - Verify appropriate PLoT behavior

### Integration Tests
- [ ] **Test Against ISL Staging**
  ```typescript
  const isl = new ISLClient({
    baseURL: 'https://isl-staging.onrender.com',
    apiKey: process.env.ISL_STAGING_KEY
  });

  test('full workflow: validate → counterfactual', async () => {
    const validation = await isl.validateCausal({...});
    expect(validation.status).toBe('identifiable');

    const cf = await isl.generateCounterfactual({...});
    expect(cf.prediction.point_estimate).toBeGreaterThan(0);
  });
  ```

- [ ] **Test Real PLoT Programs**
  - Run 3-5 representative PLoT programs
  - Verify ISL integration works end-to-end
  - Check latency is acceptable
  - Validate predictions are reasonable

### Performance Tests
- [ ] **Load Testing**
  - Simulate 10 concurrent PLoT programs
  - Measure latency (P50, P95, P99)
  - Verify no rate limit errors
  - Check cache hit rate improves over time

- [ ] **Latency Benchmarks**
  - Causal validation: <300ms
  - Counterfactual generation: <2s
  - Robustness analysis: <5s
  - Cached responses: <50ms

---

## Monitoring & Observability

### Metrics to Track
- [ ] **ISL Call Metrics**
  - Request count (by endpoint)
  - Success rate (by endpoint)
  - Latency (P50, P95, P99)
  - Error rate (by error type)

- [ ] **Cache Metrics**
  - Hit rate
  - Miss rate
  - Cache size
  - Eviction rate

- [ ] **Business Impact Metrics**
  - PLoT programs using ISL
  - Interventions validated
  - Recommendations rejected (not identifiable)
  - High-confidence recommendations (robustness >0.8)

### Dashboards
- [ ] **Create PLoT + ISL Dashboard**
  - ISL call volume over time
  - ISL error rate over time
  - Cache hit rate over time
  - Latency distribution

- [ ] **Set Up Alerts**
  - Alert: ISL error rate >5% (5min window)
  - Alert: ISL latency P95 >10s (5min window)
  - Alert: Cache hit rate <30% (1hr window)

### Logging
- [ ] **Structured Logging**
  ```typescript
  logger.info("ISL integration event", {
    event: "counterfactual_generated",
    program_id: program.id,
    intervention: intervention.variable,
    prediction: cf.prediction.point_estimate,
    latency_ms: latencyMs,
    cached: false
  });
  ```

- [ ] **Log Aggregation**
  - Send logs to central logging system
  - Enable searching by program_id, user_id, intervention
  - Set up log retention (30 days)

---

## Production Readiness

### Security
- [ ] **Secure API Key Storage**
  - Never commit API key to git
  - Store in environment variables or secrets manager
  - Rotate key quarterly
  - Monitor for unauthorized usage

- [ ] **Input Validation**
  - Validate DAGs before sending to ISL
  - Sanitize user inputs
  - Limit DAG size (max 100 nodes)
  - Validate intervention values are reasonable

### Documentation
- [ ] **Internal PLoT Documentation**
  - How to use ISL client
  - When to call each endpoint
  - Error handling guidelines
  - Troubleshooting common issues

- [ ] **Runbook for ISL Outages**
  - What happens when ISL is down
  - How PLoT degrades gracefully
  - Steps to escalate to ISL team
  - Expected resolution time

### Training
- [ ] **Team Onboarding**
  - Share integration guide with team
  - Walk through 3 integration patterns
  - Demo error handling and caching
  - Answer team questions

- [ ] **Code Review**
  - Review ISL integration code with team
  - Ensure error handling is robust
  - Verify caching strategy is sound
  - Check monitoring is comprehensive

---

## Launch

### Pre-Launch
- [ ] **All Checklist Items Complete**
- [ ] **Integration Tests Passing**
- [ ] **Monitoring Dashboards Live**
- [ ] **Team Trained**

### Launch Day
- [ ] **Deploy PLoT with ISL Integration**
- [ ] **Monitor ISL Metrics Closely**
  - Check error rate every 30min
  - Watch latency trends
  - Verify cache working

- [ ] **Test with Real Users**
  - Run 3-5 real PLoT programs
  - Verify recommendations make sense
  - Check user experience is smooth

### Post-Launch (First Week)
- [ ] **Daily Monitoring**
  - Review Grafana dashboards daily
  - Check for anomalies
  - Respond to alerts promptly

- [ ] **Performance Optimization**
  - Adjust cache TTL based on hit rate
  - Optimize slow queries
  - Tune retry backoff timings

- [ ] **Feedback Collection**
  - Gather team feedback
  - Note any integration issues
  - Document lessons learned

---

## Success Criteria

### Week 1
- ✅ ISL error rate <5%
- ✅ Cache hit rate >40%
- ✅ P95 latency <5s
- ✅ Zero production incidents

### Week 2
- ✅ 10+ PLoT programs using ISL
- ✅ Cache hit rate >50%
- ✅ Team comfortable with integration
- ✅ No escalations to ISL team

### Month 1
- ✅ 50+ PLoT programs using ISL
- ✅ Cache hit rate >60%
- ✅ Identified 2-3 optimization opportunities
- ✅ ISL integration considered stable

---

## Support

**Questions?**
- PLoT Team Slack: `#plot-team`
- ISL Team Slack: `#isl-integration`
- Email: isl-team@olumi.com

**Issues?**
- Create ticket with label: `isl-integration`
- For urgent issues: ping `@isl-oncall` in Slack

**Resources:**
- [PLoT Integration Guide](docs/integration/PLOT_INTEGRATION_GUIDE.md)
- [API Quick Reference](docs/API_QUICK_REFERENCE.md)
- [ISL Documentation](https://docs.olumi.com/isl)

---

**Status:** [ ] Not Started | [ ] In Progress | [ ] Complete
**Owner:** _________________
**Target Completion:** _________________
