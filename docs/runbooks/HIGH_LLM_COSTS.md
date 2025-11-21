# Runbook: ISL High LLM Costs

## Alert
**Alert Name:** `ISLLLMCostsHigh`
**Severity:** Warning (>$10/hour), Critical (>$50/hour)
**Threshold:** LLM costs > $10/hour for 10 minutes
**Dashboard:** https://grafana.olumi.com/d/isl-llm-costs

## Symptoms
- LLM cost alert firing in #isl-costs
- Grafana cost dashboard shows spike
- Unexpected AWS/OpenAI bill projection
- Budget alerts from finance team

## Impact
- **Budget:** Monthly projection exceeds allocated budget
- **Sustainability:** Unsustainable cost growth
- **Pilot:** May need to limit usage or disable features
- **Financial:** Unplanned expense, requires justification

## Diagnosis

### Step 1: Identify Cost Drivers
```bash
# Get cost breakdown by model (last hour)
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_cost_dollars{model=' \
  | awk -F'model="' '{print $2}' \
  | awk -F'"' '{print $1, $NF}' \
  | sort -k2 -rn

# Expected output:
# gpt-4 8.50
# gpt-3.5-turbo 1.20
# claude-3-sonnet 0.30

# Top cost-generating sessions
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_cost_dollars{session_id=' \
  | sort -t'=' -k2 -rn \
  | head -10
```

### Step 2: Check Request Volume
```bash
# LLM requests per endpoint (last hour)
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_requests_total{endpoint=' \
  | awk '{sum+=$NF} END {print "Total:", sum}'

# By endpoint
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_requests_total{endpoint=' \
  | awk -F'endpoint="' '{print $2}' \
  | awk -F'"' '{print $1, $NF}' \
  | sort -k2 -rn

# Cache hit rate (should be >30%)
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_cache_hits_total'

curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_requests_total' \
  | head -1
```

### Step 3: Check for Abuse or Anomalies
```bash
# Requests by user/API key
kubectl logs -n production deployment/isl --since=1h \
  | grep "LLM request" \
  | awk '{print $(NF-2)}' \
  | sort | uniq -c | sort -rn | head -10

# Expensive sessions (>$1)
kubectl logs -n production deployment/isl --since=1h \
  | grep "Session cost" \
  | awk '$NF > 1.0 {print $0}'

# Check for loops or repeated requests
kubectl logs -n production deployment/isl --since=1h \
  | grep "deliberation" \
  | grep "round" \
  | awk '{print $session_id, $round}' \
  | sort | uniq -c \
  | awk '$1 > 10 {print $0}'  # Sessions with >10 rounds
```

### Step 4: Project Monthly Cost
```bash
# Current hourly rate
HOURLY_COST=$(curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_cost_dollars' \
  | awk '{sum+=$NF} END {print sum}')

echo "Current hourly rate: \$$HOURLY_COST"
echo "Projected daily: \$$(echo "$HOURLY_COST * 24" | bc)"
echo "Projected monthly: \$$(echo "$HOURLY_COST * 24 * 30" | bc)"
```

## Common Causes & Solutions

### Cause 1: Low Cache Hit Rate
**Symptom:** Cache hit rate <20%

**Diagnosis:**
```bash
# Check cache metrics
curl -s http://isl.olumi.com/metrics \
  | grep -E "(cache_hits|llm_requests)" \
  | grep "total"

# Check Redis health
redis-cli -h redis.olumi.com INFO stats
redis-cli -h redis.olumi.com INFO memory
```

**Fix:**
```bash
# Increase cache TTL
kubectl set env deployment/isl \
  LLM_CACHE_TTL_SECONDS=7200 \
  -n production

# Verify Redis has adequate memory
redis-cli -h redis.olumi.com CONFIG GET maxmemory

# If Redis memory low, increase allocation
# Contact infrastructure team
```

**Expected Outcome:**
- Cache hit rate increases to >40%
- LLM request volume decreases
- Costs drop proportionally

**Validation:**
```bash
# Monitor cache hit rate
watch -n 5 'curl -s http://isl.olumi.com/metrics | grep cache_hits'
```

### Cause 2: Excessive Deliberation Rounds
**Symptom:** Sessions taking 10+ rounds to converge

**Diagnosis:**
```bash
# Check rounds per session
kubectl logs -n production deployment/isl --since=1h \
  | grep "Deliberation round" \
  | awk '{print $session_id}' \
  | sort | uniq -c \
  | sort -rn

# Check average rounds
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_habermas_rounds_per_session'
```

**Fix:**
```bash
# Reduce max rounds per session
kubectl set env deployment/isl \
  HABERMAS_MAX_ROUNDS=5 \
  -n production

# Use cheaper model for value extraction
kubectl set env deployment/isl \
  LLM_EXTRACTION_MODEL=gpt-3.5-turbo \
  -n production

# Keep GPT-4 only for consensus (higher quality needed)
# LLM_CONSENSUS_MODEL=gpt-4 (already set)
```

**Expected Outcome:**
- Average rounds per session <6
- Cost per deliberation <$0.08
- Still achieve consensus in most cases

**Validation:**
```bash
# Check updated metrics
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_habermas_rounds'
```

### Cause 3: Single User/Session Abuse
**Symptom:** One session/user generating 80%+ of costs

**Diagnosis:**
```bash
# Identify expensive session
EXPENSIVE_SESSION=$(curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_cost_dollars{session_id=' \
  | sort -t'=' -k2 -rn \
  | head -1 \
  | awk -F'session_id="' '{print $2}' \
  | cut -d'"' -f1)

echo "Most expensive session: $EXPENSIVE_SESSION"

# Get session details
kubectl logs -n production deployment/isl \
  | grep "$EXPENSIVE_SESSION" \
  | tail -20
```

**Fix:**
```bash
# Terminate abusive session
curl -X DELETE "http://isl.olumi.com/api/v1/deliberation/session/$EXPENSIVE_SESSION" \
  -H "X-API-Key: $ADMIN_API_KEY"

# Get user_id from session
USER_ID=$(kubectl logs -n production deployment/isl \
  | grep "$EXPENSIVE_SESSION" \
  | grep "user_id" \
  | head -1 \
  | awk -F'user_id=' '{print $2}' \
  | cut -d' ' -f1)

# Rate limit user temporarily
kubectl set env deployment/isl \
  "RATE_LIMIT_EXCEPTION_$USER_ID=10" \
  -n production

# Contact user to understand usage pattern
```

**Expected Outcome:**
- Cost spike stops immediately
- User contacted and educated
- Normal usage resumed

**Validation:**
```bash
# Verify session terminated
curl -s http://isl.olumi.com/metrics \
  | grep "$EXPENSIVE_SESSION" \
  | wc -l  # Should be 0
```

### Cause 4: Model Selection Too Expensive
**Symptom:** Using GPT-4 for all LLM requests

**Diagnosis:**
```bash
# Check model distribution
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_requests_total{model=' \
  | awk -F'model="' '{print $2}' \
  | awk -F'"' '{print $1, $NF}'

# Expected: mostly gpt-3.5-turbo, some gpt-4
# Bad: all gpt-4
```

**Fix:**
```bash
# Switch consensus to GPT-4-turbo (cheaper, similar quality)
kubectl set env deployment/isl \
  LLM_CONSENSUS_MODEL=gpt-4-turbo-preview \
  -n production

# Use GPT-3.5-turbo for value extraction (adequate quality)
kubectl set env deployment/isl \
  LLM_EXTRACTION_MODEL=gpt-3.5-turbo \
  -n production

# Consider Claude Sonnet for some tasks (cheaper alternative)
kubectl set env deployment/isl \
  LLM_PROVIDER_SECONDARY=anthropic \
  LLM_SECONDARY_MODEL=claude-3-sonnet \
  -n production
```

**Expected Outcome:**
- 50-70% cost reduction
- Quality remains acceptable
- Faster responses (GPT-3.5 lower latency)

**Cost Comparison:**
- GPT-4: $0.03 input, $0.06 output per 1K tokens
- GPT-4-turbo: $0.01 input, $0.03 output per 1K tokens
- GPT-3.5-turbo: $0.0015 input, $0.002 output per 1K tokens
- Claude-3-sonnet: $0.003 input, $0.015 output per 1K tokens

### Cause 5: Prompt Engineering Inefficiency
**Symptom:** Excessive token usage per request

**Diagnosis:**
```bash
# Check average tokens per request
curl -s http://isl.olumi.com/metrics \
  | grep 'isl_llm_tokens_total'

# Check prompt lengths in logs
kubectl logs -n production deployment/isl --since=1h \
  | grep "Prompt tokens" \
  | awk '{print $NF}' \
  | awk '{sum+=$1; count++} END {print "Average:", sum/count}'
```

**Fix:**
- **Review prompts** for unnecessary verbosity
- **Use few-shot examples** sparingly (expensive)
- **Truncate context** if >2000 tokens
- **Use prompt caching** for system messages

```python
# Example optimization in code:
# Before: 500 token system prompt + 300 token user prompt = 800 tokens
# After: 200 token system prompt + 250 token user prompt = 450 tokens
# Savings: 43% fewer input tokens
```

**Expected Outcome:**
- 20-40% reduction in token usage
- Lower costs per request
- Faster response times

## Immediate Mitigation

### Option 1: Enable Aggressive Fallback
```bash
# Fall back to rules if session cost exceeds $0.50
kubectl set env deployment/isl \
  LLM_FALLBACK_TO_RULES=true \
  MAX_COST_PER_SESSION=0.50 \
  -n production

# Verify setting
kubectl get deployment isl -n production -o yaml \
  | grep -A 2 "MAX_COST_PER_SESSION"
```

**Impact:**
- Deliberation quality degraded (rule-based)
- Costs immediately controlled
- Users still have functionality

### Option 2: Disable LLM Temporarily (Emergency Only)
```bash
# ⚠️  WARNING: This disables LLM features entirely
kubectl set env deployment/isl \
  LLM_ENABLED=false \
  -n production

# Verify
kubectl logs -n production deployment/isl -f \
  | grep "LLM disabled"
```

**Impact:**
- Habermas Machine uses rule-based only
- Value extraction via keyword matching
- Consensus via templates
- **Use only if costs critical**

**Rollback:**
```bash
# Re-enable when costs under control
kubectl set env deployment/isl \
  LLM_ENABLED=true \
  LLM_FALLBACK_TO_RULES=false \
  -n production
```

### Option 3: Set Daily Budget Limit
```bash
# Set hard daily limit ($100)
kubectl set env deployment/isl \
  LLM_DAILY_BUDGET_LIMIT=100.0 \
  -n production

# ISL will automatically disable LLM when exceeded
# Resets at midnight UTC
```

## Prevention

### Daily Monitoring
- [ ] Review LLM cost dashboard every morning
- [ ] Check for cost anomalies
- [ ] Verify cache hit rate >30%
- [ ] Review top expensive sessions

### Weekly Review
- [ ] Analyze cost trends (week-over-week)
- [ ] Identify optimization opportunities
- [ ] Review model selection
- [ ] Optimize prompts if needed

### Monthly Planning
- [ ] Review budget vs. actual
- [ ] Project costs for next month
- [ ] Evaluate LLM provider pricing changes
- [ ] Consider switching providers if beneficial

### Technical Improvements
1. **Optimize prompts** - Reduce token usage by 30%
2. **Increase cache TTL** - Target >50% hit rate
3. **Implement request batching** - Reduce API calls
4. **Add cost attribution** - Track by team/feature
5. **Use streaming** - Reduce latency perception
6. **Implement rate limiting** - Per user/session
7. **Add cost budgets** - Per team/project

## Escalation

**If costs >$50/hour:**

1. **Immediate:** Enable fallback mode or disable LLM
2. **Within 5 min:** Notify #isl-incidents and finance team
3. **Within 15 min:** Identify root cause
4. **Within 30 min:** Implement permanent fix
5. **Within 1 hour:** Post-mortem started

**Contacts:**
- **Finance:** finance@olumi.com
- **Engineering Lead:** eng-lead@olumi.com
- **OpenAI Support:** support@openai.com
- **Anthropic Support:** support@anthropic.com

## Cost Optimization Checklist

### Immediate (within 1 hour)
- [ ] Increase cache TTL to 7200s
- [ ] Switch to cheaper models where appropriate
- [ ] Enable fallback for high-cost sessions
- [ ] Terminate any abusive sessions

### Short-term (within 24 hours)
- [ ] Review and optimize prompts
- [ ] Implement per-session budgets
- [ ] Add cost monitoring alerts
- [ ] Analyze usage patterns

### Long-term (within 1 week)
- [ ] Implement prompt caching
- [ ] Add request batching
- [ ] Evaluate alternative providers
- [ ] Build cost attribution system
- [ ] Create cost optimization dashboard

## Post-Incident

### Analysis
```bash
# Calculate total cost impact
TOTAL_COST=$(kubectl logs -n production deployment/isl \
  | grep "LLM cost" \
  | awk '{sum+=$cost} END {print sum}')

echo "Total incident cost: \$$TOTAL_COST"

# Compare to normal
NORMAL_HOURLY_COST=5.00
INCIDENT_DURATION_HOURS=2
EXPECTED_COST=$(echo "$NORMAL_HOURLY_COST * $INCIDENT_DURATION_HOURS" | bc)
OVERSPEND=$(echo "$TOTAL_COST - $EXPECTED_COST" | bc)

echo "Expected cost: \$$EXPECTED_COST"
echo "Overspend: \$$OVERSPEND"
```

### Documentation
- [ ] Create post-mortem
- [ ] Update this runbook
- [ ] Share learnings with team
- [ ] Update cost dashboard
- [ ] Create Jira tickets for improvements

### Communication
- [ ] Notify finance team of overspend
- [ ] Explain to users if service degraded
- [ ] Update stakeholders on prevention measures
- [ ] Schedule team retro

## Related Runbooks
- [HIGH_ERROR_RATE.md](HIGH_ERROR_RATE.md)
- [LOW_CACHE_HIT_RATE.md](LOW_CACHE_HIT_RATE.md)
- [SERVICE_DOWN.md](SERVICE_DOWN.md)

## References
- **Cost Dashboard:** https://grafana.olumi.com/d/isl-llm-costs
- **Metrics:** http://isl.olumi.com/metrics
- **OpenAI Pricing:** https://openai.com/pricing
- **Anthropic Pricing:** https://www.anthropic.com/pricing
- **Budget Tracking:** Spreadsheet link
