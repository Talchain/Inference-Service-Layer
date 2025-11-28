# Sentry Error Tracking Setup

**Status:** Configuration Ready, Activation Pending
**Priority:** High
**Effort:** 30 minutes

---

## Overview

Sentry provides real-time error tracking, performance monitoring, and alerting for ISL. While the codebase references Sentry, it is not yet activated. This guide explains how to configure and enable Sentry for ISL.

---

## Quick Start

### 1. Get Sentry DSN

1. Create a Sentry project at https://sentry.io (or your self-hosted instance)
2. Navigate to **Settings** → **Projects** → **[Your Project]** → **Client Keys (DSN)**
3. Copy the DSN (looks like: `https://abc123@sentry.io/456789`)

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# Error Tracking (Sentry)
SENTRY_DSN=https://your-actual-dsn@sentry.io/project-id
SENTRY_ENVIRONMENT=production  # or staging, development
SENTRY_TRACES_SAMPLE_RATE=0.1  # 10% of transactions
SENTRY_PROFILES_SAMPLE_RATE=0.1  # 10% of profiles
SENTRY_ENABLED=true
```

### 3. Install Sentry SDK

```bash
poetry add sentry-sdk[fastapi]
```

### 4. Initialize Sentry in Application

Add to `src/api/main.py`:

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from src.config import get_settings

settings = get_settings()

if settings.SENTRY_ENABLED:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.SENTRY_ENVIRONMENT,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
        ],
        # Capture request IDs for correlation
        before_send=add_request_id_to_event,
        # Release tracking
        release=f"isl@{settings.VERSION}",
        # Custom tags
        tags={
            "service": "isl",
            "source": "inference-service-layer"
        },
    )
```

### 5. Add Request ID to Sentry Events

```python
def add_request_id_to_event(event, hint):
    """Add request_id to Sentry events for correlation."""
    from src.utils.tracing import get_trace_id

    request_id = get_trace_id()
    if request_id:
        event.setdefault("tags", {})["request_id"] = request_id
        event.setdefault("extra", {})["request_id"] = request_id

    return event
```

### 6. Verify Setup

Test Sentry is working:

```python
# In src/api/health.py or a test endpoint
@router.get("/sentry-test")
async def test_sentry():
    """Test Sentry integration (remove in production)."""
    try:
        1 / 0
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise

    return {"message": "Check Sentry for the error"}
```

---

## Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SENTRY_DSN` | Yes | None | Sentry Data Source Name |
| `SENTRY_ENVIRONMENT` | Yes | `"development"` | Environment name (development, staging, production) |
| `SENTRY_TRACES_SAMPLE_RATE` | No | `0.1` | Percentage of transactions to trace (0.0-1.0) |
| `SENTRY_PROFILES_SAMPLE_RATE` | No | `0.1` | Percentage of transactions to profile (0.0-1.0) |
| `SENTRY_ENABLED` | Yes | `false` | Enable/disable Sentry |

### Sample Rates

**Recommendations by Environment:**

| Environment | Traces Sample Rate | Profiles Sample Rate | Reasoning |
|-------------|-------------------|---------------------|-----------|
| **Development** | `1.0` (100%) | `1.0` (100%) | Capture everything for debugging |
| **Staging** | `0.5` (50%) | `0.3` (30%) | High sampling for testing |
| **Production** | `0.1` (10%) | `0.05` (5%) | Balance visibility vs. cost |

**Cost Considerations:**
- Sentry pricing is based on events and transactions
- Start with lower sample rates (10%) and increase if needed
- Use [quota management](https://docs.sentry.io/product/accounts/quotas/) to control costs

---

## Features

### 1. Error Tracking

Sentry automatically captures:
- ✅ Unhandled exceptions
- ✅ HTTP errors (4xx, 5xx)
- ✅ Validation errors
- ✅ Computation errors
- ✅ Stack traces with context

**Example:**
```python
try:
    result = compute_causal_effect(dag, model)
except Exception as e:
    # Automatically captured by Sentry
    logger.error("Computation failed", exc_info=True)
    raise
```

### 2. Performance Monitoring

Track performance metrics:
- ✅ Endpoint latency (P50, P95, P99)
- ✅ Database query performance
- ✅ External API calls
- ✅ Slow transactions

**View in Sentry:**
- **Performance** → **Transactions** → Filter by endpoint

### 3. Release Tracking

Track errors by version:
- ✅ Associate errors with specific releases
- ✅ Track error introduction/resolution
- ✅ Monitor deployment impact

**Configure:**
```python
sentry_sdk.init(
    release=f"isl@{settings.VERSION}",  # e.g., "isl@0.2.0"
)
```

### 4. Custom Context

Add ISL-specific context to errors:
- ✅ Request ID for correlation
- ✅ User ID (hashed for privacy)
- ✅ DAG structure details
- ✅ Computation parameters

**Example:**
```python
with sentry_sdk.configure_scope() as scope:
    scope.set_context("dag", {
        "node_count": len(dag.nodes),
        "edge_count": len(dag.edges),
        "treatment": treatment_var,
        "outcome": outcome_var,
    })
    scope.set_tag("computation_type", "counterfactual")
```

### 5. Breadcrumbs

Track events leading to errors:
- ✅ API calls
- ✅ Database queries
- ✅ Cache hits/misses
- ✅ Validation steps

**Example:**
```python
sentry_sdk.add_breadcrumb(
    category="causal_validation",
    message="Checking identifiability",
    level="info",
    data={"method": "backdoor", "treatment": "Price"}
)
```

---

## Integration with Olumi Error Schema

Sentry errors will include fields from our error schema:

```json
{
  "event_id": "sentry_event_id",
  "timestamp": "2025-11-27T10:30:00Z",
  "tags": {
    "request_id": "req_abc123def456",
    "source": "isl",
    "error_code": "ISL_CAUSAL_NOT_IDENTIFIABLE",
    "environment": "production"
  },
  "contexts": {
    "error_schema": {
      "code": "ISL_CAUSAL_NOT_IDENTIFIABLE",
      "retryable": false,
      "source": "isl"
    },
    "dag": {
      "node_count": 5,
      "edge_count": 6,
      "treatment": "Price",
      "outcome": "Revenue"
    }
  },
  "extra": {
    "request_id": "req_abc123def456",
    "validation_failures": ["No valid adjustment set found"],
    "attempted_methods": ["backdoor", "front_door"]
  }
}
```

---

## Alerting

### Configure Alerts in Sentry

1. **Error Rate Alert**
   - Condition: Error rate > 5% for 5 minutes
   - Action: Notify #isl-alerts Slack channel

2. **New Error Type Alert**
   - Condition: New error code appears
   - Action: Notify on-call engineer

3. **Performance Degradation**
   - Condition: P95 latency > 2s for 10 minutes
   - Action: Notify #isl-performance

### Slack Integration

1. Go to **Settings** → **Integrations** → **Slack**
2. Connect your workspace
3. Configure alert rules to post to specific channels

---

## Best Practices

### 1. Don't Log PII

❌ **Bad:**
```python
sentry_sdk.set_user({"email": user.email, "name": user.name})
```

✅ **Good:**
```python
from src.utils.secure_logging import hash_user_id

sentry_sdk.set_user({"id": hash_user_id(user.id)})
```

### 2. Use Structured Context

❌ **Bad:**
```python
logger.error(f"Failed for DAG with {len(nodes)} nodes")
```

✅ **Good:**
```python
with sentry_sdk.configure_scope() as scope:
    scope.set_context("dag", {"node_count": len(nodes)})
    logger.error("DAG validation failed")
```

### 3. Filter Sensitive Data

Configure Sentry to scrub sensitive fields:

```python
sentry_sdk.init(
    before_send=before_send_filter,
)

def before_send_filter(event, hint):
    # Remove sensitive fields
    if "request" in event:
        event["request"].get("data", {}).pop("api_key", None)
        event["request"].get("headers", {}).pop("Authorization", None)

    return event
```

### 4. Use Performance Tracing Selectively

Don't trace every operation:

```python
# Trace important operations
with sentry_sdk.start_transaction(op="causal_validation", name="validate_dag"):
    with sentry_sdk.start_span(op="y0", description="Identify adjustment sets"):
        adjustment_sets = identify_adjustment_sets(dag)

# Don't trace simple operations
result = simple_calculation()  # No transaction needed
```

---

## Monitoring

### Key Metrics to Track

1. **Error Rate**
   - Target: < 1%
   - Alert: > 5%
   - Dashboard: Sentry Performance → Overview

2. **P95 Latency**
   - Target: < 2s
   - Alert: > 5s
   - Dashboard: Sentry Performance → Transactions

3. **Most Common Errors**
   - Review: Weekly
   - Action: Fix top 5 errors
   - Dashboard: Sentry Issues → Trends

4. **Release Health**
   - Track: Errors introduced by each release
   - Review: After each deployment
   - Dashboard: Sentry Releases

---

## Cost Management

### Estimate Monthly Costs

**Assumptions:**
- 1M requests/month
- 10% trace sample rate = 100K transactions
- 5% profile sample rate = 50K profiles
- Error rate: 1% = 10K errors

**Sentry Pricing (Business Plan):**
- Errors: $26/month (included: 50K events)
- Transactions: $29/month (included: 100K transactions)
- Profiles: $15/month (included: 50K profiles)

**Total: ~$70/month** for production monitoring

### Reduce Costs

1. **Lower sample rates** for high-volume endpoints
2. **Filter out known errors** (e.g., rate limiting)
3. **Use quota management** to cap monthly costs
4. **Aggregate similar errors** to reduce event count

---

## Troubleshooting

### Sentry Not Capturing Errors

**Check:**
1. `SENTRY_ENABLED=true` in `.env`
2. `SENTRY_DSN` is correct
3. Sentry SDK installed: `poetry show sentry-sdk`
4. Network connectivity to Sentry

**Test:**
```bash
curl -X POST https://sentry.io/api/0/envelope/ \
  -H "Content-Type: application/x-sentry-envelope" \
  -H "X-Sentry-Auth: Sentry sentry_key=YOUR_KEY"
```

### Too Many Events

**Solutions:**
1. Lower `SENTRY_TRACES_SAMPLE_RATE`
2. Filter out noisy errors in `before_send`
3. Use [inbound filters](https://docs.sentry.io/product/data-management-settings/filtering/)

### Missing Context

**Ensure:**
1. Request ID added in `before_send`
2. Context set before error occurs
3. Breadcrumbs added at key points

---

## Production Deployment Checklist

- [ ] Sentry project created
- [ ] DSN added to environment variables
- [ ] `SENTRY_ENVIRONMENT` set correctly (production/staging)
- [ ] Sample rates configured appropriately
- [ ] Alert rules configured
- [ ] Slack integration set up
- [ ] Test error sent and received
- [ ] Performance monitoring enabled
- [ ] Release tracking configured
- [ ] Cost monitoring set up

---

## Resources

- **Sentry Docs:** https://docs.sentry.io/platforms/python/guides/fastapi/
- **FastAPI Integration:** https://docs.sentry.io/platforms/python/guides/fastapi/
- **Performance Monitoring:** https://docs.sentry.io/product/performance/
- **Pricing:** https://sentry.io/pricing/

---

**Status:** Documentation Complete, Activation Pending
**Next Step:** Add Sentry SDK to `pyproject.toml` and initialize in `main.py`
**Owner:** ISL Platform Team
