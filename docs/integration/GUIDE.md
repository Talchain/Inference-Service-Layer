# ISL Integration Guide

How to integrate ISL into PLoT, TAE, CEE, and UI applications.

---

## Overview

ISL provides causal inference via REST API. Integration pattern:

```
Your App → HTTP Request → ISL → HTTP Response → Your App
```

---

## Quick Integration

### 1. Get API Key

Request an API key from the ISL team. You'll receive:
- `ISL_API_KEY` - Your authentication key
- `ISL_BASE_URL` - Endpoint URL (staging or production)

### 2. Configure Your App

```bash
# Environment variables
ISL_BASE_URL=https://isl-staging.onrender.com
ISL_API_KEY=your_api_key_here
```

### 3. Make Requests

```python
import requests

response = requests.post(
    f"{ISL_BASE_URL}/api/v1/validation/assumptions",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": ISL_API_KEY
    },
    json={
        "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
        "treatment": "A",
        "outcome": "B"
    },
    timeout=60
)
```

---

## Integration Patterns

### PLoT Engine Integration

PLoT uses ISL for causal validation and counterfactual analysis.

**Typical Flow:**
```
1. User submits DAG in PLoT UI
2. PLoT → ISL: POST /validation/assumptions
3. ISL validates and returns results
4. PLoT → ISL: POST /sensitivity/analyze (if valid)
5. PLoT → ISL: POST /counterfactual/generate
6. PLoT displays combined results to user
```

**Example:**
```python
# PLoT Backend
async def analyze_causal_model(dag, treatment, outcome, data):
    # Step 1: Validate
    validation = await isl_client.post(
        "/api/v1/validation/assumptions",
        json={"dag": dag, "treatment": treatment, "outcome": outcome}
    )

    if not validation["is_valid"]:
        return {"error": "Model not identifiable", "details": validation}

    # Step 2: Sensitivity analysis
    sensitivity = await isl_client.post(
        "/api/v1/sensitivity/analyze",
        json={"model": {"dag": dag, "treatment": treatment, "outcome": outcome}, "data": data}
    )

    # Step 3: Only proceed if robust
    if sensitivity["overall_robustness"] < 0.7:
        return {"warning": "Results may be sensitive to assumptions", "sensitivity": sensitivity}

    # Step 4: Generate counterfactuals
    counterfactuals = await isl_client.post(
        "/api/v1/counterfactual/generate",
        json={"model": {...}, "intervention": {...}}
    )

    return {
        "validation": validation,
        "sensitivity": sensitivity,
        "counterfactuals": counterfactuals
    }
```

### TAE Integration

TAE uses ISL for robustness assessment.

**Pattern:**
```python
# Filter results by robustness
sensitivity = await isl_client.post("/api/v1/sensitivity/analyze", json={...})

# Only show robust results
robust_assumptions = [
    m for m in sensitivity["metrics"]
    if m["robustness_score"] > 0.75
]
```

### CEE Integration

CEE uses ISL for explanations and discovery.

**Progressive Disclosure:**
```typescript
// CEE Frontend
const explanation = await fetchISL("/api/v1/explanations/progressive", {
  model: causalModel,
  concepts: ["treatment_effect", "confounding"]
});

// Show appropriate level based on user expertise
const level = userIsExpert ? "technical" : "simple";
displayExplanation(explanation.explanations[0].levels[level]);
```

### UI Integration (Frontend)

For direct UI → ISL calls (bypassing backend).

**CORS:** Ensure your domain is in ISL's `CORS_ORIGINS`.

```typescript
// React example
const ISLClient = {
  baseUrl: process.env.NEXT_PUBLIC_ISL_URL,
  apiKey: process.env.ISL_API_KEY, // Keep server-side!

  async validate(dag: DAG, treatment: string, outcome: string) {
    const response = await fetch(`${this.baseUrl}/api/v1/validation/assumptions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": this.apiKey
      },
      body: JSON.stringify({ dag, treatment, outcome })
    });
    return response.json();
  }
};
```

**Security Note:** Never expose API keys in client-side code. Proxy through your backend.

---

## Error Handling

### Recommended Pattern

```python
import requests
from requests.exceptions import Timeout, RequestException

def call_isl(endpoint, payload, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{ISL_BASE_URL}{endpoint}",
                headers={"X-API-Key": ISL_API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()

            error = response.json()

            # Don't retry validation errors
            if response.status_code == 400:
                raise ValidationError(error["message"])

            # Retry rate limits with backoff
            if response.status_code == 429:
                wait = error.get("retry_after", 60)
                time.sleep(wait)
                continue

            # Retry server errors
            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

        except Timeout:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise Exception("Max retries exceeded")
```

### Error Codes

| Code | Action |
|------|--------|
| 400 | Fix input, don't retry |
| 401 | Check API key |
| 429 | Wait `retry_after` seconds, then retry |
| 504 | Simplify request or increase timeout |
| 500+ | Retry with exponential backoff |

---

## Best Practices

### 1. Use Request IDs

Pass `X-Request-Id` for distributed tracing:

```python
headers["X-Request-Id"] = f"plot-{uuid.uuid4()}"
```

### 2. Set Timeouts

ISL requests can take up to 60s for complex operations:

```python
response = requests.post(..., timeout=60)
```

### 3. Handle Partial Failures

If sensitivity analysis fails, you can still show validation results:

```python
result = {"validation": None, "sensitivity": None}

try:
    result["validation"] = await validate(...)
except Exception as e:
    log.error(f"Validation failed: {e}")

if result["validation"]:
    try:
        result["sensitivity"] = await sensitivity(...)
    except Exception as e:
        log.warning(f"Sensitivity failed, proceeding without: {e}")
```

### 4. Cache Results

ISL caches for 5 minutes. Same input = same response (fast).

For your app, consider additional caching:

```python
@cache(ttl=300)  # 5 minutes
def get_validation(dag_hash):
    return isl_client.validate(...)
```

### 5. Respect Rate Limits

Default: 100 requests/minute per API key.

```python
# Check remaining quota
remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
if remaining < 10:
    log.warning(f"Rate limit low: {remaining} remaining")
```

---

## Environment Configuration

### Staging

```bash
ISL_BASE_URL=https://isl-staging.onrender.com
ISL_API_KEY=your_staging_key
```

### Production

```bash
ISL_BASE_URL=https://isl.olumi.ai
ISL_API_KEY=your_production_key
```

---

## Health Checks

Verify ISL is available before making requests:

```python
def check_isl_health():
    try:
        response = requests.get(f"{ISL_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
```

---

## Troubleshooting

### "401 Unauthorized"
- Check `X-API-Key` header is present
- Verify API key is correct (no extra whitespace)
- Confirm key is active (not revoked)

### "429 Rate Limited"
- You're sending > 100 req/min
- Wait for `retry_after` seconds
- Consider batching requests

### "504 Timeout"
- Request took > 60 seconds
- Simplify input (fewer nodes/edges)
- Split into multiple smaller requests

### CORS Errors
- Your domain not in `CORS_ORIGINS`
- Contact ISL team to whitelist your domain
- Or proxy through your backend (recommended)

---

## Support

- **Issues:** GitHub Issues
- **Docs:** `/docs` in ISL repository
