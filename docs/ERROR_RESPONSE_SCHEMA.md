# ISL Error Response Schema (v1.0)

**Status:** Implemented
**Standard:** Olumi Platform Error Schema v1.0
**Service:** Inference Service Layer (ISL)
**Date:** November 2025

---

## Overview

The Inference Service Layer implements the canonical **Olumi Error Response Schema v1.0** for all error responses. This ensures consistent error handling across all Olumi services (CEE, PLoT, ISL, BFF, UI).

All ISL errors follow a flat, predictable structure with platform-required fields and ISL-specific domain fields.

---

## Error Response Structure

```typescript
interface OlumiErrorV1 {
  // Domain-specific fields (ISL defines these)
  code: string;           // e.g., 'ISL_INVALID_DAG', 'ISL_TIMEOUT'
  message: string;        // Human-readable error message
  reason?: string;        // Fine-grained reason code

  // Service-specific details (optional)
  recovery?: {
    hints: string[];      // List of actionable hints
    suggestion: string;   // Primary suggestion
    example?: string;     // Example fix
  };

  // ISL domain-specific fields (optional)
  validation_failures?: string[];     // Validation error details
  node_count?: number;                // DAG node count (structure errors)
  edge_count?: number;                // DAG edge count (structure errors)
  missing_nodes?: string[];           // Missing nodes in equations
  attempted_methods?: string[];       // Identification methods tried

  // Platform-required fields (all services MUST include)
  retryable: boolean;     // Can client retry this request?
  source: string;         // 'isl' for all ISL errors
  request_id: string;     // From X-Request-Id header
  degraded?: boolean;     // Result is partial/incomplete (default: false)
}
```

---

## Platform-Required Fields

All ISL error responses **MUST** include these fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `retryable` | `boolean` | Whether the client can retry the request | `true` or `false` |
| `source` | `string` | Always `"isl"` for ISL errors | `"isl"` |
| `request_id` | `string` | Correlation ID from `X-Request-Id` or `X-Trace-Id` header | `"req_abc123def456"` |
| `degraded` | `boolean?` | Whether the result is partial/incomplete | `false` (default) |

---

## ISL Error Codes

All ISL error codes use the `ISL_` prefix for clarity in multi-service debugging.

### DAG Structure Errors

| Code | Reason | Retryable | Description |
|------|--------|-----------|-------------|
| `ISL_INVALID_DAG` | `invalid_structure` | No | DAG has structural issues |
| `ISL_DAG_CYCLIC` | `cycle_detected` | No | DAG contains cycles |
| `ISL_DAG_EMPTY` | `empty_graph` | No | DAG has no nodes |
| `ISL_DAG_DISCONNECTED` | `disconnected_components` | No | DAG has disconnected components |
| `ISL_NODE_NOT_FOUND` | `not_found` | No | Referenced node doesn't exist |

### Model Errors

| Code | Reason | Retryable | Description |
|------|--------|-----------|-------------|
| `ISL_INVALID_MODEL` | `invalid_structure` | No | Structural model is invalid |
| `ISL_INVALID_EQUATION` | `syntax_error` | No | Equation syntax is invalid |
| `ISL_INVALID_DISTRIBUTION` | `unknown_distribution` | No | Distribution type not recognized |
| `ISL_MISSING_VARIABLE` | `undefined_variable` | No | Variable referenced but not defined |

### Validation Errors

| Code | Reason | Retryable | Description |
|------|--------|-----------|-------------|
| `ISL_VALIDATION_ERROR` | `invalid_schema` | No | Request validation failed |
| `ISL_CAUSAL_NOT_IDENTIFIABLE` | `unmeasured_confounding` | No | Causal effect cannot be identified |
| `ISL_NO_ADJUSTMENT_SET` | `no_valid_adjustment` | No | No valid adjustment set found |
| `ISL_UNMEASURED_CONFOUNDING` | `confounding_present` | No | Unmeasured confounding detected |

### Computation Errors

| Code | Reason | Retryable | Description |
|------|--------|-----------|-------------|
| `ISL_COMPUTATION_ERROR` | `internal_error` | Yes | Generic computation error |
| `ISL_Y0_ERROR` | `y0_library_error` | Yes | Y₀ library error |
| `ISL_FACET_ERROR` | `facet_computation_error` | Yes | FACET algorithm error |
| `ISL_MONTE_CARLO_ERROR` | `simulation_error` | Yes | Monte Carlo simulation error |
| `ISL_CONVERGENCE_ERROR` | `failed_to_converge` | Yes | Algorithm failed to converge |

### Input Errors

| Code | Reason | Retryable | Description |
|------|--------|-----------|-------------|
| `ISL_INVALID_INPUT` | `bad_request` | No | Invalid input parameters |
| `ISL_INVALID_PROBABILITY` | `out_of_range` | No | Probability not in [0, 1] |
| `ISL_INVALID_CONFIDENCE_LEVEL` | `out_of_range` | No | Confidence level not in [0, 1] |
| `ISL_BATCH_SIZE_EXCEEDED` | `too_many_items` | No | Batch size exceeds limit |

### Resource Errors

| Code | Reason | Retryable | Description |
|------|--------|-----------|-------------|
| `ISL_TIMEOUT` | `computation_timeout` | Yes | Computation exceeded time budget |
| `ISL_MEMORY_LIMIT` | `memory_exceeded` | No | Memory limit exceeded |
| `ISL_RATE_LIMIT_EXCEEDED` | `too_many_requests` | Yes | Rate limit exceeded |
| `ISL_SERVICE_UNAVAILABLE` | `circuit_breaker_open` | Yes | Service temporarily unavailable |

### Cache Errors

| Code | Reason | Retryable | Description |
|------|--------|-----------|-------------|
| `ISL_CACHE_ERROR` | `cache_operation_failed` | Yes | Cache operation failed |
| `ISL_REDIS_ERROR` | `redis_connection_error` | Yes | Redis connection error |

---

## Error Response Examples

### Example 1: DAG Cyclic Error

```json
{
  "code": "ISL_DAG_CYCLIC",
  "reason": "cycle_detected",
  "message": "DAG contains cycles and cannot be processed",
  "recovery": {
    "hints": [
      "Check for circular dependencies in your model",
      "Use a DAG visualization tool to identify cycles"
    ],
    "suggestion": "Remove edges that create cycles",
    "example": "If A→B→C→A exists, remove one edge"
  },
  "validation_failures": [
    "Cycle: Price → Revenue → CustomerSatisfaction → Price"
  ],
  "node_count": 5,
  "edge_count": 6,
  "retryable": false,
  "source": "isl",
  "request_id": "req_abc123def456",
  "degraded": false
}
```

### Example 2: Causal Not Identifiable

```json
{
  "code": "ISL_CAUSAL_NOT_IDENTIFIABLE",
  "reason": "unmeasured_confounding",
  "message": "Causal effect cannot be identified",
  "recovery": {
    "hints": [
      "Add measured confounders to the model",
      "Consider using instrumental variables",
      "Explore front-door criterion"
    ],
    "suggestion": "Measure and include confounding variables"
  },
  "attempted_methods": ["backdoor", "front_door", "do_calculus"],
  "retryable": false,
  "source": "isl",
  "request_id": "req_xyz789",
  "degraded": false
}
```

### Example 3: Timeout Error

```json
{
  "code": "ISL_TIMEOUT",
  "message": "Computation exceeded time budget",
  "recovery": {
    "hints": [
      "Simplify your causal model",
      "Reduce Monte Carlo iterations"
    ],
    "suggestion": "Retry with a simpler model"
  },
  "retryable": true,
  "source": "isl",
  "request_id": "req_timeout123",
  "degraded": false
}
```

### Example 4: Validation Error (Request Schema)

```json
{
  "code": "ISL_VALIDATION_ERROR",
  "message": "Request validation failed",
  "reason": "invalid_schema",
  "recovery": {
    "hints": [
      "Ensure all required fields are provided",
      "Check data types match the expected schema"
    ],
    "suggestion": "Fix validation errors and retry",
    "example": "See validation_failures field for specific issues"
  },
  "validation_failures": [
    "body.nodes: field required",
    "body.edges: value is not a valid list"
  ],
  "retryable": false,
  "source": "isl",
  "request_id": "req_val456",
  "degraded": false
}
```

### Example 5: Rate Limit Exceeded

```json
{
  "code": "ISL_RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Please wait 30 seconds before retrying.",
  "reason": "too_many_requests",
  "recovery": {
    "hints": [
      "Wait 30 seconds before retrying",
      "Reduce request frequency",
      "Consider implementing client-side rate limiting"
    ],
    "suggestion": "Retry after 30 seconds"
  },
  "retryable": true,
  "source": "isl",
  "request_id": "req_rate789",
  "degraded": false
}
```

### Example 6: Service Unavailable (Circuit Breaker)

```json
{
  "code": "ISL_SERVICE_UNAVAILABLE",
  "message": "Service temporarily unavailable due to high memory usage (87.3%)",
  "reason": "memory_circuit_breaker_open",
  "recovery": {
    "hints": [
      "Wait 30 seconds before retrying",
      "Simplify your request to reduce memory usage",
      "Consider reducing batch sizes or complexity"
    ],
    "suggestion": "Retry after 30 seconds when memory usage decreases"
  },
  "retryable": true,
  "source": "isl",
  "request_id": "req_circuit123",
  "degraded": false
}
```

---

## Implementation Details

### Request ID Extraction

ISL extracts the `request_id` from headers in the following order:

1. `X-Request-Id` header (platform standard)
2. `X-Trace-Id` header (ISL-specific)
3. Generated ID using `req_{uuid16}` format

**Example:**
```python
request_id = (
    request.headers.get("X-Request-Id") or
    request.headers.get("X-Trace-Id") or
    get_trace_id()
)
```

### Error Response Middleware

ISL implements error handling at multiple layers:

1. **Exception Handlers** (`src/api/main.py`)
   - `HTTPException` handler - Maps HTTP exceptions to ISL errors
   - `RequestValidationError` handler - Handles Pydantic validation errors
   - `Exception` handler - Catches all unhandled exceptions

2. **Rate Limiting Middleware** (`src/middleware/rate_limiting.py`)
   - Returns `ISL_RATE_LIMIT_EXCEEDED` when rate limit hit
   - Includes `Retry-After` header

3. **Circuit Breaker Middleware** (`src/middleware/circuit_breaker.py`)
   - `MemoryCircuitBreaker` - Returns `ISL_SERVICE_UNAVAILABLE` when memory > 85%
   - `HealthCircuitBreaker` - Returns `ISL_SERVICE_UNAVAILABLE` when unhealthy

### Pydantic Model

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class RecoveryHints(BaseModel):
    """Recovery hints for error resolution."""
    hints: List[str]
    suggestion: str
    example: Optional[str] = None

class ErrorResponse(BaseModel):
    """Olumi Error Response (v1.0) - ISL Implementation."""

    # Domain-specific fields
    code: str
    message: str
    reason: Optional[str] = None
    recovery: Optional[RecoveryHints] = None

    # ISL domain-specific fields
    validation_failures: Optional[List[str]] = None
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    missing_nodes: Optional[List[str]] = None
    attempted_methods: Optional[List[str]] = None

    # Platform-required fields
    retryable: bool
    source: str = "isl"
    request_id: str
    degraded: Optional[bool] = False
```

---

## Client Integration

### TypeScript

```typescript
interface OlumiErrorV1 {
  code: string;
  message: string;
  reason?: string;
  recovery?: {
    hints: string[];
    suggestion: string;
    example?: string;
  };
  validation_failures?: string[];
  node_count?: number;
  edge_count?: number;
  missing_nodes?: string[];
  attempted_methods?: string[];
  retryable: boolean;
  source: string;
  request_id: string;
  degraded?: boolean;
}

// Error handling example
async function callISL() {
  try {
    const response = await fetch('https://isl.olumi.com/api/v1/causal/validate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-Id': generateRequestId(),
      },
      body: JSON.stringify({ /* ... */ }),
    });

    if (!response.ok) {
      const error: OlumiErrorV1 = await response.json();

      console.error(`ISL Error [${error.code}]: ${error.message}`);
      console.log(`Request ID: ${error.request_id}`);

      if (error.retryable) {
        console.log('Error is retryable');
        if (error.recovery) {
          console.log(`Suggestion: ${error.recovery.suggestion}`);
        }
      } else {
        console.log('Error is not retryable');
        if (error.recovery?.hints) {
          console.log('Recovery hints:', error.recovery.hints);
        }
      }

      throw new ISLError(error);
    }

    return await response.json();
  } catch (err) {
    // Handle network errors, etc.
  }
}
```

### Python

```python
import requests
from typing import Optional
from dataclasses import dataclass

@dataclass
class OlumiErrorV1:
    code: str
    message: str
    retryable: bool
    source: str
    request_id: str
    reason: Optional[str] = None
    recovery: Optional[dict] = None
    validation_failures: Optional[list] = None
    degraded: bool = False

def call_isl():
    response = requests.post(
        'https://isl.olumi.com/api/v1/causal/validate',
        json={'nodes': [...], 'edges': [...]},
        headers={'X-Request-Id': generate_request_id()}
    )

    if not response.ok:
        error_data = response.json()
        error = OlumiErrorV1(**error_data)

        print(f"ISL Error [{error.code}]: {error.message}")
        print(f"Request ID: {error.request_id}")

        if error.retryable:
            print("Error is retryable")
            if error.recovery:
                print(f"Suggestion: {error.recovery['suggestion']}")
        else:
            print("Error is not retryable")

        raise ISLError(error)

    return response.json()
```

---

## Error Handling Best Practices

### 1. Always Use Request IDs

Include `X-Request-Id` header in all requests for end-to-end tracing:

```typescript
headers: {
  'X-Request-Id': `req_${uuid()}`,
}
```

### 2. Respect Retryable Flag

Only retry errors where `retryable: true`:

```typescript
if (error.retryable) {
  // Implement exponential backoff
  await sleep(retryAfter * 1000);
  return retry();
} else {
  // Show user-facing error message
  showError(error.message, error.recovery?.hints);
}
```

### 3. Use Recovery Hints

Display recovery hints to users or developers:

```typescript
if (error.recovery) {
  console.log('How to fix:');
  error.recovery.hints.forEach(hint => console.log(`- ${hint}`));
  console.log(`\nSuggestion: ${error.recovery.suggestion}`);
}
```

### 4. Log Request IDs

Always log the `request_id` for debugging:

```typescript
logger.error('ISL request failed', {
  requestId: error.request_id,
  errorCode: error.code,
  source: error.source,
});
```

### 5. Handle Degraded Responses

Check the `degraded` flag to determine if results are partial:

```typescript
const result = await response.json();
if (result.degraded) {
  console.warn('Results are partial/incomplete');
  // Show warning to user
}
```

---

## Migration from Old Schema

### Old Schema (Deprecated)

```json
{
  "error_code": "invalid_dag_structure",
  "message": "DAG contains cycles",
  "details": {...},
  "trace_id": "550e8400-e29b...",
  "retryable": false,
  "suggested_action": "fix_input"
}
```

### New Schema (v1.0)

```json
{
  "code": "ISL_DAG_CYCLIC",
  "reason": "cycle_detected",
  "message": "DAG contains cycles",
  "recovery": {
    "hints": [...],
    "suggestion": "Remove edges that create cycles"
  },
  "validation_failures": [...],
  "retryable": false,
  "source": "isl",
  "request_id": "req_abc123def456"
}
```

### Key Changes

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `error_code` | `code` | Now uses `ISL_` prefix |
| `trace_id` | `request_id` | Renamed for platform consistency |
| `suggested_action` | `recovery.suggestion` | Now nested with hints |
| `details` | (domain fields) | Flattened to top level |
| (none) | `source` | New required field |
| (none) | `reason` | New optional field |

---

## Testing

### Example Test Cases

```python
def test_error_response_schema():
    """Test that error responses match Olumi Error Schema v1.0."""
    response = client.post(
        "/api/v1/causal/validate",
        json={"nodes": [], "edges": []}  # Invalid: empty DAG
    )

    assert response.status_code == 400
    error = response.json()

    # Required platform fields
    assert "code" in error
    assert "message" in error
    assert "retryable" in error
    assert "source" in error
    assert "request_id" in error
    assert error["source"] == "isl"

    # ISL-specific fields
    assert error["code"].startswith("ISL_")

    # Recovery hints
    if "recovery" in error:
        assert "hints" in error["recovery"]
        assert "suggestion" in error["recovery"]
        assert isinstance(error["recovery"]["hints"], list)
```

---

## Support

For questions about the error schema or implementation:

- **Platform Coordination**: Raise in platform coordination channel
- **ISL Team**: Open an issue in the ISL repository
- **Documentation**: This document and API documentation at `/docs`

---

**Version:** 1.0
**Last Updated:** November 2025
**Status:** ✅ Implemented
