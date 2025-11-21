#!/bin/bash
# Quick smoke test for ISL deployment validation
# Usage: ./quick_check.sh <base_url> <api_key>

set -e

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-isl_prod_7k9mP2nX8vQ4rL6wF3jH5tY1cB0zS}"

echo "🔍 ISL Quick Smoke Test"
echo "========================"
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health Check
echo "✓ Testing health endpoint..."
HEALTH=$(curl -s "$BASE_URL/health")
STATUS=$(echo $HEALTH | jq -r '.status' 2>/dev/null || echo "error")

if [ "$STATUS" = "healthy" ]; then
    VERSION=$(echo $HEALTH | jq -r '.version')
    echo "  ✅ Health check passed (version: $VERSION)"
else
    echo "  ❌ Health check failed"
    exit 1
fi

# Test 2: Cache Stats
echo "✓ Testing cache stats endpoint..."
CACHE=$(curl -s "$BASE_URL/cache/stats")
HIT_RATE=$(echo $CACHE | jq -r '.hit_rate_percent' 2>/dev/null || echo "error")

if [ "$HIT_RATE" != "error" ]; then
    echo "  ✅ Cache stats accessible (hit rate: $HIT_RATE%)"
else
    echo "  ❌ Cache stats failed"
    exit 1
fi

# Test 3: Causal Validation
echo "✓ Testing causal validation..."
VALIDATE=$(curl -s -X POST "$BASE_URL/api/v1/causal/validate" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{
        "dag": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
        "treatment": "X",
        "outcome": "Y"
    }')

CAUSAL_STATUS=$(echo $VALIDATE | jq -r '.status' 2>/dev/null || echo "error")

if [ "$CAUSAL_STATUS" = "identifiable" ]; then
    echo "  ✅ Causal validation working"
else
    echo "  ❌ Causal validation failed"
    echo "  Response: $VALIDATE"
    exit 1
fi

# Test 4: Counterfactual Generation
echo "✓ Testing counterfactual generation..."
CF=$(curl -s -X POST "$BASE_URL/api/v1/causal/counterfactual" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{
        "causal_model": {
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
            "structural_equations": {"Y": "2*X + 5"}
        },
        "intervention": {"X": 10},
        "outcome_variables": ["Y"],
        "samples": 100
    }')

CF_ESTIMATE=$(echo $CF | jq -r '.prediction.point_estimate' 2>/dev/null || echo "error")

if [ "$CF_ESTIMATE" != "error" ] && [ "$CF_ESTIMATE" != "null" ]; then
    echo "  ✅ Counterfactual generation working (estimate: $CF_ESTIMATE)"
else
    echo "  ❌ Counterfactual generation failed"
    echo "  Response: $CF"
    exit 1
fi

# Test 5: Robustness Analysis
echo "✓ Testing robustness analysis..."
ROBUST=$(curl -s -X POST "$BASE_URL/api/v1/robustness/analyze" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{
        "causal_model": {
            "nodes": ["price", "demand", "revenue"],
            "edges": [["price", "demand"], ["demand", "revenue"]]
        },
        "intervention_proposal": {"price": 55},
        "target_outcome": {"revenue": [95000, 105000]},
        "perturbation_radius": 0.1
    }')

ROBUST_STATUS=$(echo $ROBUST | jq -r '.analysis.status' 2>/dev/null || echo "error")

if [ "$ROBUST_STATUS" != "error" ] && [ "$ROBUST_STATUS" != "null" ]; then
    ROBUST_SCORE=$(echo $ROBUST | jq -r '.analysis.robustness_score')
    echo "  ✅ Robustness analysis working (score: $ROBUST_SCORE)"
else
    echo "  ❌ Robustness analysis failed"
    echo "  Response: $ROBUST"
    exit 1
fi

echo ""
echo "✅ All smoke tests passed!"
echo "========================"
echo "ISL is ready for production"
