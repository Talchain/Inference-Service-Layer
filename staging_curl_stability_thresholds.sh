#!/bin/bash
#
# Staging curl command to verify stability_thresholds appears in ISLResponseV2 envelope
#
# This script demonstrates the fix for the bug where stability_thresholds was
# present in RobustnessResponseV2 but stripped from the ISLResponseV2 envelope.
#
# Expected behavior:
# - When parameter_uncertainties is provided, bootstrap analysis runs
# - The response should include top-level "stability_thresholds" field with:
#   - high_moderate_boundary (float)
#   - moderate_low_boundary (float)
#   - version (string)
#   - provisional (bool, always true)
#
# Usage:
#   # Without auth (if ISL_AUTH_DISABLED=true on staging):
#   ./staging_curl_stability_thresholds.sh
#
#   # With auth (if staging requires API key):
#   ISL_API_KEY=your-key ./staging_curl_stability_thresholds.sh
#
#   # To extract just the stability_thresholds field:
#   ./staging_curl_stability_thresholds.sh | jq '.stability_thresholds'
#

set -euo pipefail

STAGING_URL="https://isl-staging.onrender.com"
API_KEY="${ISL_API_KEY:-}"

# Build auth header if API key is provided
AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
  AUTH_HEADER="-H \"X-API-Key: $API_KEY\""
fi

# Make request with status code capture
HTTP_STATUS=$(curl -X POST "${STAGING_URL}/api/v1/robustness/analyze/v2" \
  -H "Content-Type: application/json" \
  -H "X-ISL-Response-Version: 2" \
  -H "X-Request-Id: stability-thresholds-verify-$(date +%s)" \
  ${AUTH_HEADER} \
  -w "%{http_code}" \
  -o /tmp/staging_response.json \
  -s \
  -d '{
    "graph": {
      "nodes": [
        {"id": "price", "kind": "factor", "label": "Price"},
        {"id": "marketing", "kind": "factor", "label": "Marketing Spend"},
        {"id": "sales", "kind": "outcome", "label": "Sales"}
      ],
      "edges": [
        {
          "from": "price",
          "to": "sales",
          "exists_probability": 0.95,
          "strength": {"mean": -0.6, "std": 0.15}
        },
        {
          "from": "marketing",
          "to": "sales",
          "exists_probability": 0.9,
          "strength": {"mean": 0.7, "std": 0.12}
        }
      ]
    },
    "options": [
      {"id": "opt1", "label": "Raise price, increase marketing", "interventions": {"price": 0.8, "marketing": 0.7}},
      {"id": "opt2", "label": "Lower price, moderate marketing", "interventions": {"price": 0.3, "marketing": 0.5}}
    ],
    "goal_node_id": "sales",
    "seed": 42,
    "n_samples": 500,
    "parameter_uncertainties": [
      {"node_id": "price", "distribution": "normal", "std": 0.1},
      {"node_id": "marketing", "distribution": "normal", "std": 0.15}
    ]
  }')

# Check HTTP status
if [ "$HTTP_STATUS" -ne 200 ]; then
  echo "❌ Request failed with HTTP $HTTP_STATUS" >&2
  echo "" >&2
  cat /tmp/staging_response.json >&2
  echo "" >&2
  exit 1
fi

# Display response
cat /tmp/staging_response.json

# Clean up
rm -f /tmp/staging_response.json

echo "" >&2
echo "" >&2
echo "✅ Request successful (HTTP 200)" >&2
echo "" >&2
echo "Expected stability_thresholds field (default config):" >&2
echo '  {' >&2
echo '    "high_moderate_boundary": 0.1,' >&2
echo '    "moderate_low_boundary": 0.3,' >&2
echo '    "version": "v1.0-operational-defaults",' >&2
echo '    "provisional": true' >&2
echo '  }' >&2
echo "" >&2
echo "Note: These values may differ if staging has env overrides set:" >&2
echo "  STABILITY_CV_HIGH_MODERATE, STABILITY_CV_MODERATE_LOW" >&2
echo "" >&2
