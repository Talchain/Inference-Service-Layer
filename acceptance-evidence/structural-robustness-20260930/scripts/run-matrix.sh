#!/bin/bash
# SCI-STRUCTURAL-ROBUSTNESS 30 Sep 2026: run A/B/C x option sets x seeds through local PLoT (in-process) -> local ISL.
# Prereq: local ISL on 127.0.0.1:8931 (ISL_AUTH_DISABLED=true), payloads captured by zz-ssr-capture.test.ts.
set -u -o pipefail
OUT=/private/tmp/ssr-20260930/out
PLOT=/private/tmp/ssr-20260930/plot-lite-service
export TOKEN_HMAC_SECRET="${TOKEN_HMAC_SECRET:-$(openssl rand -hex 32)}"   # throwaway local boot secret, never a credential
N=10000
cd "$PLOT" || exit 2
for seed in 1254899477 1 20260930; do
  for model in A B C; do
    for optset in cur served3; do
      extra=-; [ "$optset" = served3 ] && extra="$OUT/served-era-extra-option.json"
      tag="$model-$optset-s$seed"
      perl -e 'alarm shift; exec @ARGV' 600 npx tsx ssr-run.ts "$OUT/$model-plot-payload.json" "$OUT/runs/$tag" "$seed" "$N" "$extra" \
        > "$OUT/runs/$tag.stdout" 2> "$OUT/runs/$tag.stderr"
      rc=$?
      st=$(jq -r '.status' "$OUT/runs/$tag.plot-response.json" 2>/dev/null)
      echo "$tag rc=$rc plot_status=$st"
    done
  done
done
