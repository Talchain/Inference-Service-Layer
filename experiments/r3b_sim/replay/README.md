# Offline served-path replay (no served Olumi endpoint is called)

Captures the exact ISL request that PLoT compiles, using local services only.

1. ISL (checkout of the branch under test): `ISL_AUTH_DISABLED=true poetry run uvicorn src.api.main:app --port 8102`
2. Capture proxy: `python3 logproxy.py <out-dir>` listens on 127.0.0.1:8103, forwards to 8102 and writes every
   POST body to `<out-dir>/isl-req-<n>-<path>.json`.
3. PLoT (plot-lite-service checkout): `TOKEN_HMAC_SECRET=<any local value> AUTH_ENABLED=0 ISL_ENABLE=1
   ISL_BASE_URL=http://127.0.0.1:8103 npx tsx src/main.ts`, then POST `eh4-plot-request.json` to its run endpoint.

`../evppi_gate/adversarial/eh4-isl-request.json` is the eng-hiring-4 request captured this way (28 Sep).
PLoT's own request log redacts payloads (`sha8:`), which is why the proxy is needed.
