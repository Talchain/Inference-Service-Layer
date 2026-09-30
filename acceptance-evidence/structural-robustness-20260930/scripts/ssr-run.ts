/**
 * SCI-STRUCTURAL-ROBUSTNESS (30 Sep 2026) — scratch runner, NOT for merge into PLoT.
 *
 * Posts a captured CEE -> PLoT /v2/run payload to PLoT's REAL /v2/run route (in-process, app.inject), with PLoT's REAL
 * ISL client pointed at a LOCAL ISL (uvicorn). Every ISL request/response PLoT makes is captured from global fetch.
 *
 * Usage: npx tsx ssr-run.ts <payload.json> <out-prefix> [seed|-] [n_samples|-] [extra-options.json|-]
 */
import { readFileSync, writeFileSync } from 'node:fs';

process.env.ISL_ENABLE = '1';
process.env.ISL_BASE_URL = process.env.ISL_BASE_URL ?? 'http://127.0.0.1:8931';
process.env.ISL_API_KEY = process.env.ISL_API_KEY ?? 'local-no-auth'; // local ISL runs with ISL_AUTH_DISABLED=true
process.env.ISL_TIMEOUT_MS = process.env.ISL_TIMEOUT_MS ?? '180000';
process.env.ISL_MAX_RETRIES = process.env.ISL_MAX_RETRIES ?? '0';
process.env.AUTH_ENABLED = '0';

const [payloadPath, outPrefix, seedArg, nArg, extraOptsPath] = process.argv.slice(2);
const payload = JSON.parse(readFileSync(payloadPath!, 'utf8')) as Record<string, any>;
if (seedArg && seedArg !== '-') payload.seed = /^\d+$/.test(seedArg) ? Number(seedArg) : seedArg;
if (nArg && nArg !== '-') payload.n_samples = Number(nArg);
if (extraOptsPath && extraOptsPath !== '-') payload.options = [...payload.options, ...JSON.parse(readFileSync(extraOptsPath, 'utf8'))];

const captures: Array<{ url: string; status: number; request: unknown; response: unknown; ms: number }> = [];
const realFetch = globalThis.fetch;
globalThis.fetch = (async (input: any, init?: any) => {
  const url = typeof input === 'string' ? input : input?.url ?? String(input);
  const t0 = Date.now();
  const res = await realFetch(input, init);
  if (url.startsWith(process.env.ISL_BASE_URL!)) {
    const text = await res.clone().text();
    let body: unknown = text;
    try { body = JSON.parse(text); } catch { /* keep text */ }
    let req: unknown = init?.body;
    try { req = typeof init?.body === 'string' ? JSON.parse(init.body) : init?.body; } catch { /* keep raw */ }
    captures.push({ url, status: res.status, request: req, response: body, ms: Date.now() - t0 });
  }
  return res;
}) as typeof fetch;

const { createServer } = await import('./src/createServer.js');
const app = await createServer();
await app.ready();
const t0 = Date.now();
const res = await app.inject({ method: 'POST', url: '/v2/run', payload, headers: { 'content-type': 'application/json' } });
const ms = Date.now() - t0;
writeFileSync(`${outPrefix}.plot-request.json`, JSON.stringify(payload, null, 2));
writeFileSync(`${outPrefix}.plot-response.json`, JSON.stringify({ status: res.statusCode, ms, body: JSON.parse(res.body) }, null, 2));
writeFileSync(`${outPrefix}.isl-calls.json`, JSON.stringify(captures, null, 2));
console.log(JSON.stringify({ status: res.statusCode, ms, isl_calls: captures.map((c) => ({ url: c.url.replace(process.env.ISL_BASE_URL!, ''), status: c.status, ms: c.ms })) }));
await app.close();
process.exit(0);
