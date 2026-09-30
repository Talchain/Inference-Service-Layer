/**
 * SCI-STRUCTURAL-ROBUSTNESS (30 Sep 2026) — scratch harness, NOT for merge into CEE.
 *
 * Builds Models A, B, C from the served m0 graph and captures the EXACT PLoT /v2/run payload CEE's REAL run_analysis
 * handler assembles for each (same loader, same wire transforms). No LLM, no network: the PLoT client is a capture mock
 * that returns the served m0 body so the handler completes.
 *
 *   A = served m0 graph, byte-for-byte (fixture served-gate5-mrr-m0-and-cut-costs-15f48f0b.json).
 *   B = A after Olumi's OWN product card is accepted: proposeProductIdentity(A) -> applyIdentityConfirmEdit (the canonical
 *       writer the card's Yes calls). Nothing is hand-edited.
 *   C = A with the inferred "Non-Pro MRR" node and its one edge into MRR removed (MRR read as Pro-plan MRR only; the
 *       £1,500 gap left unexplained). Hand-edited deletion only; no value added or changed.
 */
import { describe, expect, it, vi } from 'vitest';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

import type { PLoTClient } from '../../../../orchestrator/plot-client.js';
import type { V2RunResponseEnvelope } from '../../../../orchestrator/types.js';
import { loadScenarioSnapshotForRunAnalysis } from '../../../build-turn-context.js';
import type { HandlerInvocation } from '../../registry.js';
import { createRunAnalysisHandler } from '../run-analysis.js';
import { makeMessagePayload } from '../../../__tests__/fixtures.js';
import { proposeProductIdentity } from '../../../agent-lane/identity-proposal.js';
import { applyIdentityConfirmEdit, identityConfirmReadingToken } from '../../../system-events/identity-confirm-edit.js';
import { computeAnalysisAffectingGraphHash } from '../../../context/graph-hash.js';
import { unreadGoalProduct } from '../../../agent-lane/unread-goal-product.js';

type Json = Record<string, any>;
const OUT = process.env.SSR_OUT ?? '/private/tmp/ssr-20260930/out';
mkdirSync(OUT, { recursive: true });
const F = JSON.parse(readFileSync(new URL('./fixtures/served-gate5-mrr-m0-and-cut-costs-15f48f0b.json', import.meta.url), 'utf8')) as Json;
const M0 = F.mrr_m0 as { _provenance: Json; graph: Json; plot_body: Json };
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;
const SCENARIO = 'c8108752-0000-4000-8000-00000000055a';

async function capture(graph: Json, tag: string): Promise<Json> {
  const store = {
    loadGraphAndBriefText: vi.fn(async () => ({ graph: clone(graph), briefText: M0._provenance.brief_text ?? '' })),
    loadGraph: vi.fn(async () => clone(graph)),
  };
  const snapshot = await loadScenarioSnapshotForRunAnalysis(SCENARIO, `req-ssr-load-${tag}`, store as never);
  let payload: Json | undefined;
  const run = vi.fn(async (p: Json) => {
    payload = clone(p);
    return clone(M0.plot_body) as unknown as V2RunResponseEnvelope;
  });
  const handler = createRunAnalysisHandler({
    plotClient: { run, validatePatch: vi.fn().mockResolvedValue({}) } as unknown as PLoTClient,
    scenarioReader: vi.fn(async () => snapshot),
  });
  await handler({
    context: {
      stage: 'analyse', entity_registry: { option_ids: [], goal_id: null }, capabilities: {}, messages: [],
      session_id: SCENARIO, request_id: `req-ssr-${tag}`, budgets: { turn_ms: 180_000, llm_narrate_ms: 60_000 },
      prior_turns: [], prior_facts: [], scenarioBriefText: null, persistedGraph: null,
    },
    payload: makeMessagePayload({
      turn_id: `t-ssr-${tag}`, scenario_id: SCENARIO, message: 'Run the analysis.', turn_class: 'decide', stage: 'analyse',
    } as never),
    requestId: `req-ssr-${tag}`, signal: new AbortController().signal, orientationText: '',
  } as unknown as HandlerInvocation);
  expect(run).toHaveBeenCalledTimes(1);
  return payload!;
}

describe('SSR: build A/B/C and capture CEE -> PLoT payloads', () => {
  it('A, B (card Yes via the canonical writer), C (inferred Non-Pro MRR removed)', async () => {
    const A = clone(M0.graph);

    // ---- B: the product reading Olumi's OWN Gate 5 detector names on A (unreadGoalProduct), written by the canonical
    //      card-Yes writer (applyIdentityConfirmEdit) where it admits it; else the writer's exact carrier shape by hand ----
    const card = proposeProductIdentity(A);
    const unread = unreadGoalProduct(A);
    writeFileSync(`${OUT}/B-detectors.json`, JSON.stringify({ proposeProductIdentity_A: card, unreadGoalProduct_A: unread }, null, 2));
    expect(unread).not.toBeNull();
    const factor_ids = [unread!.rate.id, unread!.count.id];
    const words = `Is “${unread!.goal.label}” “${unread!.rate.label}” × “${unread!.count.label}”?`;
    const edit = applyIdentityConfirmEdit({
      outcome_id: unread!.goal.id,
      factor_ids,
      words,
      persistedGraph: A,
      expected_graph_hash: computeAnalysisAffectingGraphHash(A as never),
      reading_token: identityConfirmReadingToken({ outcome_id: unread!.goal.id, factor_ids, words }),
    });
    writeFileSync(`${OUT}/B-edit-result.json`, JSON.stringify({ kind: edit.kind, reason: (edit as Json).reason, detail: (edit as Json).detail }, null, 2));
    let B: Json;
    if (edit.kind === 'mutated') {
      B = edit.mutatedGraph as Json;
    } else {
      // The writer refused on this graph: write ONLY its carrier, in its exact shape, on the goal node. Nothing else moves.
      B = clone(A);
      const goal = B.nodes.find((n: Json) => n.id === unread!.goal.id);
      goal.nonlinear_identity = { operation: 'product', factor_ids, stated_in_brief: true };
    }

    // ---- C: delete the inferred residual node and its one edge; nothing else ----
    const C = clone(A);
    C.nodes = C.nodes.filter((n: Json) => n.id !== 'non_pro_mrr');
    C.edges = C.edges.filter((e: Json) => e.from !== 'non_pro_mrr' && e.to !== 'non_pro_mrr');
    expect(C.nodes.length).toBe(A.nodes.length - 1);
    expect(C.edges.length).toBe(A.edges.length - 1);
    writeFileSync(`${OUT}/C-proposal.json`, JSON.stringify(proposeProductIdentity(C), null, 2));

    for (const [tag, g] of [['A', A], ['B', B], ['C', C]] as const) {
      writeFileSync(`${OUT}/${tag}-graph.json`, JSON.stringify(g, null, 2));
      const p = await capture(g, tag);
      writeFileSync(`${OUT}/${tag}-plot-payload.json`, JSON.stringify(p, null, 2));
    }
  });
});

describe('SSR: product surface — CEE run_analysis handler over the LOCAL PLoT body for each model (seed 1254899477, cur)', () => {
  it('records leader, goal figures and typed warnings CEE would return', async () => {
    const out: Record<string, unknown> = {};
    for (const tag of ['A', 'B', 'C'] as const) {
      const graph = JSON.parse(readFileSync(`${OUT}/${tag}-graph.json`, 'utf8')) as Json;
      const body = (JSON.parse(readFileSync(`${OUT}/runs/${tag}-cur-s1254899477.plot-response.json`, 'utf8')) as Json).body;
      const store = {
        loadGraphAndBriefText: vi.fn(async () => ({ graph: clone(graph), briefText: M0._provenance.brief_text ?? '' })),
        loadGraph: vi.fn(async () => clone(graph)),
      };
      const snapshot = await loadScenarioSnapshotForRunAnalysis(SCENARIO, `req-ssr-ps-${tag}`, store as never);
      const run = vi.fn(async () => clone(body) as unknown as V2RunResponseEnvelope);
      const handler = createRunAnalysisHandler({
        plotClient: { run, validatePatch: vi.fn().mockResolvedValue({}) } as unknown as PLoTClient,
        scenarioReader: vi.fn(async () => snapshot),
      });
      const outcome = await handler({
        context: {
          stage: 'analyse', entity_registry: { option_ids: [], goal_id: null }, capabilities: {}, messages: [],
          session_id: SCENARIO, request_id: `req-ssr-ps-${tag}`, budgets: { turn_ms: 180_000, llm_narrate_ms: 60_000 },
          prior_turns: [], prior_facts: [], scenarioBriefText: null, persistedGraph: null,
        },
        payload: makeMessagePayload({
          turn_id: `t-ssr-ps-${tag}`, scenario_id: SCENARIO, message: 'Run the analysis.', turn_class: 'decide', stage: 'analyse',
        } as never),
        requestId: `req-ssr-ps-${tag}`, signal: new AbortController().signal, orientationText: '',
      } as unknown as HandlerInvocation);
      const r = (outcome.handler_facts[0] as Json).result as Json;
      const env = (r.enrichment ?? r) as Json;
      const oc = (env.option_comparison ?? []) as Json[];
      out[tag] = {
        leading_option_id: r.leading_option_id ?? null,
        goal_figures: Object.fromEntries(oc.map((o) => [o.option_id ?? o.id, {
          probability_of_goal: o.probability_of_goal ?? null, win_probability: o.win_probability ?? null, mean: o.outcome?.mean ?? null,
        }])),
        warning_codes: ((env.inference_warnings ?? []) as Json[]).map((w) => w.code).concat(((r.warnings ?? []) as Json[]).map((w) => w.code ?? w)),
      };
    }
    writeFileSync(`${OUT}/product-surface-cee.json`, JSON.stringify(out, null, 2));
  });
});
