/**
 * SCI-STRUCTURAL-ROBUSTNESS, 2nd run (30 Sep 2026) — scratch harness, NOT for merge into CEE. Adapted from the first
 * run's `zz-ssr-capture.test.ts` (ISL `6f7f9b43`). Run it as an untracked file at
 * `src/orchestrator-v5/tools/handlers/__tests__/zz-ssr2-capture.test.ts` in a CEE checkout, then delete it.
 *
 * Builds the models from R3's served m2 draft of Paul's unchanged MRR brief (`r3/science-notes` @ fb3a396c,
 * `row-b-2339-ef042ce/drafts/m2.json`, served CEE `ef042ce`) and captures the EXACT PLoT /v2/run payload CEE's REAL
 * run_analysis handler assembles for each. No LLM, no network: the PLoT client is a capture mock.
 *
 *   A  = m2 after Olumi's OWN product card is accepted: proposeProductIdentity(m2) -> applyIdentityConfirmEdit (the
 *        canonical writer the card's Yes calls). Nothing is hand-edited. If the writer refuses, the test FAILS (stop).
 *   A2 = A with the subscriber node's frame re-expressed (cap 5,000 -> 20,000) and its two in-edges' strengths converted
 *        with it (natural effects unchanged). A frame-invariance CONTROL: it must give A's figures.
 *   B  = A2 with the subscriber operand read at MONTH 12 ("within a year"): Olumi's own per-month natural effects into
 *        it (-15 subscribers per +1pp monthly churn; +1 per new subscriber/month) accumulated over the brief's 12
 *        months, first order (non-compounding): -180 and +12. Same frame as A2, so B differs from A2 ONLY in those two
 *        edges' sizes and the operand's label. No level, option, node, uncertainty or other edge changes.
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

type Json = Record<string, any>;
const OUT = process.env.SSR2_OUT!;
const M2_PATH = process.env.SSR2_M2!;
const BRIEF = 'Should we raise our Pro plan price from £49 to £59 a month? We have 1,500 paying subscribers and £75k MRR. Monthly churn must stay below 5%, and we want MRR above £85k within a year.';
const SCENARIO = '3eb2e670-911d-4970-8149-842e170ab000';
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;
const HORIZON_MONTHS = 12; // the brief: "within a year"
const SUBS = 'paying_subscribers';
const NEW_FRAME = 20000;

async function capture(graph: Json, tag: string): Promise<Json> {
  const store = {
    loadGraphAndBriefText: vi.fn(async () => ({ graph: clone(graph), briefText: BRIEF })),
    loadGraph: vi.fn(async () => clone(graph)),
  };
  const snapshot = await loadScenarioSnapshotForRunAnalysis(SCENARIO, `req-ssr2-load-${tag}`, store as never);
  let payload: Json | undefined;
  const run = vi.fn(async (p: Json) => {
    payload = clone(p);
    return { status: 'ok' } as unknown as V2RunResponseEnvelope;
  });
  const handler = createRunAnalysisHandler({
    plotClient: { run, validatePatch: vi.fn().mockResolvedValue({}) } as unknown as PLoTClient,
    scenarioReader: vi.fn(async () => snapshot),
  });
  try {
    await handler({
      context: {
        stage: 'analyse', entity_registry: { option_ids: [], goal_id: null }, capabilities: {}, messages: [],
        session_id: SCENARIO, request_id: `req-ssr2-${tag}`, budgets: { turn_ms: 180_000, llm_narrate_ms: 60_000 },
        prior_turns: [], prior_facts: [], scenarioBriefText: null, persistedGraph: null,
      },
      payload: makeMessagePayload({
        turn_id: `t-ssr2-${tag}`, scenario_id: SCENARIO, message: 'Run the analysis.', turn_class: 'decide', stage: 'analyse',
      } as never),
      requestId: `req-ssr2-${tag}`, signal: new AbortController().signal, orientationText: '',
    } as unknown as HandlerInvocation);
  } catch { /* the mock body is not a real envelope; only the outbound payload matters here */ }
  expect(run).toHaveBeenCalledTimes(1);
  return payload!;
}

/** Re-express SUBS's frame as NEW_FRAME and convert its in-edges' normalised strengths with it (natural effects kept). */
function reframeSubs(g: Json, scaleNatural: number): void {
  const subs = g.nodes.find((n: Json) => n.id === SUBS);
  const oldFrame = subs.observed_state.cap as number;
  subs.observed_state.cap = NEW_FRAME;
  subs.observed_state.value = subs.observed_state.raw_value / NEW_FRAME;
  for (const e of g.edges.filter((x: Json) => x.to === SUBS)) {
    const ne = e.provenance?.natural_effect;
    expect(ne, `${e.from}->${SUBS} carries a natural effect`).toBeTruthy();
    const k = (oldFrame / NEW_FRAME) * scaleNatural;
    e.strength = { mean: e.strength.mean * k, std: e.strength.std * k };
    e.provenance.natural_effect = { ...ne, amount: ne.amount * scaleNatural, strength_mean: e.strength.mean };
  }
}

describe('SSR2: build A / A2 / B from served m2 and capture CEE -> PLoT payloads', () => {
  it('A (card Yes via the canonical writer), A2 (frame control), B (month-12 subscriber operand)', () => {
    mkdirSync(OUT, { recursive: true });
    const M2 = (JSON.parse(readFileSync(M2_PATH, 'utf8')) as Json).graph as Json;

    // ---- A: Olumi's OWN card on m2, accepted through the canonical writer. No hand edit. ----
    const card = proposeProductIdentity(M2);
    expect(card, 'the card is offered on m2').not.toBeNull();
    const edit = applyIdentityConfirmEdit({
      outcome_id: card!.outcome_id,
      factor_ids: [...card!.factor_ids],
      words: card!.words,
      persistedGraph: M2,
      expected_graph_hash: computeAnalysisAffectingGraphHash(M2 as never),
      reading_token: identityConfirmReadingToken({ outcome_id: card!.outcome_id, factor_ids: [...card!.factor_ids], words: card!.words }),
    });
    writeFileSync(`${OUT}/A-card-and-edit.json`, JSON.stringify({ card, edit_kind: edit.kind, reason: (edit as Json).reason ?? null }, null, 2));
    expect(edit.kind, 'the canonical writer accepts the Yes on m2').toBe('mutated');
    const A = (edit as Json).mutatedGraph as Json;

    const A2 = clone(A);
    reframeSubs(A2, 1);

    const B = clone(A);
    reframeSubs(B, HORIZON_MONTHS);
    const subsB = B.nodes.find((n: Json) => n.id === SUBS);
    subsB.label = 'Paying subscribers at 12 months';

    const hashes: Record<string, string> = { m2: computeAnalysisAffectingGraphHash(M2 as never) };
    for (const [tag, g] of [['A', A], ['A2', A2], ['B', B]] as const) {
      hashes[tag] = computeAnalysisAffectingGraphHash(g as never);
      writeFileSync(`${OUT}/${tag}-graph.json`, JSON.stringify(g, null, 2));
    }
    writeFileSync(`${OUT}/graph-hashes.json`, JSON.stringify(hashes, null, 2));
  });

  it('captures the PLoT payload CEE run_analysis sends for each', async () => {
    for (const tag of ['A', 'A2', 'B'] as const) {
      const g = JSON.parse(readFileSync(`${OUT}/${tag}-graph.json`, 'utf8')) as Json;
      const p = await capture(g, tag);
      writeFileSync(`${OUT}/${tag}-plot-payload.json`, JSON.stringify(p, null, 2));
    }
  });
});
