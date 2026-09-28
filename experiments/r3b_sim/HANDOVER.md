# R3-B (Cloud-2) handover: cloud session → local session (28 Sep 2026)

Read this first. The plan is `PLAN.md`; the review is `REVIEW-20260928.md`; the evaluation is `EVALUATION.md`.
Nothing here depends on the cloud container: every script and capture referenced below is in this branch.

## State at handover

| Workstream | Where | State | Next action |
|---|---|---|---|
| W1: EVPPI decision-relevance gate | ISL PR #198, branch `r3b/evppi-decision-relevance`, head `c054ce4` | REVIEW_READY on #72; AIQ ACKed the meaning (5868801032); DL APPROVE 5869397004; 10k-draw RESULT 5869625225 (current rule 4–9% false "resolved", gated 0–0.5%; 0.0042 power 60.5%); R3-B declined the merge (brief), 5869561969 | Drive to green and mergeable; answer reviews. Never merge. |
| Truncation P0 (DL assignment 5868680004, AIQ ruling 5868664986) | ISL PR {TRUNC_PR}, branch `r3b/strength-truncation`, head {TRUNC_HEAD} | REVIEW_READY on #72 ({TRUNC_POST}) | Same. AIQ must accept the eng-hiring-4 leader change on staging. |
| Next-capability proposal | #72 5868879851 + amendment in the truncation post | Awaiting the DL | On DL agreement: P1, P2 and the time contract (below). No build before that. |
| Evaluation, review and plan | ISL branch `r3b/sim-prototype`, `experiments/r3b_sim/` | Pushed | Add to it; never edit the frozen study (`FROZEN.sha256`). |
| Programme docs | `olumi-programme-docs`, branch `claude/r3b-sim-evaluation-5df3vn` | Clean | None. |

## Open items, in order
0. **#198 merge:** the DL approved it and assigned the merge to R3-B (5869397004). R3-B's brief forbids merging
   unless Paul authorises it, so R3-B released the window (5869561969). Check whether #198 has merged; never merge
   without Paul's explicit go-ahead in the session.
1. **Keep both ISL PRs green and mergeable** on every check-in: CI on the latest head, merge conflicts with
   `staging`, review threads and Claude Approvals rows. A red or conflicted head is always work now.
2. **#197 interaction (Cloud-1):** whichever of #197 and the truncation PR lands second must run #197's
   `test_no_identity_thresholds_are_base_d1cef9a` under the `former_truncated_edge_sampler` fixture. The truncation
   PR already does this for the copy on staging. Offer to do it; the trial merge otherwise resolves cleanly.
3. **W1 wording:** ask AIQ to confirm that `below_resolution` means "not detected at this resolution", never
   "no information value", and that AIQ re-bases the eng-hiring-4 control row.
4. **After the DL agrees the preparation work:**
   - P1: an offline typed-compilation agent on the frozen corpus. Score it on agreement with the frozen hand
     mapping (class precision and recall, zero invented semantics), manual effort saved and the usefulness of
     its questions.
   - P2: goal, constraint and preference regions from a grid on the A-180910Z and C-183807Z X models; add
     PRIM or scenario discovery only where it beats the grid.
   - The time-semantics contract: one drafter, one document (horizon, onset, flow meaning, evaluation rule)
     reviewed by Canonical, MG and AIQ; each part behind a flag; activated only when invalidation works end to end.
5. **Deferred:** W4/W5 (replay kit and served head-to-head). `replay/` holds the method and the eh4 captures.
6. **Follow-up to raise, not to build:** the parse-time clamp on the edge-strength *mean* and the flip-search
   ranges still use +/-1 (the ruling covered sampled draws only).

## Rules that still apply
- Never: push to `main` or `staging`, force-push, merge, deploy, change CEE, PLoT or UI, change shared schemas,
  add dependencies, or call OpenAI, Anthropic or any served Olumi endpoint. The private corpus never leaves the
  repositories.
- ISL: run `bash scripts/pre-push-validate.sh < /dev/null` before every push (the script reads stdin in hook
  mode). Never edit the working tree while it runs. Check `git status` and `git diff --staged` before committing.
- #72 posts: CLAIM, BLOCKER, REVIEW_READY or RESULT; row → root → owner → next; at most 8 lines; header
  `[TYPE — subject | LANE (session) → RECIPIENTS]`. Heads freeze at REVIEW_READY. HIGH changes go to the DL
  (work-in-progress limit 2).
- GitHub posts end with the Claude Code attribution footer.

## Local environment notes
- ISL tests need `ISL_AUTH_DISABLED=true`. Python 3.11 with Poetry (`poetry install`).
- A git worktree gets a new, empty Poetry venv. Run tests in a worktree with the main venv's python
  (`$(poetry env info -p)/bin/python -m pytest`), and run the pre-push from the main checkout.
- Many ISL files already fail `black` at base. Format only your own lines.
- Tests that pin noised numbers use the `auto_noise_enabled` fixture. Tests designed under the former +/-1
  sampler use `former_truncated_edge_sampler`.
- Use the `gh` CLI for PRs and comments. The local session has no PR webhooks: check the PRs at the start of
  each session (`gh pr checks <n>`, `gh pr view <n> --comments`) or run `/loop` for periodic check-ins.

## Evidence index
- W1: `evppi_gate/study.py` and `results.json` (known-answer study); `evppi_gate/adversarial/` (variants,
  calibration, the eh4 replay diagnosis and the captured `eh4-isl-request.json`); `evppi_gate/w1_10k.py` and
  `w1_10k_results.json` (served 10,000 draws).
- Truncation: `truncation/trunc_numbers.py`; the PR's `tests/unit/test_edge_strength_unbounded.py`.
- Replay method: `replay/`.
