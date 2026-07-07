# Lane ISL-R3 (lane 11) evidence report — V2 wire completeness (audit T1-5 / T1-6 + forward-contract seeds)

- Date: 2026-07-07
- Branch: `claude-lane11/v2-wire-completeness` (fresh worktree from `origin/staging` @ `95685e0`)
- Doctrine authorization: provisional — newly exposed wording surfaces (edge-sensitivity
  `interpretation`, path `mechanism`) tagged `provisional_doctrine_v0` in their field
  descriptions; the strings themselves are pre-existing analyzer output, not new copy.
- Contract status: **ADDITIVE ONLY** — no boundary field renamed, retyped, or removed.
  Every new field is `Optional` with default `None`, so `exclude_none=True` keeps it
  absent unless computed. The 5-repo frozen-contract manifest (lane 4) established no
  strict parsers reject unknown additive fields on this envelope.
- Cross-lane sequencing: this lane **produces** fields only. No consumption wired here;
  PLoT/UI consumption is next round.

## Scope note discovered during implementation

The V2 envelope (`ISLResponseV2`) is only emitted on
`POST /api/v1/robustness/analyze/v2?response_version=2` (or the
`X-ISL-Response-Version: 2` header); the bare endpoint defaults to
`DEFAULT_RESPONSE_VERSION = 1`, the legacy V1-shaped `RobustnessResponseV2` which
already carried `sensitivity` / `path_decomposition` / `factor_evpi`. All changes and
tests below target the `response_version=2` enhanced path — the wire PLoT consumes.
The legacy path is untouched.

## A. T1-6 — edge sensitivity on the V2 envelope (commits `60ca5a3`, `dd38781`)

**Problem.** `RobustnessAnalyzerV2._compute_sensitivity` always computes edge-level
forced-existence + magnitude contrasts (`v1_response.sensitivity`,
`List[SensitivityResult]`), but `_analyze_robustness_v2_enhanced`
(`src/api/robustness.py`) never copied them onto the envelope — computed-then-discarded.
PLoT emits `EDGE_SENSITIVITY_UNAVAILABLE_V2_WIRE` as a result.

**Change.**
- New `EdgeSensitivityV2` model (`src/models/response_v2.py`), V2 naming style matching
  `FragileEdgeV2` / `EdgeEValueV2`: `edge_id` (`from->to` format, same convention as
  fragile edges and e-values), `from_id`, `to_id`, `sensitivity_type`
  (`existence`/`magnitude`), `sensitivity_score` (0-1, |elasticity| normalized by the
  max |elasticity| in the analysis — the exact normalization already used for factor
  `sensitivity_score`), `direction` (sign of elasticity, same rule as factor
  sensitivity), plus the full V1 content preserved raw: `elasticity`,
  `importance_rank`, `interpretation`.
- Additive optional `robustness.edge_sensitivity: Optional[List[EdgeSensitivityV2]]`
  on `RobustnessResultV2`.
- API conversion in `_analyze_robustness_v2_enhanced`: populated only when
  `v1_response.sensitivity` is non-empty (absent under `exclude_none` otherwise, e.g.
  when `analysis_types` omits `"sensitivity"`).

## B. evpi_status producer — verified end-to-end, no fix needed (commit `b233544` test)

**Claim type: serialization-passthrough verification.** Traced every hop:

1. Producer: `RobustnessAnalyzerV2._compute_evpi` (`src/services/robustness_analyzer_v2.py`
   ~line 3136) appends the 4 additive keys per entry: `evpi_status`
   (`below_resolution`/`resolved`), `evpi_noise_floor` (round 6dp),
   `evpi_noise_floor_method` (`z95_worst_case_bernoulli_diff`),
   `evpi_labelling_doctrine` (`provisional_doctrine_v0`). Confirmed present on staging
   base (lane 4 merged).
2. Internal model: `RobustnessResponseV2.factor_evpi: Optional[List[Dict[str, Any]]]` —
   untyped dicts, pydantic preserves all keys.
3. API layer: `builder.set_results(..., factor_evpi=v1_response.factor_evpi)` — the api
   layer does **not** rebuild entries (the audit's concern); raw passthrough.
4. Envelope: `ISLResponseV2.factor_evpi: Optional[List[Dict[str, Any]]]`;
   `model_dump(by_alias=True, exclude_none=True)` does not strip non-None dict keys.

**Result: keys DO serialize onto the HTTP response; no code change required.** The gap
was test coverage — previously only analyzer-unit tests
(`tests/unit/test_scientific_enhancements.py`) asserted the keys; nothing asserted them
on the wire. Now locked by `TestFactorEvpiLabellingOnWire::test_evpi_labelling_keys_on_http_response`,
which asserts all 4 keys + base keys on the `response_version=2` HTTP body.

## C. T1-6 remainder + T1-5 — path decomposition + reference-option disclosure (commits `60ca5a3`, `dd38781`)

**Path decomposition.** `include_path_decomposition=true` computed
`v1_response.path_decomposition` then the V2 envelope dropped it. Added V2 mirror models
`PathContributionV2` / `PathDecompositionV2` in `src/models/response_v2.py` (same schema
as the internal V1 models; mirrored rather than imported because `robustness_v2.py`
imports from `response_v2.py` — the one-way import is documented in both files) and a
top-level `path_decomposition` field on `ISLResponseV2`, matching the V1 response's
top-level placement. Payload-size concern is inherently handled: the field remains
request-gated by `include_path_decomposition` (off by default), so it is only emitted
when explicitly requested.

**T1-5 disclosure.** Edge sensitivity, factor sensitivity, and the fragile-edge
classification derived from edge sensitivity are all computed against
`request.options[0]` (`ref_option = request.options[0]` at
`robustness_analyzer_v2.py` lines 1503, 1568, 1622, 1829) — previously undisclosed.
Added additive optional `sensitivity_reference_option_id` on the envelope, set to
`request.options[0].id` whenever edge or factor sensitivity produced results (absent
otherwise — no disclosure of an analysis that did not run). Producer owns semantics;
consumers can now display the reference instead of inventing one.

`ResponseBuilder` (`src/utils/response_builder.py`) gained the two envelope fields +
setters; `None` defaults preserve exclude_none semantics on every path including
`build_error_response`.

## Evidence — gates run (venv: repo poetry env, Python 3.11)

| Check | Result |
|---|---|
| New wire-shape tests `tests/integration/test_v2_wire_completeness.py` (TestClient, in-process, `ISL_AUTH_DISABLED=true`) | **12/12 passed** |
| Tier-0 regression gate `tests/integration/test_tier0_regression_gate.py` | **19/19 passed** |
| `mypy src/` | **Success: no issues found in 134 source files** |
| `black --check src/` | **134 files unchanged** |
| Collection (`pytest --co -q --ignore=tests/_archived`, pre-push scope) | **2509 tests collected, 0 errors** (6 pre-existing collection errors exist only under `tests/_archived/`, which the pre-push gate ignores) |
| Same-seed determinism of new sections | asserted by `test_same_seed_new_sections_are_deterministic` (identical `edge_sensitivity`, `factor_evpi`, `path_decomposition`, `sensitivity_reference_option_id` across two identical seeded requests) |
| Full pre-push gate `scripts/pre-push-validate.sh` | run at push time — result recorded in the PR |

## Additive-only proof points

- `test_existing_envelope_fields_unaffected` asserts all pre-existing envelope keys
  still present alongside the new ones.
- `test_edge_sensitivity_absent_when_sensitivity_not_requested`,
  `test_path_decomposition_absent_when_not_requested`,
  `test_reference_absent_when_no_sensitivity_computed`,
  `test_factor_evpi_absent_without_include_voi` prove absence (not `null`, not `[]`)
  when the underlying analysis did not run — exclude_none semantics preserved.

## Follow-ups (not this lane)

- PLoT: consume `robustness.edge_sensitivity` and retire
  `EDGE_SENSITIVITY_UNAVAILABLE_V2_WIRE`; surface `sensitivity_reference_option_id`
  and `path_decomposition` (next round per cross-lane sequencing rule).
- Consider typing `factor_evpi` entries (currently `Dict[str, Any]` at every hop) once
  the labelling doctrine is ratified.
