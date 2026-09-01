# Existing Monte Carlo statistics on the enhanced response

This change exposes two statistics already computed by ISL on the enhanced
`/api/v1/robustness/analyze/v2?response_version=2` response. It does not change
sampling, outcome numbers, ranking or recommendation policy.

- `tie_rate`: `tie_count / request.n_samples`. A tie is two or more finite
  outcomes exactly equal to the maximum in one draw, before optional
  auto-scaled noise. An all-nonfinite draw is not a tie but remains in the
  requested-draw denominator. This is not near-equality of expected outcomes,
  calibrated confidence, or an authoritative recommendation.
- `edge_existence_rates`: the existing map of `from->to` node-ID pairs to
  `existence_count / sampler._sample_count`. These are realised Bernoulli
  sampling frequencies, not configured probabilities, strength sensitivity or
  evidence quality. Identity keys are copied unchanged.

Both fields are optional and top-level. The API copies internal metadata
through the existing response builder; no new metadata container is introduced.
No computed value means absence. A computed zero stays zero; an explicitly
computed empty map stays empty. Blocked/error responses do not invent values.
The existing internal/legacy response remains unchanged.

Invalid optional diagnostics are withheld with the existing
`ISL_SAMPLING_DIAGNOSTICS_INVALID` inference warning instead of failing completed
computation. Each diagnostic is independent: a valid tie rate survives an invalid
edge map, and vice versa. An invalid map is omitted as a whole, never clamped or
partially salvaged. One warning message names every withheld diagnostic so a
consumer that deduplicates warning codes does not hide a second invalid field.

The existing sampler can aggregate directed and bidirected edges with the same
encoded endpoint key, producing a value above one or a misleading in-range sum.
The transport withholds that ambiguous map without changing the sampler, metric
denominator or key scheme. Uniqueness is checked on the **same filtered inference
graph** the analyzer samples, using its existing filter; discarded organisational
edges cannot cause a valid sampled map to be withheld. The underlying sampler
defect remains a separate scientific implementation issue.

## Compatibility and deployment

The baseline enhanced Pydantic model at
`28fe0c950f6ca5737f4555c863353d37b734dddf` accepts but silently strips these new
fields (`extra='ignore'`). That is wire-compatible, not successful carriage.
New consumers must accept older producers with these fields absent, without
interpreting absence as zero or stability. Prepare optional consumers before
producer rollout, or explicitly acknowledge that old consumers lose the signal.
The actual PLoT/CEE/UI combinations require their own adapter proof.

ISL's `@talchain/schemas` 0.38.0 artifact is a **test-only** drift comparison,
not a runtime parser. The official baseline refresher records precisely these
two intentional optional supersets against that unchanged artifact. Neither
its pin nor its generated contract snapshot is changed; the exceptions require
review and are not proof that the shared/CEE/UI contracts adopt the fields.

## Evidence and limits

`tests/integration/test_science_statistics_wire.py` uses real FastAPI routes and
real computation with distinct .3/.7 interventions, 100 draws and seed42. Edge
probabilities .1/.9/1 produce tie rates .91/.07/0 and realised edge rates
.09/.93/1. The enhanced response must equal the actual legacy metadata values.
Every pre-existing enhanced response field is also compared against saved
baseline responses from staging28fe0c9, excluding only timestamp and elapsed
processing time. The checked-in fixture contains the original full requests and
responses, including an actual blocked response where neither statistic exists.

Additional controls cover old successful responses remaining absent under the
new model, an explicit empty map, invalid probabilities, and an unrelated
request-ID change. Deliberately dropping either new field must fail the positive
wire controls. Computed statistics do not imply a useful UI, deployed carriage,
or recommendation authority; those remain separate integration claims.
