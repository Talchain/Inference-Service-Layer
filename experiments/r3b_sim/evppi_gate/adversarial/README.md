# W1 adversarial checks of the decision-gain gate (run 28 Sep 2026, under W1's authority)

These scripts test hypotheses raised in the R3B/Cloud-2 review brief against the gate on ISL branch
`r3b/evppi-decision-relevance`. They import `src.utils.evppi` (for `factor_evppi_estimate`,
`_fit_predict` and `_effective_degree`), so run them from a checkout of that branch:

```bash
ISL_AUTH_DISABLED=true poetry run python <path>/variants.py 200 out.json     # variants_results.json
ISL_AUTH_DISABLED=true poetry run python <path>/calibration.py               # printed table below
```

Each script's `gate()` is its own re-implementation with switchable comparator, clipping, degree and repeated
splits, so the variants can be compared at matched cost. All cases are synthetic and have a known true EVPPI.

**Gate-alone rejection rate on nulls, 400 seeds** (`calibration.py`, one-sided z = 1.645):

| Null case | n | Training-fold comparator | Full-sample comparator |
|---|---|---|---|
| independent | 500 / 2000 | 0.2% / 0.0% | 0.2% / 0.0% |
| dominated but θ-dependent | 500 / 2000 | 0.2% / 0.2% | 0.2% / 0.2% |
| near tie (0.02 apart) | 500 / 2000 | **7.5% / 7.5%** | 0.5% / 0.2% |
| exact tie | 500 / 2000 | **6.8% / 6.2%** | 0.2% / 0.2% |
| close dominated | 500 / 2000 | **8.8%** / 2.2% | 1.0% / 1.0% |
| common shift across options | 500 / 2000 | **5.5%** / 0.5% | 0.2% / 0.0% |
| three options | 500 / 2000 | **6.5% / 5.0%** | 0.5% / 0.0% |

The training-fold comparator is anti-conservative on near ties. The gate therefore now uses the full-sample best
option, which is EVPPI's own baseline. With that comparator the gate is conservative, not calibrated at 5%.

**Findings on each hypothesis:**
- **Clipping held-out θ to the training range:** not confirmed as a power loss. On the weak-tail positive, 12→10
  at n=500 and 26→25 at n=2000 without clipping.
- **Degree 2 versus degree 4:** mixed results.
- **Five repeated splits:** fewer near-tie false positives, but less power on moderate signals at n=500 (69 vs 77).
- **Not yet measured:**
  - the family-wise rate across several factors tested per run;
  - correlated inputs;
  - grouped (joint) information.
