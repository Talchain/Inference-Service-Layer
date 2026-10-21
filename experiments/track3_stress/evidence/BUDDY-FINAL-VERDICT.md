PASS

- **R2-A1:** G1 `A M0:x=0.8,y=0.8` removal → `INCOMPLETE`. None of 27 single-cell deletions returned PASS; 24 hit an unchanged, pre-existing `KeyError`.
- **R2-A2:** removing B’s sum-route rows, or the entire list, → `INCOMPLETE`, with B’s `withstand=False`. Unresolved or failed comparisons cannot produce PASS.
- **M2-1:** engine-spy replay returned `unsupported`, `engine_joint=None`, `n=0` for admitted `partial_overlap`; it never reached the engine.
- **429:** mocked transport replay → two requests, both `cache_hit=False`, cache size zero.
- **Report wording:** accurate. Numerical probe gives rho ≈ ±0.9999999900000002; ideal same/opposite-tail distinctions are correct. Capture `0075.json` confirms generic zero-variance warnings without a specific lost-spread warning ([report.md:80](/private/tmp/track3-engine/out/run-20261021T011249Z-fix2/report.md:80), [report.md:93](/private/tmp/track3-engine/out/run-20261021T011249Z-fix2/report.md:93)).
- **G2 reorder:** captured C/B/A requests replay → PASS; deliberately swapping A/C bindings → FAIL with 26 reference failures. The control is meaningful. All 23 selected pure unit tests passed.