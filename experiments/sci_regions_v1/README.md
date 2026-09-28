# Running SCI-REGIONS v1

This directory is isolated research code. Run it beside the frozen `experiments/r3b_sim` directory from commit `fdb51e84a673993de7cddb85d5b963741941b80a`.

```sh
cd experiments/sci_regions_v1
python3 study.py --output output
python3 mutant_checks.py
python3 -m pytest -q -c /dev/null -p no:cacheprovider tests/test_regions.py
python3 package_results.py
```

The study uses the frozen R3-B NumPy dependency plus Python's standard library. It makes no provider or network calls. `study.py` verifies source hashes before running R3-B, caps the run at 600 seconds, 1 GiB peak RSS, one million synthetic evaluations and 4,000 R3-B evaluations, and records `NOT_EVALUATED` coordinates on a budget stop. It writes deterministic per-case JSON separately from the run manifest's timing and memory values. `package_results.py` writes the readable study summary and a deterministic archive of raw JSON.

Open `output/comparison.html` as a local file to inspect the point, threshold and map views. It embeds its computed data and has no network resources. The underlying `output/results.zip` contains all per-case JSON, `run-manifest.json` and `mutant-evidence.json`.

The study is tier X and exploratory. Numeric preference on R3-B is a float point estimate with a heuristic roundoff screen. The full six-option comparison is incomplete because three options lack intervention values.
