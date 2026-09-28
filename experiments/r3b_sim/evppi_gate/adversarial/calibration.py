import numpy as np, sys
sys.argv=[sys.argv[0]]
exec(open(sys.path[0] + "/variants.py").read().split("CASES = {")[0])  # reuse gate()
cases = {
  "independent_null":  lambda th: [10 + 0*th, 9.0 + 0*th],
  "dominated_dep":     lambda th: [10 + 0*th, 9.0 + 0.8*np.tanh(th)],
  "close_null":        lambda th: [10 + 0*th, 9.98 + 0*th],
  "exact_tie_null":    lambda th: [10 + 0*th, 10 + 0*th],
  "close_dominated":   lambda th: [10 + 0*th, 9.90 + 0.09*np.tanh(th)],
  "common_shift_null": lambda th: [10 + 0.5*th, 9.9 + 0.5*th],
  "three_opt_null":    lambda th: [10 + 0*th, 9.95 + 0*th, 9.0 + 0.8*np.tanh(th)],
}
for name, f in cases.items():
    for n in (500, 2000):
        a = b = 0
        for s in range(400):
            r = np.random.default_rng(s); th = r.normal(0, 1, n)
            Y = np.column_stack([mu + r.normal(0, 1, n) for mu in f(th)])
            a += gate(th, Y, 1000 + s); b += gate(th, Y, 1000 + s, comparator="full")
        print(f"{name:18s} n={n:5d} gate-alone rejections /400: train-comparator {a:3d} ({a/4:.1f}%)  full-sample comparator {b:3d} ({b/4:.1f}%)", flush=True)
