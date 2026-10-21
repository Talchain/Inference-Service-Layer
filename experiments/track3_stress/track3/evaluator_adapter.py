"""Call the pinned local PLoT /v2/run (which calls the pinned local ISL) and keep everything.

The capture proxy between PLoT and ISL records the ACTUAL translated engine request and
ISL's reply. Results are read from PLoT's response (the service boundary); the captured
ISL exchange is retained beside it. Memoisation keys on the complete request body plus the
engine pins: a label or hypothesis name is never a cache key (fault `stale_cache` does
exactly that, to prove the check catches it).
"""
import hashlib, json, os, subprocess, time, urllib.error, urllib.request
from fractions import Fraction as Q

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("T3_ROOT", "/private/tmp/track3-engine")
PLOT_URL = "http://127.0.0.1:18732/v2/run"
ISL_PATH = "/api/v1/robustness/analyze/v2"


def _git(repo, *args):
    return subprocess.run(["git", "-C", os.path.join(ROOT, repo), *args], capture_output=True, text=True).stdout.strip()


def _cwd_of(pid_file):
    try:
        pid = open(os.path.join(ROOT, "logs", pid_file)).read().strip()
        out = subprocess.run(["lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"], capture_output=True, text=True).stdout
        return [line[1:] for line in out.splitlines() if line.startswith("n")][0]
    except Exception as exc:          # recorded, never silently assumed
        return f"UNKNOWN ({exc})"


class Engine:
    def __init__(self, capture_dir, faults=frozenset()):
        self.capture_dir, self.faults = capture_dir, faults
        self.cache, self.calls, self.cache_hits = {}, 0, 0

    def __enter__(self):
        env = dict(os.environ, CAPTURE=self.capture_dir, T3_ROOT=ROOT)
        out = subprocess.run(["zsh", os.path.join(HERE, "engine.sh"), "start"], env=env, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError("engine did not start: " + out.stdout[-2000:] + out.stderr[-2000:])
        self.identity = self._identity()
        return self

    def __exit__(self, *exc):
        subprocess.run(["zsh", os.path.join(HERE, "engine.sh"), "stop"], capture_output=True)

    def _identity(self):
        isl_module = subprocess.run(
            [os.path.join(ROOT, "isl-venv/bin/python"), "-c",
             "import src.services.robustness_analyzer_v2 as m, numpy, scipy, pydantic, fastapi;"
             "print(m.__file__, numpy.__version__, scipy.__version__, pydantic.__version__, fastapi.__version__)"],
            cwd=os.path.join(ROOT, "isl"), capture_output=True, text=True).stdout.split()
        node = subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()
        schemas = json.load(open(os.path.join(ROOT, "plot/node_modules/@talchain/schemas/package.json")))["version"]
        return {
            "plot_commit": _git("plot", "rev-parse", "HEAD"), "isl_commit": _git("isl", "rev-parse", "HEAD"),
            "plot_dirty_tracked": _git("plot", "status", "--porcelain", "--untracked-files=no") != "",
            "isl_dirty_tracked": _git("isl", "status", "--porcelain", "--untracked-files=no") != "",
            "isl_analyzer_module": isl_module[0] if isl_module else "UNKNOWN",
            "isl_deps": dict(zip(["numpy", "scipy", "pydantic", "fastapi"], isl_module[1:])),
            "plot_node": node, "plot_talchain_schemas": schemas,
            "isl_process_cwd": _cwd_of("isl.pid"), "plot_process_cwd": _cwd_of("plot.pid"),
            "plot_endpoint": PLOT_URL, "isl_endpoint": ISL_PATH,
        }

    def key(self, body, hypothesis):
        if "stale_cache" in self.faults:
            return hypothesis                                   # FAULT: a label is not a numerical identity
        blob = json.dumps({"body": body, "engine": [self.identity["plot_commit"], self.identity["isl_commit"]]},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def evaluate(self, body, hypothesis, timeout_s=60):
        k = self.key(body, hypothesis)
        if k in self.cache:
            self.cache_hits += 1
            return dict(self.cache[k], cache_hit=True)
        before = set(os.listdir(self.capture_dir)) if os.path.isdir(self.capture_dir) else set()
        started = time.perf_counter()
        try:
            req = urllib.request.Request(PLOT_URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                status, plot = r.status, json.load(r)
        except urllib.error.HTTPError as e:
            status, plot = e.code, json.loads(e.read() or b"{}")
        except Exception as e:                                  # timeout / refused: a failed row, kept
            status, plot = None, {"analysis_status": "failed", "status_reason": repr(e)}
        elapsed = time.perf_counter() - started
        self.calls += 1
        time.sleep(0.05)
        new = sorted(set(os.listdir(self.capture_dir)) - before) if os.path.isdir(self.capture_dir) else []
        isl = [json.load(open(os.path.join(self.capture_dir, f))) for f in new]
        isl = [c for c in isl if c["path"].startswith(ISL_PATH)]
        result = {"http_status": status, "plot": plot, "isl_exchanges": isl, "elapsed_s": elapsed, "cache_hit": False}
        self.cache[k] = result
        return result


def rows_from(result, scale):
    """Per-option rows from PLoT's response, in £000. Missing or non-computed options stay rows."""
    plot = result["plot"]
    by_id = {o.get("option_id") or o.get("id"): o for o in (plot.get("option_comparison") or [])}
    rows = {}
    for oid, o in by_id.items():
        out = o.get("outcome") or {}
        n = out.get("n_valid_samples") or 0
        mean, sd = out.get("mean"), out.get("std")
        ok = o.get("status") == "computed" and mean is not None and sd is not None and n > 0
        rows[oid] = {
            "status": "supported" if ok else "failed",
            "reason": None if ok else f"option status {o.get('status')!r}; outcome {out!r}",
            "level": float(Q(repr(mean)) * scale) if ok else None,
            "sd": float(Q(repr(sd)) * scale) if ok else None,
            "se": float(Q(repr(sd)) * scale) / n ** 0.5 if ok else None,
            "n": n, "p_goal": o.get("probability_of_goal"),
            "p_goal_interval": [(o.get("probability_of_goal_precision") or {}).get(k)
                                for k in ("interval_lower", "interval_upper")],
            "win_share": o.get("win_probability"),
        }
    return rows
