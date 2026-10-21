"""Pure tests (no engine): the oracle, the builder's encoding, and the metric-layer faults."""
import unittest
from fractions import Fraction as Q

from track3 import fixtures as F
from track3.scenario_builder import build_request, graph_level
from track3.stress_metrics import (BRIEF_REFERENCE, check_reference, engine_side, joint_failure_witnesses,
                                   oracle_side, summarise, cell_checks)


def cells_from_oracle(faults=frozenset(), se=0.03, drop=None, omit=None):
    """Cells whose 'engine' rows are the exact oracle levels with a small stated SE: a perfect engine.
    G2 rows plus the G1 extras (option A); `drop` keeps the cell without an engine row, `omit` removes the cell."""
    out = []
    settings = [(h, x, y, "G2") for h in F.HYPOTHESES for x, y in F.G2_SETTINGS] + [("M0", x, y, "G1") for x, y in F.G1_EXTRA]
    for h, x, y, group in settings:
        if True:
            body, _ = build_request(h, x, y, seed=1, n_samples=100, faults=faults)
            for o in (["A"] if group == "G1" else F.OPTIONS):
                if (h, x, y, o) == omit:
                    continue
                lvl = graph_level(body, o)
                row = None if (h, x, y, o) == drop else {"status": "supported", "level": float(lvl), "se": se}
                c = {"group": group, "hypothesis": h, "x": x, "y": y, "scenario_id": F.scenario_id(h, x, y), "option": o,
                     "oracle_level": F.oracle_level(o, h, x, y), "engine": row,
                     "meta": {"unit": F.UNIT, "horizon_months": 12, "target": 90.0, "operator": ">="}}
                c["oracle_side"] = oracle_side(c["oracle_level"])
                c["engine_side"] = engine_side(row["level"], row["se"]) if row else "missing"
                c["checks"] = cell_checks(c)
                out.append(c)
    return out


OK = {"comparable": True, "differs": []}


class Oracle(unittest.TestCase):
    def test_reproduces_brief_reference(self):
        s = summarise(cells_from_oracle())
        for o in F.OPTIONS:
            self.assertEqual(s["options"][o]["oracle"]["meets"], BRIEF_REFERENCE["meets"][o])
            self.assertEqual(s["options"][o]["oracle"]["worst"], BRIEF_REFERENCE["worst"][o])
            self.assertEqual(s["options"][o]["oracle"]["max_regret"], BRIEF_REFERENCE["max_regret"][o])

    def test_g1_boundary_inclusive(self):
        self.assertEqual(F.oracle_level("A", "M0", Q("0.75"), Q("0.75")), 90)
        self.assertTrue(F.meets(Q(90)))
        self.assertFalse(F.meets(Q(90), ">"))

    def test_joint_failure_witness_is_A_under_M0_only(self):
        w = joint_failure_witnesses(cells_from_oracle(), "oracle_side")
        self.assertEqual([(x["hypothesis"], x["option"]) for x in w], [("M0", "A")])


class Builder(unittest.TestCase):
    def test_graph_encodes_fixture_exactly(self):
        for h in F.HYPOTHESES:
            for x, y in F.G2_SETTINGS + F.G1_EXTRA:
                body, _ = build_request(h, x, y, seed=1, n_samples=100)
                for o in F.OPTIONS:
                    self.assertEqual(graph_level(body, o), F.oracle_level(o, h, x, y), (h, x, y, o))

    def test_stresses_are_parameters_not_interventions(self):
        body, _ = build_request("M1", 1, 1, seed=1, n_samples=100)
        for opt in body["options"]:
            self.assertFalse({"x", "y", "demand_stress", "cost_stress"} & set(opt["interventions"]))
        self.assertIn(("growth_commitment", "retention_loss"), {(e["from"], e["to"]) for e in body["graph"]["edges"]})

    def test_builder_faults_move_the_graph(self):
        for fault in ("omit_cost_stress", "reverse_demand_sign", "remove_m1_path"):
            fails = check_reference(cells_from_oracle(frozenset({fault})), summarise(cells_from_oracle(frozenset({fault}))), OK)
            self.assertTrue(fails, fault)


class MetricFaults(unittest.TestCase):
    def test_clean_is_green(self):
        cells = cells_from_oracle()
        self.assertEqual(check_reference(cells, summarise(cells), OK), [])

    def test_each_metric_fault_is_red(self):
        for fault in ("strict_comparison", "average_models", "unique_winner"):
            cells = cells_from_oracle()
            self.assertTrue(check_reference(cells, summarise(cells, frozenset({fault})), OK), fault)

    def test_missing_row_honest_vs_counted_as_success(self):
        cells = cells_from_oracle(drop=("M0", Q(1), Q(1), "A"))
        self.assertEqual(check_reference(cells, summarise(cells), OK), [])
        s = summarise(cells)["options"]["A"]
        self.assertEqual((s["requested"], s["completed"]), (8, 7))
        self.assertTrue(check_reference(cells, summarise(cells, frozenset({"missing_as_success"})), OK))

    def test_tie_is_unresolved_not_met(self):
        self.assertEqual(engine_side(90.0001, 0.03), "indeterminate")
        self.assertEqual(engine_side(90.5, 0.03), "meets")
        self.assertEqual(engine_side(89.5, 0.03), "falls_short")

    def test_horizon_mismatch_is_red(self):
        cells = cells_from_oracle()
        self.assertTrue(check_reference(cells, summarise(cells), {"comparable": False, "differs": ["horizon_months"]}))


if __name__ == "__main__":
    unittest.main()


def resolved_ties(cells, p_goal=1.0):
    return [{"option": c["option"], "scenario_id": c["scenario_id"], "exact_level": str(c["oracle_level"]),
             "p_goal": p_goal, "identity_evaluated": True, "resolved": p_goal == 1.0}
            for c in cells if c["oracle_level"] == F.TARGET]


class Verdict(unittest.TestCase):
    """Buddy R1/R2 BLOCKs: no incomplete or unresolved run may read as PASS."""
    def _analysis(self, cells, sum_route=None):
        from track3.run import analyse
        return analyse(cells, sum_route=resolved_ties(cells) if sum_route is None else sum_route)

    def test_clean_perfect_engine_passes(self):
        a = self._analysis(cells_from_oracle())
        self.assertEqual((a["verdict"], a["gaps"]), ("PASS", []))

    def test_missing_witness_row_is_incomplete_not_pass(self):
        a = self._analysis(cells_from_oracle(drop=("M0", Q(1), Q(1), "A")))
        self.assertEqual(a["reference_failures"], [])
        self.assertEqual(a["verdict"], "INCOMPLETE")
        self.assertTrue(any("demonstration 1" in g for g in a["gaps"]))

    def test_sum_route_strict_flip_is_red(self):
        from track3.stress_metrics import sum_route_failures
        row = {"option": "B", "scenario_id": "M0:x=1,y=1", "exact_level": "90", "identity_evaluated": True}
        self.assertEqual(sum_route_failures([dict(row, p_goal=1.0)]), [])
        self.assertTrue(sum_route_failures([dict(row, p_goal=0.0)]))
        self.assertTrue(sum_route_failures([dict(row, p_goal=1.0, identity_evaluated=False)]))


class G3G4Reference(unittest.TestCase):
    def test_laws_admitted_with_equal_marginals(self):
        for law, t in F.G3_LAWS.items():
            a = F.admit_joint(t)
            self.assertTrue(a["admitted"], law)
            self.assertEqual((a["p1"], a["p2"]), (Q(9, 10), Q(9, 10)))
        self.assertEqual([F.admit_joint(t)["p_both"] for t in F.G3_LAWS.values()], [Q(81, 100), Q(9, 10), Q(8, 10)])

    def test_invalid_tables_refused(self):
        for name, t in F.G3_INVALID.items():
            self.assertFalse(F.admit_joint(t, (Q(9, 10), Q(9, 10)))["admitted"], name)

    def test_marginals_only_gives_bounds_not_a_point(self):
        r = F.joint_from_marginals_only(Q(9, 10), Q(9, 10))
        self.assertEqual((r["identified"], r["bounds"]), (False, (Q(4, 5), Q(9, 10))))

    def test_rho_mapping_exact_or_unsupported(self):
        from track3.scenario_builder import g3_rho_for
        self.assertEqual([g3_rho_for(F.admit_joint(t)) for t in F.G3_LAWS.values()], [0.0, 1.0, -1.0])
        self.assertIsNone(g3_rho_for({"p1": Q(9, 10), "p2": Q(9, 10), "p_both": Q(85, 100)}))

    def test_g4_analytic_matches_brief(self):
        for mu, want in [(90, 0.1586552539), (100, 0.5), (110, 0.8413447461)]:
            self.assertAlmostEqual(F.normal_goal_probability(Q(mu)), want, places=10)


class AdapterCache(unittest.TestCase):
    def test_failure_is_not_memoised(self):
        from track3.evaluator_adapter import Engine
        import track3.evaluator_adapter as A
        e = Engine("/nonexistent"); e.identity = {"plot_commit": "p", "isl_commit": "i"}
        A.PLOT_URL, saved = "http://127.0.0.1:9/v2/run", A.PLOT_URL      # nothing listens: a failed call
        try:
            r1 = e.evaluate({"x": 1}, "h", timeout_s=0.2)
            r2 = e.evaluate({"x": 1}, "h", timeout_s=0.2)
        finally:
            A.PLOT_URL = saved
        self.assertEqual((r1["http_status"], r1["cache_hit"], r2["cache_hit"], e.calls), (None, False, False, 2))


class R2Gaps(unittest.TestCase):
    def _a(self, cells, sum_route):
        from track3.run import analyse
        return analyse(cells, sum_route=sum_route)

    def test_wholly_omitted_g1_cell_is_incomplete(self):
        cells = cells_from_oracle(omit=("M0", Q("0.80"), Q("0.80"), "A"))
        a = self._a(cells, resolved_ties(cells))
        self.assertEqual(a["verdict"], "INCOMPLETE")
        self.assertTrue(any("missing from the requested population" in g for g in a["gaps"]))

    def test_unresolved_ties_without_exact_route_are_incomplete(self):
        cells = cells_from_oracle()
        a = self._a(cells, [])
        self.assertEqual(a["verdict"], "INCOMPLETE")
        self.assertFalse(a["withstand"]["B"][0])
        self.assertTrue(a["withstand"]["C"][0])

    def test_b_ties_dropped_from_exact_route_is_incomplete(self):
        cells = cells_from_oracle()
        sr = [r for r in resolved_ties(cells) if r["option"] != "B"]
        self.assertEqual(self._a(cells, sr)["verdict"], "INCOMPLETE")


class M2Review(unittest.TestCase):
    def test_unmapped_law_is_unsupported_not_evaluated(self):
        from track3 import m2
        class NoEngine:
            def evaluate(self, *a, **k):
                raise AssertionError("an unmapped law must not reach the engine")
        g3 = {"rows": [], "admission": {}, "marginals_only": {}, "engine_refusal_rho_1_2": {"analysis_status": "blocked"}}
        adm = F.admit_joint(F.G3_UNMAPPED["partial_overlap"], (Q(9, 10), Q(9, 10)))
        from track3.scenario_builder import g3_rho_for
        self.assertTrue(adm["admitted"])
        self.assertIsNone(g3_rho_for(adm))
        bad = {"law": "partial_overlap", "exact_p_both": "17/20", "unmapped": True, "status": "computed", "engine_joint": 0.81,
               "engine_marginals": {}, "n": 10000}
        self.assertTrue(m2.g3_failures(dict(g3, rows=[bad], marginals_only={"identified": "False", "bounds": ("4/5", "9/10")})))
        ok = dict(bad, status="unsupported", engine_joint=None, n=0)
        self.assertEqual(m2.g3_failures(dict(g3, rows=[ok], marginals_only={"identified": "False", "bounds": ("4/5", "9/10")})), [])

    def test_429_is_not_memoised(self):
        import http.server, threading, track3.evaluator_adapter as A
        from track3.evaluator_adapter import Engine

        class TooMany(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length") or 0))
                body = b'{"analysis_status": "blocked", "status_reason": "rate limited"}'
                self.send_response(429); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *a): pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TooMany)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        e = Engine("/nonexistent"); e.identity = {"plot_commit": "p", "isl_commit": "i"}
        A.PLOT_URL, saved = f"http://127.0.0.1:{srv.server_address[1]}/v2/run", A.PLOT_URL
        try:
            r1, r2 = e.evaluate({"x": 1}, "h", timeout_s=2), e.evaluate({"x": 1}, "h", timeout_s=2)
        finally:
            A.PLOT_URL = saved; srv.shutdown()
        self.assertEqual((r1["http_status"], r2["cache_hit"], e.calls), (429, False, 2))


class FinalNote(unittest.TestCase):
    def test_every_single_cell_deletion_is_incomplete_without_error(self):
        from track3.run import analyse
        full = cells_from_oracle()
        for i in range(len(full)):
            cells = full[:i] + full[i + 1:]
            a = analyse(cells, sum_route=resolved_ties(cells))
            self.assertNotEqual(a["verdict"], "PASS", full[i]["scenario_id"] + " " + full[i]["option"])
