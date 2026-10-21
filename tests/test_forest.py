"""Run with:  python -m unittest discover tests"""
import unittest

from hforest import RULES, Forest, InvalidEdit, Record, Scope, choose_experiment, fork, merge, paired_gate, retrieve
from hforest.scope import COVERS, DISJOINT, EQUIVALENT, OVERLAPS, UNKNOWN, relation

A = Scope({"protocol": "immersion", "Cr": {"min": 0.5, "max": 3.0}})
B = Scope({"protocol": "marine", "Cr": {"min": 0.5, "max": 3.0}})
AB = Scope({"protocol": ["immersion", "marine"], "Cr": {"min": 0.5, "max": 3.0}})
PLAN = {"id": "P1", "objective": "o", "protocol": "p", "intervention": "i", "controls": ["c"],
        "observables": ["y"], "falsifiers": ["f"]}


def hyp(hid="H1", scope=A, **extra):
    return dict({"id": hid, "statement": "Cr lowers loss", "scope": scope, "predictions": ["loss falls"],
                 "falsifiers": ["loss flat"], "mechanism": "rust layer", "variables": ["Cr"],
                 "endpoints": ["loss"]}, **extra)


def forest_with(hid="H1", scope=A):
    f = Forest()
    f.edit("unexplained_observation", "propose", [hid], "lead", hypothesis=hyp(hid, scope))
    f.add_record(Record("R", "raw", scope, "lab", tier="E2"))
    f.add_record(Record("R+", "admitted", scope, "qc", tier="E4", admits="R"))
    return f


class ScopeTest(unittest.TestCase):
    def test_relations(self):
        self.assertEqual(relation(A, A)[0], EQUIVALENT)
        self.assertEqual(relation(AB, A)[0], COVERS)
        self.assertEqual(relation(A, AB)[0], OVERLAPS)
        self.assertEqual(relation(A, B)[0], DISJOINT)
        self.assertEqual(relation(A, Scope({"Cr": 1.0}))[0], UNKNOWN)  # target silent on protocol
        self.assertEqual(relation(Scope({"Cr": {"max": 1}}), Scope({"Cr": 2}))[0], DISJOINT)
        self.assertEqual(relation(Scope({}, exclusions=("protocol=marine",)), B)[0], DISJOINT)


class GrowthRulesTest(unittest.TestCase):
    def test_status_needs_admitted_in_scope_evidence(self):
        f = forest_with()
        with self.assertRaises(InvalidEdit):
            f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R"], verdict="supports", status="supported")
        f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], verdict="supports", status="supported")
        self.assertEqual(f.hypotheses["H1"].status, "supported")
        self.assertEqual([s["to"] for s in f.hypotheses["H1"].history], ["proposed", "tested", "supported"])

    def test_only_e2_e3_scientific_records_can_be_admitted(self):
        f = Forest()
        f.add_record(Record("M", "model output", A, "model", tier="E1"))
        f.add_record(Record("Ops", "pump failure", A, "ops", tier="E2", kind="infrastructure"))
        for admits, kind in (("M", "scientific"), ("Ops", "infrastructure"), ("missing", "scientific")):
            with self.assertRaises(InvalidEdit):
                f.add_record(Record(f"X-{admits}", "x", A, "qc", tier="E4", kind=kind, admits=admits))

    def test_scope_mismatch(self):
        f = forest_with("H1", AB)
        f.add_record(Record("S", "raw", B, "lab", tier="E2"))
        f.add_record(Record("S+", "ok", B, "qc", tier="E4", admits="S"))
        f.add_record(Record("T", "raw", A, "lab", tier="E2"))
        f.add_record(Record("T+", "ok", A, "qc", tier="E4", admits="T"))
        with self.assertRaises(InvalidEdit):  # disjoint evidence
            f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["S+"], verdict="supports", target_scope=A)
        with self.assertRaises(InvalidEdit):  # partial overlap, no review
            f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["T+"], verdict="supports")
        f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["T+"], verdict="supports",
               scope_review={"T+": {"reviewer": "qc", "rationale": "immersion is the deciding condition"}})

    def test_partial_scope_evidence_cannot_settle_the_whole_claim(self):
        f = forest_with("H1", AB)
        f.add_record(Record("T", "raw", A, "lab", tier="E2"))
        f.add_record(Record("T+", "ok", A, "qc", tier="E4", admits="T"))
        with self.assertRaises(InvalidEdit):
            f.edit("admitted_contradiction_in_scope", "attach_evidence", ["H1"], "x", ["T+"], verdict="contradicts",
                   target_scope=A, status="contradicted")
        f.edit("admitted_contradiction_in_scope", "attach_evidence", ["H1"], "x", ["T+"], verdict="contradicts",
               target_scope=A, status="tested")

    def test_proposition_1_failures_cannot_refute(self):
        for trigger, (ops, reachable, needs_admitted) in RULES.items():
            if reachable & {"contradicted", "scoped"}:
                self.assertTrue(needs_admitted, trigger)
        f = forest_with()
        f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], verdict="supports", status="tested")
        for trigger in ("invalid_experiment", "execution_blocked"):
            for status in ("contradicted", "scoped"):
                with self.assertRaises(InvalidEdit):
                    f.edit(trigger, "record_event", ["H1"], "pump failed", event_class="failure", status=status)

    def test_proposition_2_transfer_inherits_nothing(self):
        f = forest_with()
        f.add_plan(PLAN)
        before = (f.hypotheses["H1"].status, f.hypotheses["H1"].authorized_scope)
        bridge = dict(kind="mechanism", target_scope=B, bridge_plan="P1", shared_mechanism="rust layer",
                      shared_conditions="same Cr range", risks="chloride cycling differs",
                      differences={"protocol": "immersion -> marine air changes wet/dry cycling"})
        with self.assertRaises(InvalidEdit):  # every changed condition needs a stated impact
            f.edit("cross_scope_request", "propose_transfer", ["H1"], "reuse", no_automatic_inheritance=True,
                   **dict(bridge, differences={}))
        f.edit("cross_scope_request", "propose_transfer", ["H1"], "reuse", no_automatic_inheritance=True, **bridge)
        self.assertEqual((f.hypotheses["H1"].status, f.hypotheses["H1"].authorized_scope), before)
        self.assertEqual((f.relations[-1]["kind"], f.relations[-1]["transfer_kind"]), ("transfer_candidate", "mechanism"))
        with self.assertRaises(InvalidEdit):
            f.edit("cross_scope_request", "propose_transfer", ["H1"], "reuse", no_automatic_inheritance=False, **bridge)

    def test_propose_cannot_set_state(self):
        with self.assertRaises(InvalidEdit):
            Forest().edit("unexplained_observation", "propose", ["H1"], "x", hypothesis=hyp(status="supported"))

    def test_split_children_must_partition_inside_parent(self):
        f = forest_with("H1", AB)
        f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], verdict="supports",
               target_scope=A, status="tested")
        with self.assertRaises(InvalidEdit):
            f.edit("condition_dependent_outcome", "split", ["H1"], "x", ["R+"], dimension="protocol",
                   children=[{"id": "c1", "scope": A}, {"id": "c2", "scope": A}])
        f.edit("condition_dependent_outcome", "split", ["H1"], "x", ["R+"], dimension="protocol",
               children=[{"id": "c1", "scope": A}, {"id": "c2", "scope": B}])
        self.assertEqual(f.hypotheses["H1"].status, "scoped")
        self.assertEqual({r["src"] for r in f.relations if r["kind"] == "refines"}, {"c1", "c2"})


class MergeTest(unittest.TestCase):
    def test_admission(self):
        canonical = Forest()
        canonical.edit("unexplained_observation", "propose", ["H1"], "x", hypothesis=hyp("H1", A))
        f = fork(canonical)
        f.forest.add_plan(PLAN)
        f.forest.edit("unexplained_observation", "propose", ["dup"], "x", hypothesis=hyp("dup", A))
        f.forest.edit("unexplained_observation", "propose", ["sib"], "x", hypothesis=hyp("sib", B, plans=["P1"]))
        f.forest.edit("unexplained_observation", "propose", ["new"], "x",
                      hypothesis=hyp("new", A, mechanism="passive film", plans=["P1"]))
        report = merge(f, canonical)
        self.assertEqual([a["id"] for a in report["archived"]], ["dup"])
        self.assertEqual({a["id"]: a["decision"] for a in report["admitted"]}, {"sib": "derived_from", "new": "new"})
        self.assertIn({"kind": "derived_from", "src": "sib", "dst": "H1"},
                      [{k: r[k] for k in ("kind", "src", "dst")} for r in canonical.relations])

    def test_only_controller_commits_and_agent_stays_author(self):
        canonical = forest_with()
        with self.assertRaises(InvalidEdit):
            canonical.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], by="agent",
                           verdict="supports")
        f = fork(canonical)
        f.forest.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], by="agent",
                      verdict="supports", status="supported")
        merge(f, canonical)
        self.assertEqual((canonical.log[-1]["body"]["by"], canonical.log[-1]["body"]["author"]), ("controller", "agent"))
        self.assertEqual(canonical.assessments[-1].by, "agent")
        self.assertEqual(Forest.from_events(canonical.log).head, canonical.head)

    def test_branch_budget(self):
        canonical = Forest()
        f = fork(canonical)
        f.forest.add_plan(PLAN)
        for i in range(3):
            f.forest.edit("unexplained_observation", "propose", [f"H{i}"], "x", by="agent",
                          hypothesis=hyp(f"H{i}", A, mechanism=f"mechanism {i}", plans=["P1"]))
        report = merge(f, canonical, budget=2)
        self.assertEqual(([a["id"] for a in report["admitted"]], [a["id"] for a in report["archived"]]),
                         (["H0", "H1"], ["H2"]))

    def test_stale_fork_edit_is_rejected(self):
        canonical = forest_with()
        f = fork(canonical)
        f.forest.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], verdict="supports", status="tested")
        canonical.edit("obsolete", "retire", ["H1"], "superseded")
        report = merge(f, canonical)
        self.assertTrue(report["stale_base"])
        self.assertEqual(len(report["rejected"]), 1)
        self.assertEqual(canonical.hypotheses["H1"].status, "retired")


class LogAndPriorsTest(unittest.TestCase):
    def test_replay_and_tamper_detection(self):
        f = forest_with()
        f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], verdict="supports", status="supported")
        self.assertEqual(Forest.from_events(f.log).head, f.head)
        tampered = [dict(e) for e in f.log]
        tampered[-1] = dict(tampered[-1], body=dict(tampered[-1]["body"], evidence=["R"]))
        with self.assertRaises(InvalidEdit):
            Forest.from_events(tampered)

    def test_direct_prior_needs_authorization(self):
        f = forest_with()
        f.edit("admitted_support_in_scope", "attach_evidence", ["H1"], "x", ["R+"], verdict="supports", status="supported")
        target = Scope({"protocol": "immersion", "Cr": 2.0})
        self.assertEqual(retrieve(f, target, variables=["Cr"])[0]["role"], "in_scope_unvalidated")
        f.edit("promotion_passed", "authorize", ["H1"], "gate", ["R+"], decision=paired_gate([(0, 1)] * 5), authorized_scope=A)
        card = retrieve(f, target, variables=["Cr"])[0]
        self.assertEqual((card["role"], card["inherits_conclusion"]), ("direct_prior", True))
        self.assertEqual(retrieve(f, B, variables=["Cr"])[0]["role"], "transfer_candidate")

    def test_choose_experiment_separates_rivals(self):
        f = Forest()
        for t in range(4):
            f.edit("unexplained_observation", "propose", [f"T{t}"], "x", hypothesis=hyp(f"T{t}", AB, mechanism=f"t{t}"))
        f.edit("unresolved_competition", "link_competition", ["T0", "T1", "T2", "T3"], "x", observable="high")
        doses = {f"x{x}": Scope({"protocol": "immersion", "Cr": 1.0, "x": x}) for x in range(4)}
        best, bits = choose_experiment(f, doses, lambda h, e: int(e[1:]) >= int(h.id[1:]))
        self.assertEqual((best, bits), ("x1", 1.0))  # x1 splits thresholds {0,1} | {2,3}
        self.assertEqual(choose_experiment(f, {"far": Scope({"protocol": "salt"})}, lambda h, e: 0), (None, 0.0))
        # a run that fails half the time and then reads low carries less, and credence moves the best dose
        noisy = lambda h, e: {True: 0.5, False: 0.5} if int(e[1:]) >= int(h.id[1:]) else {False: 1.0}  # noqa: E731
        self.assertLess(choose_experiment(f, doses, noisy)[1], 1.0)
        weight = lambda h, e: 1.0 if h.id in ("T2", "T3") else 0.0  # noqa: E731
        self.assertEqual(choose_experiment(f, doses, lambda h, e: int(e[1:]) >= int(h.id[1:]), weight=weight),
                         ("x2", 1.0))  # only T2 and T3 still carry weight

    def test_paired_gate(self):
        self.assertAlmostEqual(paired_gate([(0, 1)] * 5)["p_value"], 1 / 32)
        self.assertFalse(paired_gate([(0, 1)] * 4)["accepted"])  # p = 1/16 > 0.05
        self.assertFalse(paired_gate([(0, 1)] * 9 + [(1, 0)])["accepted"])  # one regression


if __name__ == "__main__":
    unittest.main()
