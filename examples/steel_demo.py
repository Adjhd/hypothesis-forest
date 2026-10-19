"""Formation, evolution and merge on a toy weathering-steel forest.

Run from the repository root:  python examples/steel_demo.py
Numbers are illustrative, not measurements.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hforest import Forest, InvalidEdit, Record, Scope, fork, merge, paired_gate, retrieve  # noqa: E402

BOTH = Scope({"steel": "low_alloy", "protocol": ["weekly_immersion", "marine_atmosphere"], "Cr_wt": {"min": 0.5, "max": 3.0}})
IMMERSION = Scope({"steel": "low_alloy", "protocol": "weekly_immersion", "Cr_wt": {"min": 0.5, "max": 3.0}})
MARINE = Scope({"steel": "low_alloy", "protocol": "marine_atmosphere", "Cr_wt": {"min": 0.5, "max": 3.0}})
H_CR = {"id": "H-Cr", "statement": "Cr lowers corrosion loss of low-alloy steel", "scope": BOTH,
        "predictions": ["loss ratio to Q345B falls as Cr rises"], "falsifiers": ["loss ratio flat or rising with Cr"],
        "mechanism": "Cr-enriched protective inner rust layer", "variables": ["Cr"], "endpoints": ["corrosion_loss"]}


def step(title):
    print(f"\n== {title}")


canonical = Forest()

step("1. Formation: an agent proposes in a fork; the merge admits only testable propositions")
f = fork(canonical)
f.forest.add_plan({"id": "P-Cr-ladder", "objective": "test the Cr effect", "protocol": "45C, 168h, 0.5 wt% NaCl",
                   "intervention": "Cr 0.5 / 1.5 / 3.0 wt%, other elements fixed", "controls": ["Q345B"],
                   "observables": ["mass-loss ratio to Q345B"], "falsifiers": ["no monotone decrease"]})
f.forest.edit("unexplained_observation", "propose", ["H-Cr"], "literature lead", by="agent",
              hypothesis=dict(H_CR, plans=["P-Cr-ladder"]))
f.forest.edit("unexplained_observation", "propose", ["H-Cr-again"], "same idea, reworded", by="agent",
              hypothesis=dict(H_CR, id="H-Cr-again", statement="Chromium reduces corrosion"))
report = merge(f, canonical)
print("admitted:", [a["id"] for a in report["admitted"]], "| archived:", [(a["id"], a["reason"]) for a in report["archived"]])

step("2. Evidence moves status only when admitted and in scope")
canonical.add_record(Record("R1", "immersion ladder: loss ratio 0.62 -> 0.41 for Cr 0.5 -> 3.0", IMMERSION, "batch-07", tier="E2"))
canonical.add_record(Record("R3", "marine exposure: loss ratio flat across Cr", MARINE, "site-02", tier="E3"))
for raw in ("R1", "R3"):
    canonical.add_record(Record(f"{raw}-ok", f"{raw} passed QC and protocol review",
                                canonical.records[raw].scope, "QC", tier="E4", admits=raw))
try:
    canonical.edit("admitted_support_in_scope", "attach_evidence", ["H-Cr"], "ladder", ["R1"], verdict="supports", status="supported")
except InvalidEdit as err:
    print("rejected (raw record):", err)
try:
    canonical.edit("admitted_support_in_scope", "attach_evidence", ["H-Cr"], "ladder", ["R1-ok"], verdict="supports", status="supported")
except InvalidEdit as err:
    print("rejected (evidence covers only part of the claim):", err)
canonical.edit("admitted_support_in_scope", "attach_evidence", ["H-Cr"], "ladder, judged on the immersion part only",
               ["R1-ok"], verdict="supports", target_scope=IMMERSION, status="tested")
print("H-Cr:", canonical.hypotheses["H-Cr"].status)

step("3. A condition-dependent outcome splits the claim by protocol")
canonical.edit("condition_dependent_outcome", "split", ["H-Cr"], "effect in immersion, none in marine air",
               ["R1-ok", "R3-ok"], dimension="protocol",
               children=[{"id": "H-Cr/immersion", "scope": IMMERSION}, {"id": "H-Cr/marine", "scope": MARINE}])
canonical.edit("admitted_support_in_scope", "attach_evidence", ["H-Cr/immersion"], "ladder", ["R1-ok"],
               verdict="supports", status="supported")
canonical.edit("admitted_contradiction_in_scope", "attach_evidence", ["H-Cr/marine"], "flat in marine air", ["R3-ok"],
               verdict="contradicts", status="contradicted")
print({h.id: h.status for h in canonical.hypotheses.values()})

step("4. A failed run is recorded but cannot refute (Proposition 1)")
canonical.add_record(Record("R5", "furnace trip, samples not aged", IMMERSION, "ops-log", tier="E1", kind="infrastructure"))
canonical.edit("invalid_experiment", "record_event", ["H-Cr/immersion"], "furnace trip", ["R5"], event_class="invalid_experiment")
try:
    canonical.edit("invalid_experiment", "record_event", ["H-Cr/immersion"], "furnace trip", ["R5"],
                   event_class="invalid_experiment", status="contradicted")
except InvalidEdit as err:
    print("rejected:", err)
print("status still:", canonical.hypotheses["H-Cr/immersion"].status)

step("5. Authorization widens only through the paired promotion gate")
decision = paired_gate([(False, True)] * 6 + [(True, True)] * 2)
canonical.edit("promotion_passed", "authorize", ["H-Cr/immersion"], "six paired heats", ["R1-ok"], decision=decision,
               authorized_scope=Scope({"steel": "low_alloy", "protocol": "weekly_immersion", "Cr_wt": {"min": 1.0, "max": 3.0}}))
print("gate:", {k: decision[k] for k in ("wins", "losses", "p_value", "accepted")})

step("6. A transfer proposes a bridge; it inherits nothing (Proposition 2)")
canonical.add_plan({"id": "P-salt-spray", "objective": "bridge to salt spray", "protocol": "ASTM B117, 240h",
                    "intervention": "Cr 1.0 / 3.0 wt%", "controls": ["Q345B"], "observables": ["mass loss"],
                    "falsifiers": ["no Cr effect"]})
canonical.edit("cross_scope_request", "propose_transfer", ["H-Cr/immersion"], "reuse in salt spray", kind="mechanism",
               target_scope=Scope({"steel": "low_alloy", "protocol": "salt_spray"}), bridge_plan="P-salt-spray",
               no_automatic_inheritance=True)
h = canonical.hypotheses["H-Cr/immersion"]
print("relation:", canonical.relations[-1]["kind"], "| status:", h.status, "| authorized:", h.authorized_scope.dims)

step("7. Priors for new tasks: roles, not conclusions")
for protocol, cr in (("weekly_immersion", 2.0), ("marine_atmosphere", 2.0), ("salt_spray", 2.0)):
    target = Scope({"steel": "low_alloy", "protocol": protocol, "Cr_wt": cr})
    print(protocol, [(c["id"], c["role"]) for c in retrieve(canonical, target, variables=["Cr"])])

step("8. The forest is its log: save, replay, same head")
path = Path(tempfile.gettempdir()) / "steel_demo_forest.jsonl"
canonical.save(path)
replayed = Forest.load(path)
print(f"{len(canonical.log)} events, head {canonical.head}, replayed head {replayed.head}, saved to {path}")
