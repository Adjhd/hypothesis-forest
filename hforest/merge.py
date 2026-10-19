"""Exploration forks and their admission back into the canonical forest.

An agent works in a fork: a private copy of the canonical forest at some head.
Merging replays the fork's new events onto the canonical forest at its
*current* head:

* records and plans are always appended, for provenance;
* each proposed hypothesis is identity-resolved against the canonical
  hypotheses by content key and scope; untested restatements stay in the fork;
* every other edit is re-validated against the current canonical state, so an
  edit that was valid on a stale head is rejected instead of being forced in.

Merges into one canonical forest must run one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

from .forest import Forest, Hypothesis, InvalidEdit, content_key, plan_errors
from .scope import DISJOINT, EQUIVALENT, Scope, relation


@dataclass
class Fork:
    forest: Forest
    base: str  # canonical head when the fork was made
    base_len: int


def fork(canonical: Forest) -> Fork:
    return Fork(canonical.copy(), canonical.head, len(canonical.log))


def admission(candidate: Hypothesis, canonical: Forest, actionable: bool) -> tuple[str, str | None, str]:
    """Return (decision, existing hypothesis id, reason) for one fork hypothesis.

    ``actionable`` means the candidate was tested in the fork or carries a
    declared discriminating plan. Decisions: new, refines, derived_from,
    alias (same proposition and scope as an existing node) or archive.
    """
    key = content_key(candidate)
    if not any(key):
        return "archive", None, "no mechanism, variables or endpoints to resolve identity"
    same = [h for h in canonical.hypotheses.values()
            if h.status != "retired" and not h.merged_into and content_key(h) == key]
    for h in same:
        if relation(h.scope, candidate.scope)[0] == EQUIVALENT == relation(candidate.scope, h.scope)[0]:
            if actionable:
                return "alias", h.id, "same proposition and scope; its tests apply to the existing node"
            return "archive", h.id, "restates an existing hypothesis without a new test"
    for h in same:
        if relation(h.scope, candidate.scope)[0] != DISJOINT:
            if actionable:
                return "refines", h.id, "same proposition in an overlapping scope"
            return "archive", h.id, "overlaps an existing hypothesis without a new test"
    if same:
        if actionable:
            return "derived_from", same[0].id, "same proposition in a disjoint scope"
        return "archive", same[0].id, "same proposition in a new scope without a test"
    if actionable:
        return "new", None, "new testable proposition"
    return "archive", None, "new proposition without evidence or a discriminating plan"


def merge(f: Fork, canonical: Forest) -> dict:
    """Replay the fork's new events onto ``canonical``; return what happened to each."""
    branch = f.forest
    if canonical.log[:f.base_len] != branch.log[:f.base_len]:
        raise InvalidEdit("the fork does not descend from this canonical forest")
    tested = {a.hypothesis for a in branch.assessments}
    report = {"base": f.base, "head_before": canonical.head, "stale_base": canonical.head != f.base,
              "admitted": [], "aliased": [], "archived": [], "skipped": [], "rejected": []}
    alias, dropped = {}, set()
    for event in branch.log[f.base_len:]:
        b = event["body"]
        try:
            if event["kind"] == "record":
                if b["id"] not in canonical.records:
                    canonical.add_record(branch.records[b["id"]])
            elif event["kind"] == "plan":
                if b["id"] not in canonical.plans:
                    canonical.add_plan(b)
            elif b["op"] == "propose":
                d = b["params"]["hypothesis"]
                candidate = Hypothesis(**dict(d, scope=Scope.from_dict(d.get("scope", {}))))
                actionable = candidate.id in tested or any(
                    pid in branch.plans and not plan_errors(branch.plans[pid]) for pid in candidate.plans)
                decision, existing, reason = admission(candidate, canonical, actionable)
                item = {"id": candidate.id, "decision": decision, "existing": existing, "reason": reason}
                if decision == "archive":
                    dropped.add(candidate.id)
                    report["archived"].append(item)
                elif decision == "alias":
                    alias[candidate.id] = existing
                    report["aliased"].append(item)
                else:
                    params = dict(b["params"])
                    if decision in ("refines", "derived_from"):
                        params["link"] = {"kind": decision, "to": existing}
                    elif params.get("link"):
                        params["link"] = dict(params["link"], to=alias.get(params["link"]["to"], params["link"]["to"]))
                    canonical.edit(b["trigger"], "propose", b["targets"], b["reason"], b["evidence"], b["by"], **params)
                    report["admitted"].append(item)
            elif any(t in dropped for t in b["targets"]):
                report["skipped"].append({"seq": event["seq"], "op": b["op"], "targets": b["targets"]})
            else:
                params = dict(b["params"])
                if "survivor" in params:
                    params["survivor"] = alias.get(params["survivor"], params["survivor"])
                targets = [alias.get(t, t) for t in b["targets"]]
                canonical.edit(b["trigger"], b["op"], targets, b["reason"], b["evidence"], b["by"], **params)
        except InvalidEdit as err:
            if event["kind"] == "edit" and b["op"] == "propose":
                dropped.add(b["targets"][0])
            report["rejected"].append({"seq": event["seq"], "kind": event["kind"],
                                       "op": b.get("op"), "error": str(err)})
    report["head_after"] = canonical.head
    return report
