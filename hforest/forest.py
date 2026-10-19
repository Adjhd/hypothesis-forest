"""The forest object F = (H, A, R, Lambda, E) and its evidence-gated growth rules.

Every change is one committed event: a record (R), a verification plan (A)
or a typed edit that answers a declared trigger. Hypotheses (H), assessments
(Lambda) and relations (E) change only through edits. The event log is
append-only and the forest is its replay; each event names its parent head,
like a git commit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .scope import DISJOINT, OVERLAPS, UNKNOWN, Scope, covers, relation

# Lifecycle: which status may follow which.
TRANSITIONS = {
    "proposed": {"tested", "blocked", "retired"},
    "tested": {"supported", "contradicted", "scoped", "revised", "frozen", "blocked", "retired"},
    "supported": {"scoped", "revised", "frozen", "reactivated", "retired"},
    "contradicted": {"scoped", "revised", "reactivated", "retired"},
    "scoped": {"tested", "revised", "frozen", "reactivated", "retired"},
    "revised": {"tested", "blocked", "retired"},
    "frozen": {"reactivated", "retired"},
    "reactivated": {"tested", "blocked", "frozen", "retired"},
    "blocked": {"reactivated", "retired"},
    "retired": set(),
}

# Trigger -> (allowed operations, reachable statuses, must cite admitted evidence).
# An empty status set means the edit cannot change scientific status.
RULES = {
    "unexplained_observation": ({"propose"}, set(), False),
    "refined_content": ({"revise"}, {"revised"}, False),
    "admitted_support_in_scope": ({"attach_evidence"}, {"tested", "supported"}, True),
    "admitted_contradiction_in_scope": ({"attach_evidence", "narrow_scope"}, {"tested", "contradicted", "scoped"}, True),
    "condition_dependent_outcome": ({"split", "narrow_scope"}, {"scoped"}, True),
    "unresolved_competition": ({"link_competition"}, set(), False),
    "equivalent_hypotheses": ({"merge"}, set(), False),
    "invalid_experiment": ({"record_event"}, set(), False),
    "execution_blocked": ({"record_event"}, {"blocked"}, False),
    "stale_without_budget": ({"suspend"}, {"frozen"}, False),
    "relevant_new_evidence": ({"reactivate"}, {"reactivated"}, False),
    "cross_scope_request": ({"propose_transfer"}, set(), False),
    "promotion_passed": ({"authorize"}, set(), True),
    "obsolete": ({"retire"}, {"retired"}, False),
}
IMPLIED_STATUS = {"revise": "revised", "narrow_scope": "scoped", "split": "scoped",
                  "suspend": "frozen", "reactivate": "reactivated", "retire": "retired"}

# E0 context lead, E1 model output, E2 controlled replay, E3 external validation, E4 admitted.
TIERS = ("E0", "E1", "E2", "E3", "E4")
RECORD_KINDS = ("scientific", "engineering", "infrastructure")
VERDICTS = ("supports", "contradicts", "inconclusive")
APPLICABILITY = ("applicable", "partially_applicable", "not_applicable")
PLAN_FIELDS = ("objective", "protocol", "intervention", "controls", "observables", "falsifiers")
PROPOSE_FIELDS = ("id", "statement", "scope", "predictions", "falsifiers", "mechanism", "variables", "endpoints", "plans")


class InvalidEdit(ValueError):
    """The event violates the forest rules and was not committed."""


@dataclass
class Hypothesis:
    id: str
    statement: str
    scope: Scope
    predictions: list
    falsifiers: list
    mechanism: str = ""
    variables: list = field(default_factory=list)
    endpoints: list = field(default_factory=list)
    plans: list = field(default_factory=list)  # discriminating verification plan ids
    status: str = "proposed"
    revision: int = 1
    authorized_scope: Scope | None = None  # None: may not drive candidate design
    merged_into: str | None = None
    history: list = field(default_factory=list)


@dataclass(frozen=True)
class Record:
    id: str
    statement: str
    scope: Scope
    source: str
    tier: str = "E0"
    kind: str = "scientific"
    admits: str | None = None  # an E4 record admits one E2/E3 record

    @property
    def admitted(self) -> bool:
        return self.tier == "E4"


@dataclass(frozen=True)
class Assessment:
    hypothesis: str
    revision: int
    evidence: tuple
    target_scope: Scope
    verdict: str
    applicability: str
    by: str


def content_key(h: Hypothesis) -> tuple:
    """Identity of a proposition independent of its scope."""
    def norm(text: str) -> str:
        return " ".join(str(text).lower().replace("_", " ").replace("-", " ").split())
    return norm(h.mechanism), tuple(sorted({norm(v) for v in h.variables})), tuple(sorted({norm(v) for v in h.endpoints}))


def plan_errors(plan: dict) -> list[str]:
    errors = [f"plan needs {name}" for name in ("id", *PLAN_FIELDS) if not plan.get(name)]
    if plan.get("status", "planned") != "planned":
        errors.append("a plan is a declared design, its status stays planned")
    return errors


def _jsonable(value):
    if isinstance(value, Scope):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class Forest:
    def __init__(self):
        self.hypotheses: dict[str, Hypothesis] = {}
        self.records: dict[str, Record] = {}
        self.plans: dict[str, dict] = {}
        self.assessments: list[Assessment] = []
        self.relations: list[dict] = []
        self.log: list[dict] = []
        self.head = "root"

    # ------------------------------------------------------------------ API
    def add_record(self, record: Record) -> str:
        body = _jsonable(record.__dict__)
        return self._commit("record", body, self.record_errors(record))

    def add_plan(self, plan: dict) -> str:
        plan = {"status": "planned", **plan}
        errors = plan_errors(plan) + ([f"plan exists: {plan['id']}"] if plan.get("id") in self.plans else [])
        return self._commit("plan", _jsonable(plan), errors)

    def edit(self, trigger: str, op: str, targets, reason: str, evidence=(), by: str = "controller", **params) -> str:
        body = {"trigger": trigger, "op": op, "targets": list(targets), "reason": reason,
                "evidence": list(evidence), "by": by, "params": _jsonable(params)}
        return self._commit("edit", body, self.edit_errors(body))

    def copy(self) -> "Forest":
        return Forest.from_events(self.log)

    def save(self, path) -> None:
        Path(path).write_text("".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in self.log))

    @classmethod
    def load(cls, path) -> "Forest":
        return cls.from_events(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())

    @classmethod
    def from_events(cls, events) -> "Forest":
        """Replay a log; every event is re-validated and its head re-derived."""
        forest = cls()
        for event in events:
            if event["parent"] != forest.head:
                raise InvalidEdit(f"event {event['seq']} has parent {event['parent']}, expected {forest.head}")
            if event["kind"] == "record":
                body = dict(event["body"], scope=Scope.from_dict(event["body"]["scope"]))
                head = forest.add_record(Record(**body))
            elif event["kind"] == "plan":
                head = forest.add_plan(event["body"])
            else:
                b = event["body"]
                head = forest.edit(b["trigger"], b["op"], b["targets"], b["reason"], b["evidence"], b["by"], **b["params"])
            if head != event["head"]:
                raise InvalidEdit(f"event {event['seq']} replays to {head}, log says {event['head']}")
        return forest

    # ----------------------------------------------------------- validation
    def record_errors(self, r: Record) -> list[str]:
        errors = []
        if r.id in self.records:
            errors.append(f"record exists: {r.id}")
        if not (r.statement and r.source):
            errors.append("a record needs a statement and a source")
        if r.tier not in TIERS or r.kind not in RECORD_KINDS:
            errors.append(f"unknown tier/kind: {r.tier}/{r.kind}")
        if r.admitted:
            raw = self.records.get(r.admits)
            if r.kind != "scientific":
                errors.append("only scientific records can be admitted")
            if raw is None or raw.tier not in ("E2", "E3"):
                errors.append("E4 must admit an existing E2/E3 record; E0/E1 cannot be admitted")
        return errors

    def edit_errors(self, body: dict) -> list[str]:
        trigger, op, targets, p = body["trigger"], body["op"], body["targets"], body["params"]
        if trigger not in RULES:
            return [f"unknown trigger: {trigger}"]
        ops, reachable, needs_admitted = RULES[trigger]
        errors = []
        if op not in ops:
            errors.append(f"{op} is not allowed for {trigger}; allowed: {sorted(ops)}")
        if not body["reason"].strip():
            errors.append("an edit needs a reason")
        status = p.get("status") or IMPLIED_STATUS.get(op)
        if status and status not in reachable:
            errors.append(f"{trigger} cannot move a hypothesis to {status}")
        errors += [f"unknown record: {e}" for e in body["evidence"] if e not in self.records]
        if needs_admitted:
            if not body["evidence"]:
                errors.append(f"{trigger} must cite admitted evidence")
            errors += [f"not admitted: {e}" for e in body["evidence"] if e in self.records and not self.records[e].admitted]
        if op == "propose":
            return errors + self._check_propose(targets, p)
        if not targets:
            return errors + ["an edit needs targets"]
        for t in targets:
            h = self.hypotheses.get(t)
            if h is None:
                errors.append(f"unknown hypothesis: {t}")
            elif h.merged_into:
                errors.append(f"{t} was merged into {h.merged_into}")
            elif status and status != h.status and self._path(h.status, status, op) is None:
                errors.append(f"{t}: {h.status} -> {status} is not a lifecycle transition")
        if errors:
            return errors
        check = getattr(self, f"_check_{op}", None)
        return check(targets, p, body) if check else errors

    def _check_propose(self, targets, p) -> list[str]:
        d = p.get("hypothesis", {})
        errors = [f"hypothesis needs {k}" for k in ("id", "statement", "predictions", "falsifiers") if not d.get(k)]
        errors += [f"a new hypothesis cannot set {k}" for k in d if k not in PROPOSE_FIELDS]
        if targets != [d.get("id")]:
            errors.append("propose targets exactly the new hypothesis id")
        if d.get("id") in self.hypotheses:
            errors.append(f"hypothesis exists: {d.get('id')}")
        errors += [f"unknown plan: {a}" for a in d.get("plans", []) if a not in self.plans]
        link = p.get("link")
        if link and (link.get("kind") not in ("derived_from", "refines") or link.get("to") not in self.hypotheses):
            errors.append("link must be derived_from/refines to an existing hypothesis")
        return errors

    def _check_revise(self, targets, p, body):
        return [] if any(k in p for k in ("statement", "predictions", "falsifiers", "mechanism")) else ["revise changes no content"]

    def _check_attach_evidence(self, targets, p, body):
        h = self.hypotheses[targets[0]]
        verdict, applicability = p.get("verdict"), p.get("applicability", "applicable")
        errors = [] if len(targets) == 1 else ["one assessment judges one hypothesis"]
        if verdict not in VERDICTS or applicability not in APPLICABILITY:
            return errors + [f"bad verdict/applicability: {verdict}/{applicability}"]
        if not body["evidence"]:
            errors.append("an assessment must cite evidence")
        if applicability == "not_applicable" and verdict != "inconclusive":
            errors.append("not-applicable evidence can only be inconclusive")
        expected = {"admitted_support_in_scope": "supports", "admitted_contradiction_in_scope": "contradicts"}
        if expected.get(body["trigger"], verdict) != verdict:
            errors.append(f"{body['trigger']} needs a {expected[body['trigger']]} verdict")
        target_scope = Scope.from_dict(p.get("target_scope") or h.scope)
        for eid in body["evidence"]:
            record = self.records[eid]
            if verdict == "inconclusive":
                continue
            if record.kind != "scientific":
                errors.append(f"{eid}: {record.kind} records cannot support or contradict")
            rel, _ = relation(record.scope, target_scope)
            if rel == DISJOINT:
                errors.append(f"{eid}: evidence scope is disjoint from the assessed scope")
            elif rel in (OVERLAPS, UNKNOWN) and not p.get("scope_review", {}).get(eid):
                errors.append(f"{eid}: evidence scope {rel}; needs an explicit scope_review")
        return errors

    def _check_narrow_scope(self, targets, p, body):
        h, new = self.hypotheses[targets[0]], Scope.from_dict(p.get("new_scope", {}))
        return [] if covers(h.scope, new) and new != h.scope else ["new_scope must be strictly inside the current scope"]

    def _check_split(self, targets, p, body):
        h, dim, children = self.hypotheses[targets[0]], p.get("dimension"), p.get("children", [])
        errors = [] if dim and len(children) >= 2 else ["split needs a dimension and at least two children"]
        for c in children:
            scope = Scope.from_dict(c.get("scope", {}))
            if not c.get("id") or c["id"] in self.hypotheses:
                errors.append(f"child needs a new id: {c.get('id')}")
            if not covers(h.scope, scope) or dim not in scope.dims:
                errors.append(f"child {c.get('id')} must lie inside the parent and fix {dim}")
        values = [json.dumps(c.get("scope", {}).get("dims", {}).get(dim), sort_keys=True) for c in children]
        return errors + ([] if len(set(values)) == len(values) else [f"children must differ on {dim}"])

    def _check_link_competition(self, targets, p, body):
        return [] if len(set(targets)) >= 2 and p.get("observable") else ["competition links >=2 hypotheses on an observable"]

    def _check_merge(self, targets, p, body):
        keys = {content_key(self.hypotheses[t]) for t in targets}
        errors = [] if len(set(targets)) >= 2 and p.get("survivor") in targets else ["merge needs >=2 targets and a survivor among them"]
        return errors + ([] if len(keys) == 1 else ["only hypotheses with the same content key are equivalent"])

    def _check_propose_transfer(self, targets, p, body):
        h, errors = self.hypotheses[targets[0]], []
        if Scope.from_dict(p.get("target_scope", {})) == h.scope:
            errors.append("a transfer needs a different target scope")
        if p.get("kind") not in ("mechanism", "method", "property"):
            errors.append("transfer kind is mechanism, method or property")
        if p.get("bridge_plan") not in self.plans:
            errors.append("a transfer needs a declared bridge plan")
        if p.get("no_automatic_inheritance") is not True:
            errors.append("a transfer must declare no_automatic_inheritance=True")
        return errors

    def _check_authorize(self, targets, p, body):
        h, errors = self.hypotheses[targets[0]], []
        if h.status not in ("supported", "scoped"):
            errors.append("only supported or scoped hypotheses can be authorized")
        if not (p.get("decision") or {}).get("accepted"):
            errors.append("authorization needs an accepted promotion decision")
        if not covers(h.scope, Scope.from_dict(p.get("authorized_scope", {}))):
            errors.append("authorized scope must lie inside the validity scope")
        return errors

    def _check_record_event(self, targets, p, body):
        return [] if p.get("event_class") else ["record_event needs an event_class"]

    @staticmethod
    def _path(current: str, target: str, op: str):
        if target in TRANSITIONS[current]:
            return [target]
        if op == "attach_evidence" and "tested" in TRANSITIONS[current] and target in TRANSITIONS["tested"]:
            return ["tested", target]  # an assessment first makes the hypothesis tested
        return None

    # -------------------------------------------------------------- commit
    def _commit(self, kind: str, body: dict, errors: list[str]) -> str:
        if errors:
            raise InvalidEdit("; ".join(errors))
        text = json.dumps({"kind": kind, "body": body}, sort_keys=True, ensure_ascii=False)
        head = hashlib.sha256((self.head + text).encode()).hexdigest()[:12]
        if kind == "record":
            self.records[body["id"]] = Record(**dict(body, scope=Scope.from_dict(body["scope"])))
        elif kind == "plan":
            self.plans[body["id"]] = body
        else:
            self._apply(body, head)
        self.log.append({"seq": len(self.log) + 1, "parent": self.head, "head": head, "kind": kind, "body": body})
        self.head = head
        return head

    def _apply(self, b: dict, head: str) -> None:
        op, p, targets = b["op"], b["params"], b["targets"]
        status = p.get("status") or IMPLIED_STATUS.get(op)
        if op == "propose":
            d = p["hypothesis"]
            self.hypotheses[d["id"]] = Hypothesis(**dict(d, scope=Scope.from_dict(d.get("scope", {}))),
                                                  history=[{"event": head, "to": "proposed"}])
            if p.get("link"):
                self.relations.append({"kind": p["link"]["kind"], "src": d["id"], "dst": p["link"]["to"], "event": head})
            return
        h = self.hypotheses[targets[0]]
        if op == "revise":
            for k in ("statement", "predictions", "falsifiers", "mechanism"):
                if k in p:
                    setattr(h, k, p[k])
            h.revision += 1
        elif op == "attach_evidence":
            self.assessments.append(Assessment(h.id, h.revision, tuple(b["evidence"]),
                                               Scope.from_dict(p.get("target_scope") or h.scope),
                                               p["verdict"], p.get("applicability", "applicable"), b["by"]))
        elif op == "narrow_scope":
            h.scope, h.revision = Scope.from_dict(p["new_scope"]), h.revision + 1
        elif op == "split":
            for c in p["children"]:
                child = Hypothesis(c["id"], c.get("statement", h.statement), Scope.from_dict(c["scope"]),
                                   list(h.predictions), list(h.falsifiers), h.mechanism, list(h.variables),
                                   list(h.endpoints), history=[{"event": head, "to": "proposed"}])
                self.hypotheses[child.id] = child
                self.relations.append({"kind": "refines", "src": child.id, "dst": h.id, "event": head})
        elif op == "link_competition":
            ids = sorted(set(targets))
            self.relations += [{"kind": "competes", "src": a, "dst": z, "observable": p["observable"], "event": head}
                               for i, a in enumerate(ids) for z in ids[i + 1:]]
        elif op == "merge":
            for t in targets:
                if t != p["survivor"]:
                    self.hypotheses[t].merged_into = p["survivor"]
                    self.relations.append({"kind": "merged_into", "src": t, "dst": p["survivor"], "event": head})
        elif op == "propose_transfer":
            self.relations.append({"kind": "transfer_candidate", "src": h.id, "event": head, "transfer_kind": p["kind"],
                                   **{k: p[k] for k in ("target_scope", "bridge_plan", "no_automatic_inheritance")}})
        elif op == "authorize":
            h.authorized_scope = Scope.from_dict(p["authorized_scope"])
        elif op == "record_event":
            h.history.append({"event": head, "note": p["event_class"], "reason": b["reason"]})
        for t in targets if status else ():
            node = self.hypotheses[t]
            for step in (self._path(node.status, status, op) or []) if status != node.status else []:
                node.history.append({"event": head, "from": node.status, "to": step, "evidence": b["evidence"]})
                node.status = step
