"""Role-based prior retrieval for a new task.

Content keys (variables, endpoints, mechanism) decide whether a hypothesis is
relevant; its status and scope relation to the target decide its role. Only a
supported hypothesis whose *authorized* scope covers the target is a direct
prior. Every other card comes with the dimensions it would have to bridge and
never passes its conclusion on automatically.
"""
from __future__ import annotations

from .forest import Forest, Hypothesis, content_key
from .scope import COVERS, DISJOINT, EQUIVALENT, Scope, bridge_dims, covers, relation

ROLES = ("direct_prior", "negative_prior", "open_discrimination", "transfer_candidate",
         "cross_domain_transfer", "in_scope_unvalidated", "dormant")
_OPEN = ("proposed", "tested", "revised", "reactivated")


def classify(h: Hypothesis, forest: Forest, target: Scope, keys: set, domain_dims=()) -> dict | None:
    if h.status == "retired" or h.merged_into:
        return None
    mechanism, variables, endpoints = content_key(h)
    hits = keys & ({mechanism} | set(variables) | set(endpoints))
    if keys and not hits:
        return None
    rel, per = relation(h.scope, target)
    competing = any(r["kind"] == "competes" and h.id in (r["src"], r["dst"]) for r in forest.relations)
    if rel == DISJOINT:
        if not keys:
            return None
        crossed = {d for d, r in per.items() if r == DISJOINT} & set(domain_dims)
        role = "cross_domain_transfer" if crossed else "transfer_candidate"
    elif h.status == "contradicted":
        role = "negative_prior"
    elif h.status == "frozen":
        role = "dormant"
    elif rel in (EQUIVALENT, COVERS):
        if h.status in ("supported", "scoped") and h.authorized_scope and covers(h.authorized_scope, target):
            role = "direct_prior"
        elif competing and h.status in _OPEN:
            role = "open_discrimination"
        else:
            role = "in_scope_unvalidated"
    else:
        role = "transfer_candidate"
    admitted = [a for a in forest.assessments if a.hypothesis == h.id
                and any(forest.records[e].admitted for e in a.evidence)]
    score = len(hits) + 2 * sum(a.verdict == "supports" for a in admitted) + sum(a.verdict == "contradicts" for a in admitted)
    return {"id": h.id, "role": role, "status": h.status, "relation": rel, "statement": h.statement,
            "bridge": [] if role == "direct_prior" else bridge_dims(per),
            "inherits_conclusion": role == "direct_prior", "score": score}


def retrieve(forest: Forest, target: Scope, *, variables=(), endpoints=(), mechanism="",
             domain_dims=(), max_cards: int = 12) -> list[dict]:
    """A bounded, role-diverse prior pack: first the best card of each role, then by score."""
    probe = Hypothesis("query", "query", target, ["-"], ["-"], mechanism, list(variables), list(endpoints))
    m, v, e = content_key(probe)
    keys = ({m} if m else set()) | set(v) | set(e)
    cards = [c for h in forest.hypotheses.values() if (c := classify(h, forest, target, keys, domain_dims))]
    cards.sort(key=lambda c: (ROLES.index(c["role"]), -c["score"], c["id"]))
    first = [next(c for c in cards if c["role"] == role) for role in ROLES if any(c["role"] == role for c in cards)]
    rest = [c for c in cards if c not in first]
    return (first + rest)[:max_cards]
