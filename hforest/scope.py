"""Scope: the conditions under which a claim is meant to hold.

A scope maps dimension names to one of

* a scalar          ``"weekly_immersion"``, ``45``
* a finite set      ``["Q345B", "3Ni"]``
* an interval       ``{"min": 0.1, "max": 0.8}`` (inclusive, either end optional)

plus ``exclusions`` (``"dim=value"`` labels the claim rules out) and
``unknowns`` (dimensions the claim has not pinned down).
"""
from __future__ import annotations

from dataclasses import dataclass, field

EQUIVALENT, COVERS, OVERLAPS, UNKNOWN, DISJOINT = "equivalent", "covers", "overlaps", "unknown", "disjoint"
_WORST_FIRST = (DISJOINT, OVERLAPS, UNKNOWN, COVERS, EQUIVALENT)


@dataclass(frozen=True)
class Scope:
    dims: dict = field(default_factory=dict)
    exclusions: tuple = ()
    unknowns: tuple = ()

    def to_dict(self) -> dict:
        return {"dims": dict(self.dims), "exclusions": list(self.exclusions), "unknowns": list(self.unknowns)}

    @classmethod
    def from_dict(cls, data: dict | "Scope") -> "Scope":
        if isinstance(data, Scope):
            return data
        return cls(dict(data.get("dims", {})), tuple(data.get("exclusions", ())), tuple(data.get("unknowns", ())))


def _interval(value):
    if isinstance(value, dict):
        return float(value.get("min", float("-inf"))), float(value.get("max", float("inf")))
    return None


def _points(value) -> frozenset:
    return frozenset(value) if isinstance(value, (list, tuple, set, frozenset)) else frozenset([value])


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def relate_values(claim, target) -> str:
    """How the claim's value on one dimension applies to the target's value."""
    claim_iv, target_iv = _interval(claim), _interval(target)
    if claim_iv and target_iv:
        if claim_iv == target_iv:
            return EQUIVALENT
        if claim_iv[0] <= target_iv[0] and target_iv[1] <= claim_iv[1]:
            return COVERS
        return OVERLAPS if claim_iv[0] <= target_iv[1] and target_iv[0] <= claim_iv[1] else DISJOINT
    if claim_iv or target_iv:
        low, high = claim_iv or target_iv
        points = _points(target if claim_iv else claim)
        if not all(_is_number(p) for p in points):
            return DISJOINT
        inside = [low <= p <= high for p in points]
        if claim_iv:
            return COVERS if all(inside) else OVERLAPS if any(inside) else DISJOINT
        if low == high and low in points:
            return COVERS
        return OVERLAPS if any(inside) else DISJOINT
    claim_set, target_set = _points(claim), _points(target)
    if claim_set == target_set:
        return EQUIVALENT
    if target_set <= claim_set:
        return COVERS
    return OVERLAPS if claim_set & target_set else DISJOINT


def relation(claim: Scope, target: Scope) -> tuple[str, dict]:
    """Return (overall relation, per-dimension relations) of a claim scope to a target.

    A dimension only the claim constrains is UNKNOWN: the target did not say
    whether it matches. A dimension only the target constrains is COVERED,
    unless the claim declared it unknown. The overall relation is the worst
    per-dimension relation.
    """
    per = {}
    for name in set(claim.dims) | set(target.dims):
        if name in claim.dims and name in target.dims:
            per[name] = relate_values(claim.dims[name], target.dims[name])
        elif name in claim.dims or name in claim.unknowns:
            per[name] = UNKNOWN
        else:
            per[name] = COVERS
    for name in target.unknowns:
        if name in claim.dims:
            per[name] = UNKNOWN
    labels = {f"{name}={p}" for name, v in target.dims.items() if not isinstance(v, dict) for p in _points(v)}
    if labels & set(claim.exclusions):
        per["<exclusions>"] = DISJOINT
    overall = next((r for r in _WORST_FIRST if r in per.values()), EQUIVALENT)
    return overall, per


def covers(claim: Scope, target: Scope) -> bool:
    return relation(claim, target)[0] in (EQUIVALENT, COVERS)


def bridge_dims(per: dict) -> list[str]:
    """Dimensions a transfer would have to bridge with target-scope tests."""
    return sorted(name for name, r in per.items() if r in (OVERLAPS, UNKNOWN, DISJOINT))
