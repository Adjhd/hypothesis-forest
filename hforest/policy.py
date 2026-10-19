"""Choose the next experiment from the forest's competition structure.

The forest steers what to test only through declared rivals: active hypotheses
linked by ``competes`` relations. An experiment is worth running in proportion
to how evenly the rivals that apply to its condition disagree on its outcome
(the expected information gain under a uniform prior and noise-free outcomes).
"""
from __future__ import annotations

import math
from collections import Counter

from .scope import covers

ACTIVE = {"proposed", "tested", "supported", "scoped", "revised", "reactivated"}


def rival_groups(forest) -> list[list[str]]:
    """Connected components of the competes graph over active hypotheses."""
    alive = {i for i, h in forest.hypotheses.items() if h.status in ACTIVE and not h.merged_into}
    root = {i: i for i in alive}

    def find(i):
        while root[i] != i:
            i = root[i]
        return i
    for r in forest.relations:
        if r["kind"] == "competes" and r["src"] in alive and r["dst"] in alive:
            root[find(r["src"])] = find(r["dst"])
    groups: dict[str, list[str]] = {}
    for i in sorted(alive):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def disagreement(outcomes) -> float:
    """Entropy in bits of the predicted outcomes; 0 when all rivals agree."""
    counts = Counter(outcomes)
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values()) if n else 0.0


def choose_experiment(forest, experiments: dict, predict, cost=None) -> tuple:
    """Return (experiment id, bits per unit cost) that best separates rivals, or (None, 0.0).

    ``experiments`` maps ids to condition Scopes; ``predict(hypothesis, experiment_id)``
    returns that hypothesis's predicted outcome. Rivals whose scope does not cover
    the condition make no prediction there.
    """
    groups, best, best_score = rival_groups(forest), None, 0.0
    for eid, condition in experiments.items():
        bits = sum(disagreement([predict(forest.hypotheses[i], eid) for i in g
                                 if covers(forest.hypotheses[i].scope, condition)]) for g in groups)
        score = bits / (cost(eid) if cost else 1.0)
        if score > best_score:
            best, best_score = eid, score
    return best, best_score
