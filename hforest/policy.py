"""Choose the next experiment from the forest's competition structure.

The forest steers what to test only through declared rivals: active hypotheses
linked by ``competes`` relations. An experiment is worth running in proportion
to the information its outcome carries about which rival is right: the mutual
information between rival and outcome. Predictions may be exact outcomes or
distributions over outcomes (an observation model with failures or noise), and
rivals may carry credence weights; with exact predictions and equal weights this
is the entropy of the rivals' predictions.
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


def _entropy(probs) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def disagreement(outcomes) -> float:
    """Entropy in bits of the predicted outcomes; 0 when all rivals agree."""
    counts = Counter(outcomes)
    n = sum(counts.values())
    return _entropy(c / n for c in counts.values()) if n else 0.0


def information(predictions, weights=None) -> float:
    """Mutual information in bits between rival and outcome.

    Each prediction is an outcome or a dict {outcome: probability}; ``weights`` are
    unnormalized credences (equal if omitted).
    """
    dists = [p if isinstance(p, dict) else {p: 1.0} for p in predictions]
    weights = list(weights) if weights is not None else [1.0] * len(dists)
    z = sum(weights)
    if not z:
        return 0.0
    mix = Counter()
    for d, w in zip(dists, weights):
        for outcome, p in d.items():
            mix[outcome] += w / z * p
    return _entropy(mix.values()) - sum(w / z * _entropy(d.values()) for d, w in zip(dists, weights))


def choose_experiment(forest, experiments: dict, predict, cost=None, consider=None, weight=None) -> tuple:
    """Return (experiment id, bits per unit cost) that best separates rivals, or (None, 0.0).

    ``experiments`` maps ids to condition Scopes; ``predict(hypothesis, experiment_id)``
    returns that hypothesis's predicted outcome or outcome distribution. Rivals whose
    scope does not cover the condition make no prediction there. ``consider(hypothesis,
    experiment_id)`` can drop rivals the caller already doubts; ``weight(hypothesis,
    experiment_id)`` gives the caller's credence in a rival (it never changes the forest).
    """
    groups, best, best_score = rival_groups(forest), None, 0.0
    for eid, condition in experiments.items():
        bits = 0.0
        for g in groups:
            hs = [forest.hypotheses[i] for i in g if covers(forest.hypotheses[i].scope, condition)
                  and (consider is None or consider(forest.hypotheses[i], eid))]
            bits += information([predict(h, eid) for h in hs], [weight(h, eid) for h in hs] if weight else None)
        score = bits / (cost(eid) if cost else 1.0)
        if score > best_score + 1e-12:
            best, best_score = eid, score
    return best, best_score
