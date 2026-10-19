"""Paired promotion gate: the only way an authorized scope can widen."""
from __future__ import annotations

from math import comb


def paired_gate(pairs, *, alpha: float = 0.05, max_losses: int = 0, min_discordant: int = 1) -> dict:
    """One-sided exact sign test on paired ``(baseline_ok, candidate_ok)`` outcomes.

    Ties carry no information and are ignored. The candidate passes only with
    more wins than losses, at most ``max_losses`` regressions and p <= alpha.
    """
    pairs = [(bool(b), bool(c)) for b, c in pairs]
    wins = sum(c and not b for b, c in pairs)
    losses = sum(b and not c for b, c in pairs)
    n = wins + losses
    p_value = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0
    accepted = n >= min_discordant and wins > losses and losses <= max_losses and p_value <= alpha
    return {"wins": wins, "losses": losses, "ties": len(pairs) - n, "p_value": p_value,
            "alpha": alpha, "accepted": accepted}
