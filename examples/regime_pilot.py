"""Pilot regime map: when do the forest's three decision mechanisms pay off?

Toy world: K conditions, dose x in 0..5, outcome is high iff x >= t_c. The rivals
are the seven thresholds t = 0..6; with probability p_dep a condition's true
threshold is redrawn instead of sharing the common one.

Mechanisms, each of which can be switched off:
  D  discriminate: run the (x, c) whose outcome best splits the surviving rivals; else random
  G  gate failures: a run flagged by the ops log is kept on record but cannot eliminate
  S  split on conflict: rivals are pooled across conditions until no pooled threshold fits
     every usable record, then re-judged per condition from the same retained records

A failed run (probability q) reads "low". The ops flag catches it with recall 0.9 and
false-alarms on 5% of good runs, so gating has a price. Splitting has one too: per-condition
rivals need more runs than pooled ones. This is a rule-based pilot, not paper evidence.

Run:  python examples/regime_pilot.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hforest.policy import disagreement  # noqa: E402

K, DOSES, THRESHOLDS, BUDGET, SEEDS = 3, range(6), range(7), 18, 400
RECALL, FALSE_ALARM = 0.9, 0.05
ARMS = {"forest D+G+S": "DGS", "- discriminate": "GS", "- failure gate": "DS", "- split": "DG", "flat (none)": ""}


def survivors(records, split):
    if split:
        return [{t for t in THRESHOLDS if all((x >= t) == high for c, x, high in records if c == k)} for k in range(K)]
    pooled = {t for t in THRESHOLDS if all((x >= t) == high for _, x, high in records)}
    return [pooled] * K


def episode(arm, truth, q, rng):
    """Return the number of runs needed to identify every condition's threshold, or None."""
    records, split = [], False
    for runs in range(BUDGET + 1):
        alive = survivors(records, split)
        if "S" in arm and not split and not alive[0]:
            split, alive = True, survivors(records, True)
        if all(alive[c] == {truth[c]} for c in range(K)):
            return runs
        if runs == BUDGET:
            return None
        options = [(c, x) for c in range(K) for x in DOSES]
        rng.shuffle(options)
        if "D" in arm:
            options.sort(key=lambda o: -disagreement([o[1] >= t for t in alive[o[0]]]))
        c, x = options[0]
        failed = rng.random() < q
        flagged = rng.random() < (RECALL if failed else FALSE_ALARM)
        if not ("G" in arm and flagged):
            records.append((c, x, x >= truth[c] and not failed))


def main():
    grid = [(q, p) for q in (0.0, 0.1, 0.2) for p in (0.0, 0.5, 1.0)]
    print(f"success within {BUDGET} runs over {SEEDS} worlds (median runs when solved); K={K}")
    print(f"{'arm':16}" + "".join(f"  q={q:.1f} dep={p:.1f}" for q, p in grid))
    for name, arm in ARMS.items():
        cells = []
        for q, p in grid:
            rng = random.Random(f"{q}-{p}")
            results = []
            for _ in range(SEEDS):
                base = rng.choice(THRESHOLDS)
                truth = [rng.choice(THRESHOLDS) if rng.random() < p else base for _ in range(K)]
                results.append(episode(arm, truth, q, random.Random(rng.random())))
            solved = sorted(r for r in results if r is not None)
            median = solved[len(solved) // 2] if solved else "-"
            cells.append(f"{len(solved) / SEEDS:>8.0%} ({median:>2})")
        print(f"{name:16}" + "".join(f"  {c:>15}" for c in cells))


if __name__ == "__main__":
    main()
