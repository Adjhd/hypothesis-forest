"""Threshold world: a controlled discovery task with known ground truth.

Three conditions. In condition c a run at dose x in 0..5 reads "high" iff x >= t_c, with
t_c in 0..6 unknown; with probability ``dep`` a condition's threshold is redrawn instead
of sharing a common one. A run fails with probability ``q`` and then reads "low"; the
ops alarm catches failures with recall 0.9 and fires on 5% of good runs.

Agents:
  forest    the kernel alone. The seven thresholds are competing hypotheses over all
            conditions; an admitted run that contradicts one narrows its scope away from
            that condition; alarmed runs are recorded but cannot refute; the next run is
            the one that best separates the surviving rivals. No language model.
  flat      a language model reading the full run history.
  assisted  the same model, also shown the forest's state and suggestion.

Every model decision is scored against what the admitted evidence allows, for three
errors: a non-discriminating run, a failure used as refutation, and an exclusion that
rests only on other conditions (overgeneralization). Runs use common random numbers:
the k-th run of a world fails or alarms identically in every arm.

  python experiments/threshold_world.py --worlds 200          # kernel agent only
Model agents need a ``call(system, payload) -> dict`` (see ``openai_call``).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hforest import Forest, Record, Scope, choose_experiment  # noqa: E402
from hforest.policy import ACTIVE, disagreement  # noqa: E402

CONDITIONS, DOSES, THRESHOLDS = ("c0", "c1", "c2"), range(6), range(7)
RECALL, FALSE_ALARM, BUDGET = 0.9, 0.05, 18
CELLS = {"easy": (0.0, 0.0), "dependent": (0.0, 1.0), "failures": (0.2, 0.0), "both": (0.2, 1.0)}


# ------------------------------------------------------------------ world
def make_world(seed, dep):
    rng = random.Random(f"world-{seed}")
    base = rng.choice(THRESHOLDS)
    return {c: (rng.choice(THRESHOLDS) if rng.random() < dep else base) for c in CONDITIONS}


def run_once(truth, c, x, q, seed, k):
    rng = random.Random(f"run-{seed}-{k}")
    failed = rng.random() < q
    alarm = rng.random() < (RECALL if failed else FALSE_ALARM)
    return {"run": k, "condition": c, "dose": x, "reading": "high" if x >= truth[c] and not failed else "low",
            "ops_alarm": alarm}


def allowed(runs, c, with_alarmed=False):
    """Thresholds consistent with the (unalarmed, or all) runs in condition c."""
    return {t for t in THRESHOLDS if all((r["dose"] >= t) == (r["reading"] == "high") for r in runs
                                         if r["condition"] == c and (with_alarmed or not r["ops_alarm"]))}


# ----------------------------------------------------------- kernel agent
class ThresholdForest:
    """The kernel as the agent's state: one hypothesis per threshold, scoped by condition."""

    def __init__(self):
        f = self.forest = Forest()
        f.add_plan({"id": "P-dose", "objective": "locate the threshold", "protocol": "one run at a condition and dose",
                    "intervention": "dose 0..5", "controls": ["dose 0", "dose 5"], "observables": ["reading"],
                    "falsifiers": ["reading opposite to the prediction"]})
        for t in THRESHOLDS:
            f.edit("unexplained_observation", "propose", [f"t{t}"], "candidate threshold", hypothesis={
                "id": f"t{t}", "statement": f"reading is high iff dose >= {t}",
                "scope": Scope({"condition": list(CONDITIONS)}), "predictions": [f"high iff dose >= {t}"],
                "falsifiers": ["a reading opposite to the prediction"], "mechanism": f"threshold {t}",
                "variables": ["dose"], "endpoints": ["reading"], "plans": ["P-dose"]})
        f.edit("unresolved_competition", "link_competition", [f"t{t}" for t in THRESHOLDS], "rival thresholds",
               observable="reading")

    def observe(self, run):
        f, c, rid = self.forest, run["condition"], f"run{run['run']}"
        here = Scope({"condition": c})
        f.add_record(Record(rid, json.dumps(run), here, "threshold-world", tier="E3",
                            kind="infrastructure" if run["ops_alarm"] else "scientific"))
        if run["ops_alarm"]:
            return
        f.add_record(Record(rid + "+", "no ops alarm", here, "ops-log", tier="E4", admits=rid))
        for h in list(f.hypotheses.values()):
            t, conds = int(h.id[1:]), h.scope.dims["condition"]
            if h.status not in ACTIVE or c not in conds or (run["dose"] >= t) == (run["reading"] == "high"):
                continue
            rest = [k for k in conds if k != c]
            if not rest:
                f.edit("admitted_contradiction_in_scope", "attach_evidence", [h.id], "reading contradicts", [rid + "+"],
                       verdict="contradicts", status="contradicted")
                continue
            f.edit("admitted_contradiction_in_scope", "attach_evidence", [h.id], f"contradicted in {c}", [rid + "+"],
                   verdict="contradicts", target_scope=here, status="tested")
            f.edit("admitted_contradiction_in_scope", "narrow_scope", [h.id], f"fails in {c}", [rid + "+"],
                   new_scope=Scope({"condition": rest}))

    def alive(self, c):
        return {int(h.id[1:]) for h in self.forest.hypotheses.values()
                if h.status in ACTIVE and c in h.scope.dims["condition"]}

    def suggest(self):
        options = {f"{c}:{x}": Scope({"condition": c, "dose": x}) for c in CONDITIONS for x in DOSES}
        eid, bits = choose_experiment(self.forest, options, lambda h, e: int(e.split(":")[1]) >= int(h.id[1:]))
        return (None, 0.0) if eid is None else ({"condition": eid.split(":")[0], "dose": int(eid.split(":")[1])}, bits)

    def view(self, runs):
        nxt, bits = self.suggest()
        return {"surviving_thresholds": {c: sorted(self.alive(c)) for c in CONDITIONS},
                "recorded_but_not_refuting": [r["run"] for r in runs if r["ops_alarm"]],
                "suggested_next": nxt and dict(nxt, bits=round(bits, 3))}


def forest_episode(seed, q, dep):
    truth, state, runs = make_world(seed, dep), ThresholdForest(), []
    while len(runs) < BUDGET and not all(len(state.alive(c)) == 1 for c in CONDITIONS):
        nxt, _ = state.suggest()
        if nxt is None:
            break
        runs.append(run_once(truth, nxt["condition"], nxt["dose"], q, seed, len(runs) + 1))
        state.observe(runs[-1])
        assert all(state.alive(c) == allowed(runs, c) for c in CONDITIONS), "kernel state != admitted evidence"
    final = {c: sorted(state.alive(c)) for c in CONDITIONS}
    return {"arm": "forest", "seed": seed, "truth": truth, "runs": len(runs), "final": final,
            "solved": all(final[c] == [truth[c]] for c in CONDITIONS), "steps": []}


# ------------------------------------------------------------ model agents
SYSTEM = """You run experiments to identify a hidden rule.
There are three conditions: c0, c1, c2. In each condition, a run at integer dose x (0..5) reads "high" if x >= t and "low" otherwise, for an unknown integer threshold t in 0..6. Conditions may share one threshold or have different thresholds; do not assume either.
Some runs fail. A failed run reads "low" whatever the dose. The operations log raises "ops_alarm": true on most failed runs, but misses some and occasionally alarms on a good run.
You have at most 18 runs. Each turn, return only JSON:
{"possible": {"c0": [...], "c1": [...], "c2": [...]}, "action": {"condition": "c1", "dose": 3}}
"possible" lists every threshold you still consider possible in each condition. Use "action": "stop" when you are done or no runs are left.
You succeed if, when you stop, each condition's list contains exactly its true threshold."""
ASSIST = """
You also receive "forest": the state kept by a hypothesis forest. "surviving_thresholds" are the thresholds consistent with every run without an ops alarm in that condition; "recorded_but_not_refuting" lists alarmed runs, which were kept but not used to eliminate; "suggested_next" is the run that best separates the surviving thresholds (null when nothing separates them). You may use or ignore it."""


def parse(out):
    possible, action = out.get("possible"), out.get("action")
    if not isinstance(possible, dict) or set(possible) != set(CONDITIONS):
        return None
    if any(not isinstance(v, list) or any(not isinstance(t, int) or t not in THRESHOLDS for t in v)
           for v in possible.values()):
        return None
    if action != "stop" and not (isinstance(action, dict) and action.get("condition") in CONDITIONS
                                 and isinstance(action.get("dose"), int) and action["dose"] in DOSES):
        return None
    return {c: sorted(set(possible[c])) for c in CONDITIONS}, action


def score(runs, possible, action):
    """Which of the three errors this decision makes, judged against the admitted evidence."""
    ok = {c: allowed(runs, c) for c in CONDITIONS}
    raw = {c: allowed(runs, c, with_alarmed=True) for c in CONDITIONS}
    dropped = {c: ok[c] - set(possible[c]) for c in CONDITIONS}
    best = max(disagreement([x >= t for t in ok[c]]) for c in CONDITIONS for x in DOSES)
    return {
        "non_discriminating": isinstance(action, dict) and best > 0
        and disagreement([action["dose"] >= t for t in ok[action["condition"]]]) == 0,
        "failure_as_refutation": any(t not in raw[c] for c in CONDITIONS for t in dropped[c]),
        "overgeneralization": any(t in raw[c] and any(t not in ok[k] for k in CONDITIONS if k != c)
                                  for c in CONDITIONS for t in dropped[c]),
        "unsupported_exclusion": any(dropped.values()),
        "kept_refuted": any(set(possible[c]) - ok[c] for c in CONDITIONS),
    }


def model_episode(call, seed, q, dep, assisted):
    truth, runs, steps = make_world(seed, dep), [], []
    state = ThresholdForest() if assisted else None
    system = SYSTEM + (ASSIST if assisted else "")
    possible = {c: list(THRESHOLDS) for c in CONDITIONS}
    while True:
        payload = {"runs": runs, "runs_left": BUDGET - len(runs)}
        if assisted:
            payload["forest"] = state.view(runs)
        decision, error = None, None
        for _ in range(2):
            try:
                decision = parse(call(system, payload if error is None else dict(payload, last_error=error)))
            except Exception as exc:  # transport or JSON failure: one retry, then the episode stops
                error = f"{type(exc).__name__}: {exc}"[:200]
                continue
            if decision:
                break
            error = "invalid output; follow the JSON format exactly"
        if decision is None:
            return {"arm": "assisted" if assisted else "flat", "seed": seed, "truth": truth, "runs": len(runs),
                    "final": possible, "solved": False, "invalid": error, "steps": steps}
        possible, action = decision
        steps.append({"runs": len(runs), "possible": possible, "action": action, "errors": score(runs, possible, action)})
        if action == "stop" or len(runs) >= BUDGET:
            break
        runs.append(run_once(truth, action["condition"], action["dose"], q, seed, len(runs) + 1))
        if assisted:
            state.observe(runs[-1])
    return {"arm": "assisted" if assisted else "flat", "seed": seed, "truth": truth, "runs": len(runs),
            "final": possible, "solved": all(possible[c] == [truth[c]] for c in CONDITIONS), "steps": steps}


def openai_call(system, payload):
    """Minimal OpenAI-compatible client: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL."""
    body = {"model": os.environ["OPENAI_MODEL"], "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload)}]}
    req = urllib.request.Request(os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(json.loads(resp.read())["choices"][0]["message"]["content"])


# ----------------------------------------------------------------- runner
def summarize(episodes):
    rows = {}
    for e in episodes:
        rows.setdefault((e["cell"], e["arm"]), []).append(e)
    print(f"{'cell':10} {'arm':9} {'n':>3} {'solved':>7} {'runs':>5} {'nondisc':>8} {'fail=ref':>8} "
          f"{'overgen':>8} {'unsupp':>7} {'invalid':>7}")
    for (cell, arm), es in sorted(rows.items()):
        steps = [s for e in es for s in e["steps"]]
        acts = [s for s in steps if isinstance(s["action"], dict)]

        def rate(key, pool):
            return f"{sum(s['errors'][key] for s in pool) / len(pool):.0%}" if pool else "-"
        print(f"{cell:10} {arm:9} {len(es):>3} {sum(e['solved'] for e in es) / len(es):>7.0%} "
              f"{sum(e['runs'] for e in es) / len(es):>5.1f} {rate('non_discriminating', acts):>8} "
              f"{rate('failure_as_refutation', steps):>8} {rate('overgeneralization', steps):>8} "
              f"{rate('unsupported_exclusion', steps):>7} {sum(bool(e.get('invalid')) for e in es):>7}")


def main(call=None, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--arms", default="forest", help="comma list of forest, flat, assisted")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--out", default="", help="episodes JSONL; finished (cell, seed, arm) are skipped on rerun")
    ap.add_argument("--jobs", type=int, default=1, help="model episodes run in parallel")
    args = ap.parse_args(argv)
    episodes, lock = [], threading.Lock()
    if args.out and Path(args.out).exists():
        episodes = [json.loads(line) for line in Path(args.out).read_text().splitlines() if line.strip()]
    done = {(e["cell"], e["seed"], e["arm"]) for e in episodes}
    todo = [(cell, seed, arm) for cell in args.cells.split(",") for seed in range(args.worlds)
            for arm in args.arms.split(",") if (cell, seed, arm) not in done]

    def one(job):
        cell, seed, arm = job
        q, dep = CELLS[cell]
        e = forest_episode(seed, q, dep) if arm == "forest" else model_episode(call, seed, q, dep, arm == "assisted")
        e["cell"] = cell
        with lock:
            episodes.append(e)
            if args.out:
                with open(args.out, "a") as out:
                    out.write(json.dumps(e) + "\n")
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        list(pool.map(one, todo))
    summarize(episodes)
    return episodes


if __name__ == "__main__":
    main(openai_call if os.environ.get("OPENAI_MODEL") else None)
