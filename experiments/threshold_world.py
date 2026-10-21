"""Threshold world: a controlled discovery task with known ground truth.

Three conditions. In condition c a run at dose x in 0..5 reads "high" iff x >= t_c, with
t_c in 0..6 unknown; with probability ``dep`` a condition's threshold is redrawn instead
of sharing a common one. A run fails with probability ``q`` and then reads "low"; the
ops alarm catches failures with recall 0.9 and fires on 5% of good runs.

Agents:
  forest    the kernel alone. The seven thresholds are competing hypotheses over all
            conditions; an admitted run that contradicts one narrows its scope away from
            that condition; alarmed runs are recorded but cannot refute. A "high" reading
            is admitted at once; a "low" one, which a silent failure could also produce, is
            admitted only when the same run is replicated. The next run best separates the
            rivals left by admitted and pending readings; when only one is left, the policy
            replicates the low reading that conclusion rests on. No language model.
  forest_adaptive  the kernel with the failure rate estimated from all runs so far: a low reading
            is admitted once the chance it is a silent failure is at most ADMIT_RISK, and the
            next run carries the most information under that failure model, so a replication
            is chosen only when it is worth more than a new dose.
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
import math
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
RECALL, FALSE_ALARM, BUDGET, ADMIT_RISK = 0.9, 0.05, 18, 0.01
WINDOW, NOTE_CHARS = None, 1200  # model agents see only the last WINDOW runs (None: all) plus their own notes


def configure(conditions=3, per_condition=6, window=None):
    global CONDITIONS, BUDGET, WINDOW
    CONDITIONS, BUDGET, WINDOW = tuple(f"c{i}" for i in range(conditions)), per_condition * conditions, window
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


def likelihood(t, run, q):
    """P(reading, alarm | threshold t) when a run fails with probability q."""
    good = (1 - q) * (FALSE_ALARM if run["ops_alarm"] else 1 - FALSE_ALARM)
    bad = q * (RECALL if run["ops_alarm"] else 1 - RECALL)
    return good * ((run["dose"] >= t) == (run["reading"] == "high")) + bad * (run["reading"] == "low")


def failure_rate(runs):
    """Posterior mean failure rate given every run, flat prior on 0..0.5 and on each threshold."""
    grid, like = [i / 100 for i in range(51)], []
    for q in grid:
        p = 1.0
        for c in {r["condition"] for r in runs}:
            mine = [r for r in runs if r["condition"] == c]
            p *= sum(math.prod(likelihood(t, r, q) for r in mine) for t in THRESHOLDS) / len(THRESHOLDS)
        like.append(p)
    return sum(q * w for q, w in zip(grid, like)) / sum(like)


# ----------------------------------------------------------- kernel agent
class ThresholdForest:
    """The kernel as the agent's state: one hypothesis per threshold, scoped by condition."""

    def __init__(self, policy="replicate"):
        # replicate: a low reading waits for one replication; trust: every unalarmed reading counts at once;
        # adaptive: a low reading is admitted once the chance it is a silent failure is at most ADMIT_RISK,
        # under the failure rate estimated so far, and runs are chosen by information under that model
        self.policy, self.replicate, self.runs, self._q = policy, policy != "trust", [], (-1, 0.0)
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

        self.pending, self.admitted = {}, []  # (condition, dose) -> unreplicated low runs; admitted runs

    def observe(self, run):
        f, c, x, rid = self.forest, run["condition"], run["dose"], f"run{run['run']}"
        f.add_record(Record(rid, json.dumps(run), Scope({"condition": c}), "threshold-world", tier="E3",
                            kind="infrastructure" if run["ops_alarm"] else "scientific"))
        self.runs.append(run)
        if run["ops_alarm"]:
            return
        if self.policy == "adaptive":
            if run["reading"] == "high":
                self._admit(run, "no ops alarm; a failure cannot read high")
                for key in [key for key in self.pending if key[0] == c and key[1] >= x]:
                    self.pending.pop(key)
            else:
                self.pending.setdefault((c, x), []).append(run)
            self._admit_safe_lows()
            return
        if run["reading"] == "high" or not self.replicate:
            self._admit(run, "no ops alarm" if not self.replicate else "no ops alarm; a failure cannot read high")
            for (k, d) in [key for key in self.pending if key[0] == c and key[1] >= x]:
                self.pending.pop((k, d))  # a low at or above an admitted high was a silent failure
        elif (c, x) in self.pending and min(self.alive(c), default=x + 1) <= x:
            for earlier in self.pending.pop((c, x)) + [run]:
                self._admit(earlier, f"low reading replicated at {c}, dose {x}")
        elif min(self.alive(c), default=x + 1) <= x:  # informative: it would rule out thresholds <= x
            self.pending.setdefault((c, x), []).append(run)

    def _admit_safe_lows(self):
        q = self.q()
        silent, good = q * (1 - RECALL), (1 - q) * (1 - FALSE_ALARM)
        changed = True
        while changed:
            changed = False
            for (c, x), lows in list(self.pending.items()):
                alive = self.alive(c)
                if not alive or min(alive) > x:  # it no longer rules anything out
                    self.pending.pop((c, x))
                    continue
                n, p_low = len(lows), sum(t > x for t in alive) / len(alive)
                risk = silent ** n / (silent ** n + good ** n * p_low)
                if risk <= ADMIT_RISK:
                    for run in self.pending.pop((c, x)):
                        self._admit(run, f"{n} low reading(s) at {c}, dose {x}; silent-failure risk {risk:.4f}")
                    changed = True

    def q(self):
        if self._q[0] != len(self.runs):
            self._q = (len(self.runs), failure_rate(self.runs))
        return self._q[1]

    def credence(self, c):
        """Weights of the admitted thresholds in condition c given every recorded run there."""
        q = self.q()
        w = {t: 1.0 for t in self.alive(c)}
        for r in self.runs:
            if r["condition"] == c:
                w = {t: v * likelihood(t, r, q) for t, v in w.items()}
        return w

    def _admit(self, run, reason):
        f, c, rid = self.forest, run["condition"], f"run{run['run']}"
        here = Scope({"condition": c})
        f.add_record(Record(rid + "+", reason, here, "ops-log", tier="E4", admits=rid))
        self.admitted.append(run)
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

    def provisional(self, c):
        lows = [x for (k, x) in self.pending if k == c]
        return {t for t in self.alive(c) if all(x < t for x in lows)}

    def final(self, c):
        if len(self.alive(c)) == 1 or self.policy != "adaptive":
            return self.alive(c) if len(self.alive(c)) == 1 else self.provisional(c)
        w = self.credence(c)
        return {max(w, key=w.get)} if w else set()

    def suggest(self):
        if self.policy == "adaptive":
            q, options = self.q(), {f"{c}:{x}": Scope({"condition": c, "dose": x})
                                                   for c in CONDITIONS for x in DOSES}
            weights = {c: self.credence(c) for c in CONDITIONS}

            def predict(h, e):
                x = int(e.split(":")[1])
                return {(r, a): likelihood(int(h.id[1:]), {"dose": x, "reading": r, "ops_alarm": a}, q)
                        for r in ("high", "low") for a in (False, True)}
            eid, bits = choose_experiment(self.forest, options, predict,
                                          weight=lambda h, e: weights[e.split(":")[0]].get(int(h.id[1:]), 0.0))
            return (None, 0.0) if eid is None else ({"condition": eid.split(":")[0], "dose": int(eid.split(":")[1])}, bits)
        for c in CONDITIONS:
            left = self.provisional(c)
            if len(left) == 1 and len(self.alive(c)) > 1:
                return {"condition": c, "dose": min(left) - 1}, 0.0  # replicate the deciding low reading
        options = {f"{c}:{x}": Scope({"condition": c, "dose": x}) for c in CONDITIONS for x in DOSES}
        eid, bits = choose_experiment(self.forest, options, lambda h, e: int(e.split(":")[1]) >= int(h.id[1:]),
                                      consider=lambda h, e: int(h.id[1:]) in self.provisional(e.split(":")[0]))
        return (None, 0.0) if eid is None else ({"condition": eid.split(":")[0], "dose": int(eid.split(":")[1])}, bits)

    def view(self, runs):
        nxt, bits = self.suggest()
        return {"admitted_thresholds": {c: sorted(self.alive(c)) for c in CONDITIONS},
                "provisional_thresholds": {c: sorted(self.provisional(c)) for c in CONDITIONS},
                "low_readings_awaiting_replication": sorted(r["run"] for runs_ in self.pending.values() for r in runs_),
                "recorded_but_not_refuting": [r["run"] for r in runs if r["ops_alarm"]],
                "suggested_next": nxt and dict(nxt, bits=round(bits, 3))}


KERNEL_ARMS = {"forest": "replicate", "forest_trusting": "trust", "forest_adaptive": "adaptive"}


def forest_episode(seed, q, dep, policy="replicate"):
    truth, state, runs = make_world(seed, dep), ThresholdForest(policy), []
    while len(runs) < BUDGET and not all(len(state.alive(c)) == 1 for c in CONDITIONS):
        nxt, _ = state.suggest()
        if nxt is None:
            break
        runs.append(run_once(truth, nxt["condition"], nxt["dose"], q, seed, len(runs) + 1))
        state.observe(runs[-1])
        assert all(state.alive(c) == allowed(state.admitted, c) for c in CONDITIONS), "kernel != admitted evidence"
    final = {c: sorted(state.final(c)) for c in CONDITIONS}
    return {"arm": {v: k for k, v in KERNEL_ARMS.items()}[policy], "seed": seed, "truth": truth, "runs": len(runs), "final": final,
            "solved": all(final[c] == [truth[c]] for c in CONDITIONS), "steps": []}


# ------------------------------------------------------------ model agents
SYSTEM = """You run experiments to identify a hidden rule.
There are {k} conditions: {names}. In each condition, a run at integer dose x (0..5) reads "high" if x >= t and "low" otherwise, for an unknown integer threshold t in 0..6. Conditions may share one threshold or have different thresholds; do not assume either.
Some runs fail. A failed run reads "low" whatever the dose. The operations log raises "ops_alarm": true on most failed runs, but misses some and occasionally alarms on a good run.
You have at most {budget} runs. Each turn, return only JSON:
{{"possible": {{"c0": [...], ...one list per condition...}}, "action": {{"condition": "c1", "dose": 3}}, "notes": "..."}}
"possible" lists every threshold you still consider possible in each condition. Use "action": "stop" when you are done or no runs are left.
You succeed if, when you stop, each condition's list contains exactly its true threshold.{memory}"""
MEMORY = """
You see only the {window} most recent runs. "notes" is your own memory: whatever you write is returned to you next turn, cut to {chars} characters. Keep there what you will need from earlier runs."""
ASSIST = """
You also receive "forest": the state kept by a hypothesis forest over all runs so far. "admitted_thresholds" are consistent with every admitted run in that condition (an unalarmed high reading is admitted at once; a low reading, which a failure could also produce, only once the same run is replicated). "provisional_thresholds" also apply the low readings awaiting replication, listed in "low_readings_awaiting_replication". "recorded_but_not_refuting" lists alarmed runs, kept but never used to eliminate. "suggested_next" is the run that best separates the provisional thresholds, or the replication a conclusion rests on (null when nothing separates them). You may use or ignore it."""


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
    truth, runs, steps, notes = make_world(seed, dep), [], [], ""
    state = ThresholdForest() if assisted else None
    memory = MEMORY.format(window=WINDOW, chars=NOTE_CHARS) if WINDOW else ""
    system = SYSTEM.format(k=len(CONDITIONS), names=", ".join(CONDITIONS), budget=BUDGET, memory=memory)
    system += ASSIST if assisted else ""
    possible = {c: list(THRESHOLDS) for c in CONDITIONS}
    while True:
        payload = {"runs": runs, "runs_left": BUDGET - len(runs)}
        if WINDOW:
            payload = {"recent_runs": runs[-WINDOW:], "earlier_runs_not_shown": max(0, len(runs) - WINDOW),
                       "notes": notes, "runs_left": BUDGET - len(runs)}
        if assisted:
            payload["forest"] = state.view(runs)
        decision, error = None, None
        for _ in range(2):
            try:
                out = call(system, payload if error is None else dict(payload, last_error=error))
                decision = parse(out)
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
        notes = str(out.get("notes") or "")[:NOTE_CHARS]
        steps.append({"runs": len(runs), "possible": possible, "action": action, "notes_chars": len(notes),
                      "errors": score(runs, possible, action)})
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
    print(f"{'cell':10} {'arm':15} {'n':>3} {'solved':>7} {'runs':>5} {'nondisc':>8} {'fail=ref':>8} "
          f"{'overgen':>8} {'unsupp':>7} {'invalid':>7}")
    for (cell, arm), es in sorted(rows.items()):
        steps = [s for e in es for s in e["steps"]]
        acts = [s for s in steps if isinstance(s["action"], dict)]

        def rate(key, pool):
            return f"{sum(s['errors'][key] for s in pool) / len(pool):.0%}" if pool else "-"
        print(f"{cell:10} {arm:15} {len(es):>3} {sum(e['solved'] for e in es) / len(es):>7.0%} "
              f"{sum(e['runs'] for e in es) / len(es):>5.1f} {rate('non_discriminating', acts):>8} "
              f"{rate('failure_as_refutation', steps):>8} {rate('overgeneralization', steps):>8} "
              f"{rate('unsupported_exclusion', steps):>7} {sum(bool(e.get('invalid')) for e in es):>7}")


def main(call=None, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0, help="first world seed")
    ap.add_argument("--conditions", type=int, default=3)
    ap.add_argument("--per-condition", type=int, default=6, help="run budget per condition")
    ap.add_argument("--window", type=int, default=0, help="model agents see only this many recent runs (0: all)")
    ap.add_argument("--arms", default="forest,forest_trusting", help="comma list of forest, forest_trusting, forest_adaptive, flat, assisted")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--out", default="", help="episodes JSONL; finished (cell, seed, arm) are skipped on rerun")
    ap.add_argument("--jobs", type=int, default=1, help="model episodes run in parallel")
    args = ap.parse_args(argv)
    configure(args.conditions, args.per_condition, args.window or None)
    episodes, lock = [], threading.Lock()
    if args.out and Path(args.out).exists():
        episodes = [json.loads(line) for line in Path(args.out).read_text().splitlines() if line.strip()]
    done = {(e["cell"], e["seed"], e["arm"]) for e in episodes}
    todo = [(cell, seed, arm) for cell in args.cells.split(",") for seed in range(args.seed0, args.seed0 + args.worlds)
            for arm in args.arms.split(",") if (cell, seed, arm) not in done]

    def one(job):
        cell, seed, arm = job
        q, dep = CELLS[cell]
        e = (forest_episode(seed, q, dep, KERNEL_ARMS[arm]) if arm in KERNEL_ARMS
             else model_episode(call, seed, q, dep, arm == "assisted"))
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
