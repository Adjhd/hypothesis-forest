"""Mechanism world: discover a law over condition features and predict conditions you cannot run.

Sixteen conditions are the combinations of four binary features f0..f3; eight can be run,
eight are held out. In condition c a run at dose x in 0..5 reads "high" iff x >= t_c, and the
thresholds follow one law: t = base + slope * f_j, or two such rules split by a feature. Runs
fail as in the threshold world (a failed run reads "low"; the ops alarm catches most failures).
Success is predicting every held-out threshold, which needs the law, not one condition at a time.

World size, budget and memory are set by configure(); with a window the model sees only the most recent
runs plus its own notes, while the forest keeps every record.

Agents (all language-model agents see the same prompt; "assisted" and "delegated" also see the forest):
  flat      a language model that proposes candidate laws, picks runs and predicts thresholds.
  assisted  the same model; the laws it proposes are registered in a hypothesis forest as rivals.
            A low reading is admitted once its silent-failure risk is small; a law contradicted by
            an admitted run is narrowed away from that condition, so it no longer covers the domain.
            The forest shows each law's state and credence and suggests the run that carries the
            most information about which consistent law is right.
  delegated the same, except that the forest runs its own suggestion every turn: the model proposes
            laws and answers, the forest chooses the experiments.

  python experiments/mechanism_world.py --worlds 50   # scripted stand-in for the model, no model calls
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hforest import Forest, Record, Scope, choose_experiment  # noqa: E402
from threshold_world import (ADMIT_RISK, DOSES, FALSE_ALARM, RECALL, THRESHOLDS, allowed,  # noqa: E402
                             failure_rate, likelihood, openai_call)

OPEN, OPEN_PRIOR = "L0", 0.3
SPLIT_PRIOR = 0.01  # a split law states two rules and a feature: an Occam penalty on its prior weight
CELLS = {"clean": 0.0, "failures": 0.2}


def configure(features=4, testable=8, budget=16, window=None):
    global NF, FEATURES, CONDITIONS, TESTABLE, BUDGET, WINDOW
    NF, TESTABLE, BUDGET, WINDOW = features, testable, budget, window
    FEATURES = {f"c{i:02d}": bits for i, bits in enumerate(itertools.product((0, 1), repeat=features))}
    CONDITIONS = tuple(FEATURES)


configure()


# ------------------------------------------------------------------ laws
def rule_value(rule, c):
    j = rule.get("feature")
    return rule["base"] + (rule.get("slope", 0) * FEATURES[c][int(j[1:])] if j else 0)


def law_thresholds(law):
    """Thresholds a law gives every condition, or None if it is malformed or leaves 0..6."""
    try:
        if "split_on" in law:
            s = int(law["split_on"][1:])
            t = {c: rule_value(law["if_1"] if FEATURES[c][s] else law["if_0"], c) for c in CONDITIONS}
        else:
            t = {c: rule_value(law, c) for c in CONDITIONS}
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return t if all(isinstance(v, int) and v in THRESHOLDS for v in t.values()) else None


def simple_rules():
    out = [{"base": a, "slope": 0, "feature": None} for a in THRESHOLDS]
    for j, b, a in itertools.product(range(NF), (-3, -2, -1, 1, 2, 3), THRESHOLDS):
        if 0 <= a + b <= 6:
            out.append({"base": a, "slope": b, "feature": f"f{j}"})
    return out


def make_world(seed):
    rng, rules = random.Random(f"mech-{seed}"), simple_rules()
    if rng.random() < 0.5:
        law = rng.choice(rules)
    else:
        r0, r1 = rng.sample(rules, 2)
        law = {"split_on": f"f{rng.randrange(NF)}", "if_0": r0, "if_1": r1}
    testable = sorted(rng.sample(CONDITIONS, TESTABLE))
    return law, law_thresholds(law), testable


def run_once(truth, c, x, q, seed, k):
    rng = random.Random(f"mrun-{seed}-{k}")
    failed = rng.random() < q
    alarm = rng.random() < (RECALL if failed else FALSE_ALARM)
    return {"run": k, "condition": c, "dose": x, "reading": "high" if x >= truth[c] and not failed else "low",
            "ops_alarm": alarm}


# ----------------------------------------------------------- the forest
class LawForest:
    """Candidate laws as rival hypotheses over all sixteen conditions; runs as records."""

    def __init__(self, testable, recombine=True):
        self.testable, self.runs, self.admitted, self.pending, self.laws = testable, [], [], {}, {}
        self.recombine = recombine
        self.everywhere, self._q = Scope({"condition": list(CONDITIONS)}), (-1, 0.0)
        f = self.forest = Forest()
        f.add_plan({"id": "P-dose", "objective": "find the law", "protocol": "one run at a condition and dose",
                    "intervention": "dose 0..5", "controls": ["dose 0", "dose 5"], "observables": ["reading"],
                    "falsifiers": ["reading opposite to the prediction"]})
        # the open alternative: no proposed law is right, each condition has its own unknown threshold
        f.edit("unexplained_observation", "propose", [OPEN], "the proposed laws may all be wrong", hypothesis={
            "id": OPEN, "statement": "no common law; thresholds vary by condition", "scope": self.everywhere,
            "predictions": ["each condition's threshold anywhere in 0..6 allowed by its own runs"],
            "falsifiers": ["none from readings alone"], "variables": ["dose"], "endpoints": ["reading"],
            "plans": ["P-dose"]})

    def propose(self, laws, author, parent=None):
        """Register new laws (one per distinct set of thresholds); returns how many were new."""
        new = 0
        for law in laws or []:
            t = law_thresholds(law) if isinstance(law, dict) else None
            key = t and tuple(t[c] for c in CONDITIONS)
            if not t or key in {tuple(v[1][c] for c in CONDITIONS) for v in self.laws.values()}:
                continue
            hid = f"L{len(self.laws) + 1}"
            reason = f"{law['split_on']} splits {parent}" if parent else "candidate law"
            self.forest.edit("unexplained_observation", "propose", [hid], reason, author=author, hypothesis={
                "id": hid, "statement": json.dumps(law, sort_keys=True), "scope": self.everywhere,
                "predictions": [json.dumps(t, sort_keys=True)], "falsifiers": ["an admitted reading opposite to t_c"],
                "mechanism": json.dumps(law, sort_keys=True), "variables": [f"f{j}" for j in range(NF)] + ["dose"],
                "endpoints": ["reading"], "plans": ["P-dose"]}, link=parent and {"kind": "derived_from", "to": parent})
            self.forest.edit("unresolved_competition", "link_competition", [hid, OPEN], "rival laws",
                             observable="reading")
            self.laws[hid] = (law, t)
            for run in self.admitted:
                self._test(hid, run)
            new += 1
        if author != "forest" and self.recombine:
            new += self.split_failures()
        return new

    def failed_in(self, h):
        return set(CONDITIONS) - set(self.forest.hypotheses[h].scope.dims["condition"])

    def explained(self, h, region):
        """Conditions in region where h agrees with at least one admitted run."""
        return {r["condition"] for r in self.admitted if r["condition"] in region and r["condition"] not in self.failed_in(h)}

    def split_hints(self):
        """(law, feature, value): a simple law that fails in at most 3 conditions, all sharing one feature
        value, and agrees with admitted runs in at least 2 other conditions."""
        hints = []
        for h, (law, _) in self.laws.items():
            failed = self.failed_in(h)
            if "split_on" in law or not 0 < len(failed) <= 3 or len(self.explained(h, set(CONDITIONS))) < 2:
                continue
            hints += [(h, s, FEATURES[min(failed)][s]) for s in range(NF) if len({FEATURES[c][s] for c in failed}) == 1]
        return hints

    def split_failures(self):
        """Split a law where it fails, with the (at most 3) proposed laws that explain most runs there."""
        new = 0
        for h, s, v in self.split_hints():
            region = {c for c in CONDITIONS if FEATURES[c][s] == v}
            partners = [m for m, (other, _) in self.laws.items()
                        if m != h and "split_on" not in other and not self.failed_in(m) & region]
            partners.sort(key=lambda m: -len(self.explained(m, region)))
            for m in partners[:3]:
                law = {"split_on": f"f{s}", f"if_{1 - v}": self.laws[h][0], f"if_{v}": self.laws[m][0]}
                new += self.propose([law], "forest", parent=h)
        return new

    def consistent(self):
        return [h for h in self.laws if self.forest.hypotheses[h].scope == self.everywhere]

    def q(self):
        if self._q[0] != len(self.runs):
            self._q = (len(self.runs), failure_rate(self.runs))
        return self._q[1]

    def local(self, c):
        """Threshold weights in condition c from its own runs alone (the open alternative's view)."""
        q, w = self.q(), {t: 1.0 for t in THRESHOLDS}
        for r in self.runs:
            if r["condition"] == c:
                w = {t: v * likelihood(t, r, q) for t, v in w.items()}
        return w

    def credence(self):
        """Weights of the consistent laws and the open alternative, which holds OPEN_PRIOR of the prior."""
        q, laws, w = self.q(), self.consistent(), {}
        prior = {h: SPLIT_PRIOR if "split_on" in self.laws[h][0] else 1.0 for h in laws}
        for h in laws:
            t, p = self.laws[h][1], (1 - OPEN_PRIOR) * prior[h] / sum(prior.values())
            for r in self.runs:
                p *= likelihood(t[r["condition"]], r, q)
            w[h] = p
        w[OPEN] = OPEN_PRIOR if laws else 1.0
        for c in {r["condition"] for r in self.runs}:
            w[OPEN] *= sum(self.local(c).values()) / len(THRESHOLDS)
        z = sum(w.values())
        return {h: v / z for h, v in w.items()}

    def observe(self, run):
        c, x, rid = run["condition"], run["dose"], f"run{run['run']}"
        self.forest.add_record(Record(rid, json.dumps(run), Scope({"condition": c}), "mechanism-world", tier="E3",
                                      kind="infrastructure" if run["ops_alarm"] else "scientific"))
        self.runs.append(run)
        if run["ops_alarm"]:
            return
        if run["reading"] == "high":
            self._admit(run, "no ops alarm; a failure cannot read high")
            for key in [k for k in self.pending if k[0] == c and k[1] >= x]:
                self.pending.pop(key)
        else:
            self.pending.setdefault((c, x), []).append(run)
        silent, good = self.q() * (1 - RECALL), (1 - self.q()) * (1 - FALSE_ALARM)
        changed = True
        while changed:
            changed = False
            for (c, x), lows in list(self.pending.items()):
                left = allowed(self.admitted, c)
                if not left or min(left) > x:
                    self.pending.pop((c, x))
                    continue
                n, p_low = len(lows), sum(t > x for t in left) / len(left)
                risk = silent ** n / (silent ** n + good ** n * p_low)
                if risk <= ADMIT_RISK:
                    for r in self.pending.pop((c, x)):
                        self._admit(r, f"{n} low reading(s) at {c}, dose {x}; silent-failure risk {risk:.4f}")
                    changed = True
        if self.recombine:
            self.split_failures()

    def _admit(self, run, reason):
        rid = f"run{run['run']}"
        self.forest.add_record(Record(rid + "+", reason, Scope({"condition": run["condition"]}), "ops-log",
                                      tier="E4", admits=rid))
        self.admitted.append(run)
        for h in self.laws:
            self._test(h, run)

    def _test(self, h, run):
        c, rid, hyp = run["condition"], f"run{run['run']}+", self.forest.hypotheses[h]
        if c not in hyp.scope.dims["condition"] or (run["dose"] >= self.laws[h][1][c]) == (run["reading"] == "high"):
            return
        here = Scope({"condition": c})
        self.forest.edit("admitted_contradiction_in_scope", "attach_evidence", [h], f"contradicted in {c}", [rid],
                         verdict="contradicts", target_scope=here, status="tested")
        self.forest.edit("admitted_contradiction_in_scope", "narrow_scope", [h], f"fails in {c}", [rid],
                         new_scope=Scope({"condition": [k for k in hyp.scope.dims["condition"] if k != c]}))

    def suggest(self):
        q, w = self.q(), self.credence()
        options = {f"{c}:{x}": Scope({"condition": c, "dose": x}) for c in self.testable for x in DOSES}

        local = {c: self.local(c) for c in self.testable}

        def predict(h, e):
            c, x = e.split(":")[0], int(e.split(":")[1])
            ts = {self.laws[h.id][1][c]: 1.0} if h.id != OPEN else local[c]
            z = sum(ts.values())
            return {(r, a): sum(v / z * likelihood(t, {"dose": x, "reading": r, "ops_alarm": a}, q) for t, v in ts.items())
                    for r in ("high", "low") for a in (False, True)}
        eid, bits = choose_experiment(self.forest, options, predict, consider=lambda h, e: h.id in w,
                                      weight=lambda h, e: w.get(h.id, 0.0))
        return (None, 0.0) if eid is None else ({"condition": eid.split(":")[0], "dose": int(eid.split(":")[1])}, bits)

    def view(self):
        w, nxt = self.credence(), self.suggest()
        laws = []
        for h, (law, _) in self.laws.items():
            hyp = self.forest.hypotheses[h]
            failed = sorted(set(CONDITIONS) - set(hyp.scope.dims["condition"]))
            laws.append({"id": h, "law": law, "consistent": not failed, "contradicted_in": failed,
                         "credence": round(w.get(h, 0.0), 3)})
        best = max((h for h in w if h != OPEN), key=w.get, default=None)
        hints = [f"{h} fails only where f{s} = {v}: a law for that region may complete it"
                 for h, s, v in self.split_hints()][:8]
        return {"laws": laws, "credence_no_proposed_law_is_right": round(w[OPEN], 3), "split_hints": hints,
                "estimated_failure_rate": round(self.q(), 3),
                "low_readings_awaiting_admission": sorted(r["run"] for rs in self.pending.values() for r in rs),
                "alarmed_runs_not_used": [r["run"] for r in self.runs if r["ops_alarm"]],
                "thresholds_under_most_credible_law": best and self.laws[best][1],
                "suggested_next": nxt[0] and dict(nxt[0], bits=round(nxt[1], 3))}


# ------------------------------------------------------------ model agents
SYSTEM = """You are discovering a hidden law from experiments.
There are {n} conditions c00..{last}. Each is defined by {nf} binary features {names} (table in "conditions"). Only the {k} conditions listed in "testable" can be run.
A run at integer dose x (0..5) in condition c reads "high" if x >= t_c and "low" otherwise, where t_c is an integer threshold in 0..6.
The thresholds follow one law over the features, of one of two forms:
  a rule  {{"base": b, "slope": s, "feature": "f1"}}  meaning t = b + s * f1 (slope may be negative; use "feature": null for a constant), or
  a split {{"split_on": "f2", "if_0": rule, "if_1": rule}}  meaning one rule where f2 = 0 and another where f2 = 1.
Some runs fail. A failed run reads "low" whatever the dose. The operations log raises "ops_alarm": true on most failed runs, but misses some and occasionally alarms on a good run.
You have at most {budget} runs. You are scored on the thresholds of the {held} conditions you cannot run, so you must find the law.
Each turn return only JSON:
{{"laws": [up to 10 candidate laws you still consider, most likely first], "thresholds": {{"c00": t, ... one integer for each of the {n} conditions}}, "action": {{"condition": "c03", "dose": 2}}, "notes": "..."}}
Use "action": "stop" when you are done. Your last "thresholds" are your answer.{memory}"""
MEMORY = """
You see only the {window} most recent runs ("recent_runs"). "notes" is your own memory: whatever you write is returned to you next turn, cut to 1200 characters. Keep there what you will need from earlier runs."""
ASSIST = """
You also receive "forest": the state kept by a hypothesis forest over every law you have proposed, plus split laws it assembles from them. A high reading without alarm is admitted at once; a low one only once the chance it is a silent failure is small (shown in "low_readings_awaiting_admission"). A law contradicted by an admitted reading is marked with the conditions where it failed; "split_hints" name laws whose failures all share one feature value, where a law for that region may complete them. "credence" weighs the consistent laws by all runs under the estimated failure rate, a split law starting lower than a single rule; "credence_no_proposed_law_is_right" is the weight left for laws nobody has proposed yet. "suggested_next" is the run that carries the most information about which of these is right. You may use or ignore it."""
DELEGATE = """
In this setting the forest runs its own "suggested_next" every turn; your "action" is ignored and may be omitted. Spend your effort on proposing laws and on your thresholds."""


def parse(out, testable, delegated=False):
    th, action = out.get("thresholds"), out.get("action")
    if not isinstance(th, dict) or set(th) != set(CONDITIONS) or any(t not in THRESHOLDS or not isinstance(t, int)
                                                                       for t in th.values()):
        return None
    if not delegated and action != "stop" and not (isinstance(action, dict) and action.get("condition") in testable
                                 and isinstance(action.get("dose"), int) and action["dose"] in DOSES):
        return None
    laws = out.get("laws") if isinstance(out.get("laws"), list) else []
    laws = [x["law"] if isinstance(x, dict) and isinstance(x.get("law"), dict) else x for x in laws]
    return th, action, laws[:10]


def score(truth, th, testable):
    held = [c for c in CONDITIONS if c not in testable]
    return {"heldout_exact": sum(th[c] == truth[c] for c in held) / len(held),
            "law_exact": all(th[c] == truth[c] for c in CONDITIONS)}


def model_episode(call, seed, q, mode, recombine=True):
    """mode: flat, assisted or delegated."""
    law, truth, testable = make_world(seed)
    assisted, delegated = mode != "flat", mode == "delegated"
    state = LawForest(testable, recombine) if assisted else None
    memory = MEMORY.format(window=WINDOW) if WINDOW else ""
    system = SYSTEM.format(budget=BUDGET, n=len(CONDITIONS), last=CONDITIONS[-1], nf=NF, k=len(testable),
                           held=len(CONDITIONS) - len(testable), names=", ".join(f"f{j}" for j in range(NF)),
                           memory=memory) + (ASSIST if assisted else "") + (DELEGATE if delegated else "")
    runs, steps, notes, th = [], [], "", {c: 0 for c in CONDITIONS}
    while True:
        payload = {"conditions": {c: {f"f{j}": v for j, v in enumerate(FEATURES[c])} for c in CONDITIONS},
                   "testable": testable, "runs": runs, "notes": notes, "runs_left": BUDGET - len(runs)}
        if WINDOW:
            payload.pop("runs")
            payload.update(recent_runs=runs[-WINDOW:], earlier_runs_not_shown=max(0, len(runs) - WINDOW))
        if assisted:
            payload["forest"] = state.view()
        decision, error = None, None
        for _ in range(2):
            try:
                out = call(system, payload if error is None else dict(payload, last_error=error))
                decision = parse(out, testable, delegated)
            except Exception as exc:  # transport or JSON failure: one retry, then the episode stops
                error = f"{type(exc).__name__}: {exc}"[:200]
                continue
            if decision:
                break
            error = "invalid output; follow the JSON format exactly"
        if decision is None:
            return {"arm": mode, "seed": seed, "law": law, "truth": truth, "runs": len(runs), "final": th, **score(truth, th, testable), "invalid": error, "steps": steps}
        th, action, laws = decision
        notes = str(out.get("notes") or "")[:1200]
        valid = [x for x in laws if isinstance(x, dict) and law_thresholds(x)]
        step = {"runs": len(runs), "action": action, "laws": laws, "valid_laws": len(valid),
                "true_law_listed": any(law_thresholds(x) == truth for x in valid), **score(truth, th, testable)}
        if assisted:
            step["suggested"] = payload["forest"]["suggested_next"]
            step["new_laws"] = state.propose(valid, "model")
        if delegated and len(runs) < BUDGET:
            nxt = state.suggest()[0]
            least = min(testable, key=lambda c: sum(r["condition"] == c for r in runs))
            action = step["action"] = nxt or {"condition": least, "dose": 3}
        steps.append(step)
        if action == "stop" or len(runs) >= BUDGET:
            break
        runs.append(run_once(truth, action["condition"], action["dose"], q, seed, len(runs) + 1))
        if assisted:
            state.observe(runs[-1])
    return {"arm": mode, "seed": seed, "law": law, "truth": truth, "runs": len(runs), "final": th, **score(truth, th, testable), "steps": steps}


def scripted_call(system, payload):
    """Stand-in for a model, for checks without one: proposes up to 10 simple laws (never a split law) that
    fit the unalarmed runs, or only those in the region of a split hint; follows the forest's suggestion and
    answers with its most credible law."""
    forest = payload.get("forest") or {}
    region = set(CONDITIONS)
    for hint in forest.get("split_hints", [])[:1]:
        feature, value = hint.split(" fails only where ")[1].split(":")[0].split(" = ")
        region = {c for c in CONDITIONS if FEATURES[c][int(feature[1:])] == int(value)}
    runs = [r for r in payload.get("runs", payload.get("recent_runs")) if not r["ops_alarm"] and r["condition"] in region]
    fits = [law for law in simple_rules() if all((r["dose"] >= law_thresholds(law)[r["condition"]]) ==
                                                 (r["reading"] == "high") for r in runs)]
    laws = random.Random(json.dumps(runs) + str(payload["runs_left"])).sample(fits, min(10, len(fits)))
    nxt = forest.get("suggested_next")
    action = {"condition": nxt["condition"], "dose": nxt["dose"]} if nxt and payload["runs_left"] else "stop"
    th = forest.get("thresholds_under_most_credible_law") or (law_thresholds(laws[0]) if laws else
                                                               {c: 3 for c in CONDITIONS})
    if payload["runs_left"] == BUDGET and action == "stop":
        action = {"condition": payload["testable"][0], "dose": 3}
    return {"laws": laws, "thresholds": th, "action": action, "notes": ""}


# ----------------------------------------------------------------- runner
def summarize(episodes):
    rows = {}
    for e in episodes:
        rows.setdefault((e["cell"], e["arm"]), []).append(e)
    print(f"{'cell':9} {'arm':16} {'n':>3} {'held-out':>9} {'law':>6} {'runs':>5} {'true law listed':>16} {'invalid':>7}")
    for (cell, arm), es in sorted(rows.items()):
        listed = sum(any(s["true_law_listed"] for s in e["steps"]) for e in es)
        print(f"{cell:9} {arm:16} {len(es):>3} {sum(e['heldout_exact'] for e in es) / len(es):>9.0%} "
              f"{sum(e['law_exact'] for e in es) / len(es):>6.0%} {sum(e['runs'] for e in es) / len(es):>5.1f} "
              f"{listed / len(es):>16.0%} {sum(bool(e.get('invalid')) for e in es):>7}")


def main(call=None, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--arms", default="scripted_forest,scripted_plain",
                    help="comma list of flat, assisted, delegated, scripted_forest, scripted_plain (no splitting)")
    ap.add_argument("--features", type=int, default=4)
    ap.add_argument("--testable", type=int, default=8)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--window", type=int, default=0, help="model agents see only this many recent runs (0: all)")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    configure(args.features, args.testable, args.budget, args.window or None)
    episodes, lock = [], threading.Lock()
    if args.out and Path(args.out).exists():
        episodes = [json.loads(line) for line in Path(args.out).read_text().splitlines() if line.strip()]
    done = {(e["cell"], e["seed"], e["arm"]) for e in episodes}
    todo = [(cell, seed, arm) for cell in args.cells.split(",") for seed in range(args.seed0, args.seed0 + args.worlds)
            for arm in args.arms.split(",") if (cell, seed, arm) not in done]

    def one(job):
        cell, seed, arm = job
        if arm.startswith("scripted"):
            e = model_episode(scripted_call, seed, CELLS[cell], "assisted", recombine=arm == "scripted_forest")
            e["arm"] = arm
        else:
            e = model_episode(call, seed, CELLS[cell], arm)
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
