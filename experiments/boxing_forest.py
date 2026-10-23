"""BoxingGym with a hypothesis forest: the forest chooses every measurement, the model proposes laws and answers.

BoxingGym (Gandhi et al., arXiv 2501.01540, MIT) gives scientific environments as generative models: an agent
measures the response at inputs of its choice, then predicts the response at held-out inputs. Its standard
protocol is ten measurements with the whole history in the conversation.

Arms (the same model and the same environment prompt):
  flat       BoxingGym's own agent loop (LMExperimenter + iterative_experiment); the model picks each input.
  delegated  the model proposes candidate laws: numpy expressions in the inputs with free parameters. Each law
             is a rival hypothesis in a forest, competing with an open alternative ("no proposed law is right").
             The forest fits every law to the admitted measurements and weighs it by its prequential score:
             how well it predicted each measurement before seeing it, so extra parameters pay for themselves.
             A law more than a factor 1000 behind the best rival is marked contradicted. The forest picks the
             input whose outcome carries the most information about which rival is right (choose_experiment);
             the model sees the fitted laws, their credence and their predictions when it answers.

Needs the boxing-gym package (pip install -e boxing-gym) and its dependencies (pymc, scipy).
  python experiments/boxing_forest.py --envs dugongs --seeds 2      # scripted stand-in for the model
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import xlogy
from scipy.stats import norm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hforest import Forest, Record, Scope, choose_experiment  # noqa: E402

OPEN, OPEN_PRIOR, CONTRADICT_FACTOR = "L0", 0.3, 1000.0
CANDIDATES, CONSIDERED, BINS, NEW_LAWS, FLOOR = 64, 8, 40, 4, -50.0
SCHEMA = {"type": "object"}

# env name -> (module, env class, goal class, env params, input names, input box, outcome kind)
ENVS = {
    "dugongs": ("dugongs", "Dugongs", "DirectGoal",
                {"lower_limit": 0, "upper_limit": 5, "alpha": 2, "beta": 1.5, "lambda_": 0.4},
                ("x",), ((0.0, 5.0),), "real"),
    "peregrines": ("peregrines", "Peregrines", "DirectGoal",
                   {"lower_limit": 0, "upper_limit": 5, "alpha": 4.5, "beta1": 1.2, "beta2": 0.07, "beta3": -0.24},
                   ("x",), ((0.0, 5.0),), "count"),
    "location_finding": ("location_finding", "Signal", "DirectGoal", {"num_sources": 3, "dim": 2},
                         ("x1", "x2"), ((-2.0, 2.0), (-2.0, 2.0)), "real"),
}


# ------------------------------------------------------------------ laws
FUNCS = {name: getattr(np, name) for name in ("exp", "log", "sqrt", "abs", "sin", "cos", "tanh", "minimum", "maximum")}
NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub,
         ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


class Law:
    """A numpy expression in the inputs; every other name is a free parameter."""

    def __init__(self, expr: str, inputs, init=None):
        self.expr = re.sub(r"\b(np|numpy|math)\.", "", str(expr)).replace("^", "**").strip()
        tree = ast.parse(self.expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, NODES):
                raise ValueError(f"not allowed in a law: {type(node).__name__}")
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCS
                                               or node.keywords):
                raise ValueError("only exp, log, sqrt, abs, sin, cos, tanh, minimum, maximum may be called")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError("constants must be numbers")
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.inputs, self.params = tuple(inputs), sorted(names - set(inputs) - set(FUNCS))
        if len(self.params) > 10:
            raise ValueError("at most 10 parameters")
        self.code = compile(tree, "<law>", "eval")
        init = init if isinstance(init, dict) else {}
        self.init = np.array([float(init.get(p, 1.0)) if _number(init.get(p, 1.0)) else 1.0 for p in self.params])

    def __call__(self, X, theta):
        scope = {"__builtins__": {}, **FUNCS, **dict(zip(self.inputs, X.T)), **dict(zip(self.params, theta))}
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(eval(self.code, scope), dtype=float), (len(X),)).copy()


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def residuals(law, X, y, theta, kind):
    mu = law(X, theta)
    if kind == "count":
        mu = np.maximum(mu, 1e-9)
        r = np.sign(y - mu) * np.sqrt(np.maximum(2 * (xlogy(y, y / mu) - (y - mu)), 0))
    else:
        r = mu - y
    return np.where(np.isfinite(r), r, 1e6)


def fit(law, X, y, kind, rng):
    """Least squares (deviance residuals for counts) from the proposed values and three jittered restarts."""
    if not law.params or not len(y):
        return law.init
    best, starts = None, [law.init] + [law.init * np.exp(rng.normal(0, 0.7, len(law.init))) + rng.normal(0, 0.1, len(law.init))
                                       for _ in range(3)]
    for s in starts:
        try:
            r = least_squares(lambda th: residuals(law, X, y, th, kind), s, method="trf", max_nfev=300)
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    return law.init if best is None else best.x


def open_scale(y):
    return float(np.std(y) * 1.5) if len(y) >= 2 and np.std(y) > 0 else float(abs(y[0]) + 1.0) if len(y) else 1.0


def jacobian(law, X, theta):
    base, cols = law(X, theta), []
    for i in range(len(theta)):
        step = np.zeros_like(theta)
        step[i] = 1e-6 * (abs(theta[i]) + 1)
        cols.append((law(X, theta + step) - base) / step[i])
    return np.column_stack(cols) if cols else np.zeros((len(X), 0))


def law_predictive(law, X, y, theta, kind, Xq):
    """(mean, scale, degrees of freedom) of a new measurement at Xq: the noise plus the fitted parameters'
    uncertainty, linearized. Real responses get a Student t with m - k degrees of freedom; counts a Poisson
    variance plus parameter uncertainty."""
    m, k = len(y), len(law.params)
    mu, G, Gq = law(Xq, theta), jacobian(law, X, theta), jacobian(law, Xq, theta)
    with np.errstate(all="ignore"):
        if kind == "count":
            info = (G / np.maximum(law(X, theta), 1e-9)[:, None]).T @ G
            noise, cov, dof = np.maximum(mu, 0.25), np.linalg.pinv(info), math.inf
        else:
            s2 = max(float(np.sum((law(X, theta) - y) ** 2)) / (m - k), (1e-3 * open_scale(y)) ** 2)
            noise, cov, dof = s2, s2 * np.linalg.pinv(G.T @ G), m - k
        var = noise + np.einsum("ij,jk,ik->i", Gq, cov, Gq)
    return mu, np.sqrt(np.maximum(var, 1e-12)), dof


def spread(scale, dof):
    """Standard deviation of a t (or normal) predictive, capped at three scales for very few dof."""
    return scale * (math.sqrt(dof / (dof - 2)) if 2 < dof < math.inf else 3.0 if dof <= 2 else 1.0)


def log_density(v, mu, scale, dof=math.inf):
    """Log predictive density, floored at FLOOR so that one wild prediction cannot become -inf or nan."""
    if not (math.isfinite(mu) and math.isfinite(scale)) or scale <= 0:
        return FLOOR
    lp = norm.logpdf(v, mu, scale) if dof == math.inf else student_t.logpdf(v, dof, mu, scale)
    return max(float(lp), FLOOR) if math.isfinite(lp) else FLOOR


# ----------------------------------------------------------- the forest
class BoxForest:
    """Candidate laws as rival hypotheses over the environment's input space; measurements as records."""

    def __init__(self, env_name, seed):
        _, _, _, _, self.inputs, self.box, self.kind = ENVS[env_name]
        self.rng, self.laws, self.X, self.y, self.score = np.random.default_rng(seed), {}, [], [], {}
        self.domain = Scope({"environment": env_name})
        f = self.forest = Forest()
        f.add_plan({"id": "P-measure", "objective": "find the response law", "protocol": "one measurement at an input",
                    "intervention": "input in the declared range", "controls": ["repeat an input"],
                    "observables": ["response"], "falsifiers": ["response far outside the law's prediction"]})
        f.edit("unexplained_observation", "propose", [OPEN], "the proposed laws may all be wrong", hypothesis={
            "id": OPEN, "statement": "no proposed law describes the response", "scope": self.domain,
            "predictions": ["the response varies around its running mean"], "falsifiers": ["none from responses alone"],
            "variables": list(self.inputs), "endpoints": ["response"], "plans": ["P-measure"]})
        self.score[OPEN] = 0.0

    # --- data
    def arrays(self, n=None):
        n = len(self.y) if n is None else n
        return np.array(self.X[:n], dtype=float).reshape(n, len(self.inputs)), np.array(self.y[:n], dtype=float)

    def observe(self, x, value):
        k = len(self.y)
        rid = f"m{k + 1}"
        self.forest.add_record(Record(rid, json.dumps({"input": list(map(float, x)), "response": float(value)}),
                                      self.domain, "boxing-gym", tier="E3"))
        self.forest.add_record(Record(rid + "+", "the environment reports no execution failures", self.domain,
                                      "boxing-gym", tier="E4", admits=rid))
        X, y = self.arrays(k)
        xi = np.array(x, dtype=float).reshape(1, -1)
        self.score[OPEN] += self._open_term(y, value)
        for h, law in self.laws.items():
            self.score[h] += self._term(law, X, y, xi, value)
        self.X.append(list(map(float, x)))
        self.y.append(float(value))
        self._contradict()

    @staticmethod
    def _open_term(y, value):
        return log_density(value, y.mean() if len(y) else 0.0, open_scale(y))

    def _term(self, law, X, y, xi, value):
        """Log density of the next measurement under the law fitted to the earlier ones. Until the law has more
        measurements than parameters its fit is not determined, and it is scored like the open alternative."""
        if len(y) <= len(law.params):
            return self._open_term(y, value)
        mu, scale, dof = law_predictive(law, X, y, fit(law, X, y, self.kind, self.rng), self.kind, xi)
        return log_density(value, float(mu[0]), float(scale[0]), dof)

    def _contradict(self):
        best = max(self.score[h] for h in self.alive())
        for h in self.alive():
            if (h != OPEN and self.score[h] < best - math.log(CONTRADICT_FACTOR)
                    and len(self.y) >= len(self.laws[h].params) + 3):
                evidence = [f"m{i + 1}+" for i in range(len(self.y))]
                self.forest.edit("admitted_contradiction_in_scope", "attach_evidence", [h],
                                 f"predicted the measurements {CONTRADICT_FACTOR:.0f}x worse than the best rival",
                                 evidence, verdict="contradicts", status="contradicted")

    # --- laws
    def alive(self):
        return [h for h in self.score if self.forest.hypotheses[h].status != "contradicted"]

    def propose(self, items, author="model"):
        """Register laws; returns (new ids, rejection messages). A law's score starts from its prequential
        score on the measurements so far, so a late law is judged on the same footing as an early one."""
        new, rejected = [], []
        for item in (items or [])[:NEW_LAWS]:
            try:
                expr, init = (item.get("expr"), item.get("init")) if isinstance(item, dict) else (item, None)
                law = Law(expr, self.inputs, init)
                law(np.zeros((1, len(self.inputs))), law.init)
            except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
                rejected.append(f"{str(item)[:80]}: {exc}")
                continue
            if law.expr in {v.expr for v in self.laws.values()}:
                continue
            hid = f"L{len(self.laws) + 1}"
            self.forest.edit("unexplained_observation", "propose", [hid], "candidate law", author=author, hypothesis={
                "id": hid, "statement": f"response = {law.expr}", "scope": self.domain,
                "predictions": [f"response = {law.expr} with fitted {', '.join(law.params) or 'no parameters'}"],
                "falsifiers": ["measurements it predicts far worse than a rival"], "mechanism": law.expr,
                "variables": list(self.inputs), "endpoints": ["response"], "plans": ["P-measure"]})
            self.forest.edit("unresolved_competition", "link_competition", [hid, OPEN], "rival laws", observable="response")
            self.laws[hid], self.score[hid] = law, self._prequential(law)
            new.append(hid)
        if new:
            self._contradict()
        return new, rejected

    def _prequential(self, law):
        total = 0.0
        for i in range(len(self.y)):
            X, y = self.arrays(i)
            total += self._term(law, X, y, np.array(self.X[i], dtype=float).reshape(1, -1), self.y[i])
        return total

    def credence(self):
        alive = self.alive()
        laws = [h for h in alive if h != OPEN]
        logw = {h: math.log((1 - OPEN_PRIOR) / len(laws)) + self.score[h] for h in laws}
        logw[OPEN] = math.log(OPEN_PRIOR if laws else 1.0) + self.score[OPEN]
        top = max(logw.values())
        w = {h: math.exp(v - top) for h, v in logw.items()}
        z = sum(w.values())
        return {h: v / z for h, v in w.items()}

    def fitted(self):
        X, y = self.arrays()
        return {h: fit(law, X, y, self.kind, self.rng) for h, law in self.laws.items() if h in self.alive()}

    def predictive(self, Xq, fitted=None):
        """{rival: (mean, sd)} at the query inputs, fitted to every admitted measurement."""
        X, y = self.arrays()
        fitted = self.fitted() if fitted is None else fitted
        out = {OPEN: (np.full(len(Xq), y.mean() if len(y) else 0.0), np.full(len(Xq), open_scale(y)))}
        for h, theta in fitted.items():
            if len(y) <= len(self.laws[h].params):  # not determined yet: predicts like the open alternative
                out[h] = out[OPEN]
                continue
            mu, scale, dof = law_predictive(self.laws[h], X, y, theta, self.kind, Xq)
            out[h] = (mu, spread(scale, dof))
        return out

    def prediction(self, x):
        """Credence-weighted median of the rivals' predicted responses at one input."""
        w, pred = self.credence(), self.predictive(np.array(x, dtype=float).reshape(1, -1))
        pts = sorted((float(pred[h][0][0]), w[h]) for h in w if np.isfinite(pred[h][0][0]))
        half, acc = sum(v for _, v in pts) / 2, 0.0
        for value, v in pts:
            acc += v
            if acc >= half:
                return value
        return 0.0

    def suggest(self):
        """The input among random candidates whose outcome is most informative about which rival is right."""
        w = self.credence()
        keep = sorted(w, key=lambda h: -w[h])[:CONSIDERED + 1]
        cand = np.column_stack([self.rng.uniform(lo, hi, CANDIDATES) for lo, hi in self.box])
        pred = {h: v for h, v in self.predictive(cand).items() if h in keep}
        dists = {}
        for j in range(len(cand)):
            mus = {h: (m[j], s[j]) for h, (m, s) in pred.items() if np.isfinite(m[j]) and np.isfinite(s[j])}
            if not mus:
                continue
            lo = min(m - 4 * s for m, s in mus.values())
            hi = max(m + 4 * s for m, s in mus.values())
            edges = np.linspace(lo, hi, BINS + 1) if hi > lo else np.array([lo - 1, lo + 1])
            dists[j] = {h: {b: float(p) for b, p in enumerate(np.diff(norm.cdf(np.r_[-np.inf, edges[1:-1], np.inf], m, s)))}
                        for h, (m, s) in mus.items()}
        options = {str(j): self.domain for j in dists}
        eid, bits = choose_experiment(self.forest, options, lambda h, e: dists[int(e)].get(h.id, {0: 1.0}),
                                      consider=lambda h, e: h.id in dists[int(e)],
                                      weight=lambda h, e: w.get(h.id, 0.0))
        j = int(eid) if eid is not None else 0
        return [float(v) for v in cand[j]], bits

    def view(self, limit=8):
        w, fitted = self.credence(), self.fitted()
        laws = sorted(fitted, key=lambda h: -w.get(h, 0))[:limit]
        X, y = self.arrays()
        rows = []
        for h in laws:
            law, theta = self.laws[h], fitted[h]
            err = float(np.sqrt(np.mean((law(X, theta) - y) ** 2))) if len(y) else None
            rows.append({"id": h, "law": law.expr, "fitted": {p: round(float(v), 4) for p, v in zip(law.params, theta)},
                         "credence": round(w.get(h, 0), 4), "rms_error": err and round(err, 4)})
        dead = [h for h in self.laws if h not in w]
        return {"laws": rows, "laws_not_shown": max(0, len(fitted) - limit),
                "contradicted": [self.laws[h].expr for h in dead[-5:]], "contradicted_count": len(dead),
                "credence_no_proposed_law_is_right": round(w[OPEN], 4)}


# ------------------------------------------------------------------ prompts
DELEGATE = """
You work with a hypothesis forest. You do not choose the measurements: the forest chooses each one so that its
outcome best tells your candidate laws apart. Your job is to propose candidate laws for the response and, when
asked, to answer prediction questions.
A law is a numpy expression in {inputs} whose other names are free parameters, for example "a - b * c**x".
Allowed functions: exp, log, sqrt, abs, sin, cos, tanh, minimum, maximum. The forest fits the parameters to the
measurements and weighs each law by how well it predicted every measurement before seeing it, so a law with
needless parameters loses to a simpler correct one. Laws that predict much worse than a rival are contradicted.
Reply with one JSON object: {{"laws": [{{"expr": "...", "init": {{"a": 1.0}}}}], "notes": "..."}} with at most {n}
new laws (laws already in the forest are kept; do not repeat them; an empty list is fine)."""

ANSWER = """
You work with a hypothesis forest that fits your candidate laws to every measurement. Answer the prediction
question. The forest's laws, their fitted parameters and credence, and their predictions at the queried input are
given; use them or not. Reply with one JSON object: {"answer": <number>}"""

REPLY = """
The conversation so far is the JSON list in the user message; its last entry is the message you must answer now.
Reply with one JSON object {"reply": "<your message>"}, where reply is exactly the text you would send, including
any <thought>, <observe> or <answer> tags the conversation asks for."""


# ------------------------------------------------------------------ agents
def gateway_experimenter(call):
    """BoxingGym's LMExperimenter with the chat call replaced by a JSON call(system, payload)."""
    from boxing_gym.agents.agent import LMExperimenter

    class GatewayExperimenter(LMExperimenter):
        def __init__(self):
            self.model_name, self.temperature, self.max_tokens = "gateway", 0.0, None
            self.system, self.messages, self.all_messages = None, [], []

        def set_system_message(self, message):
            self.all_messages.append(f"role:system, messaage:{message}")
            self.system = message

        def add_message(self, message, role="user"):
            self.all_messages.append(f"role:{role}, messaage:{message}")
            self.messages.append({"role": role, "content": message})

        def prompt_llm(self, request_prompt):
            self.add_message(request_prompt)
            out = call(self.system + REPLY, {"conversation": self.messages})
            reply = out.get("reply") if isinstance(out, dict) else None
            reply = reply if isinstance(reply, str) else json.dumps(out)
            self.add_message(reply, "assistant")
            return reply

    return GatewayExperimenter()


class ForestScientist:
    """Stands where BoxingGym's scientist stands: the forest picks inputs, the model proposes laws and answers."""

    def __init__(self, call, env_name, goal, seed, window=None):
        self.call, self.goal, self.window = call, goal, window
        self.state, self.inputs, self.seen, self.proposed_at = BoxForest(env_name, seed), ENVS[env_name][4], 0, None
        self.system, self.messages, self.all_messages, self.log = None, [], [], []

    def set_system_message(self, message):
        self.system = message
        self.all_messages.append(f"role:system, messaage:{message}")

    def _data(self):
        env = self.goal.env
        return env.observed_data if hasattr(env, "observed_data") else env.observation_data

    def _sync(self):
        data = self._data()
        for x, value in data[self.seen:]:
            self.state.observe(np.atleast_1d(np.asarray(x, dtype=float)), float(value))
        self.seen = len(data)

    def _records(self):
        rows = [{"input": x if len(x) > 1 else x[0], "response": v} for x, v in zip(self.state.X, self.state.y)]
        if self.window and len(rows) > self.window:
            return {"recent_measurements": rows[-self.window:], "earlier_measurements_not_shown": len(rows) - self.window}
        return {"measurements": rows}

    def _propose(self):
        if self.proposed_at == len(self.state.y):
            return
        system = self.system + DELEGATE.format(inputs=", ".join(self.inputs), n=NEW_LAWS)
        out = self.call(system, {**self._records(), "forest": self.state.view()})
        new, rejected = self.state.propose(out.get("laws") if isinstance(out, dict) else None)
        self.log.append({"step": len(self.state.y), "new_laws": new, "rejected": rejected})
        self.proposed_at = len(self.state.y)

    def generate_actions(self, experiment_results=None):
        self._sync()
        self._propose()
        x, bits = self.state.suggest()
        self.log.append({"step": len(self.state.y), "chosen": x, "bits": round(bits, 4),
                         "credence_open": round(self.state.credence()[OPEN], 4)})
        text = f"{x[0]:.4f}" if len(x) == 1 else "[" + ", ".join(f"{v:.4f}" for v in x) + "]"
        self.all_messages.append(f"role:forest, messaage:<observe>{text}</observe> ({bits:.3f} bits)")
        return text

    def prompt_llm_and_parse(self, request_prompt, is_observation, max_tries=4):
        return self.generate_actions(), 0

    def generate_predictions(self, request_prompt):
        self._sync()
        self._propose()
        x = np.atleast_1d(np.asarray(self.goal.eval_points[self.goal.eval_pointer - 1][0], dtype=float))
        forest = self.state.view()
        if self.state.y:  # without measurements no law is fitted and the forest has no prediction
            w, pred = self.state.credence(), self.state.predictive(x.reshape(1, -1))
            forest["predictions_at_query"] = [{"id": h, "credence": round(w[h], 4), "prediction": round(float(pred[h][0][0]), 4)}
                                              for h in sorted(w, key=lambda h: -w[h])[:CONSIDERED] if np.isfinite(pred[h][0][0])]
            forest["credence_weighted_median_prediction"] = round(self.state.prediction(x), 4)
        out = self.call(self.system + ANSWER, {"question": request_prompt, **self._records(), "forest": forest})
        answer = out.get("answer") if isinstance(out, dict) else None
        fallback = forest.get("credence_weighted_median_prediction", 0.0)
        self.log.append({"eval_input": x.tolist(), "answer": answer, "forest_prediction": forest.get("credence_weighted_median_prediction")})
        return str(float(answer)) if _number(answer) else str(fallback)


# ------------------------------------------------------------------ episodes
def _seed_everything(seed):
    """Same world and evaluation points for every arm: pymc draws take seeds from numpy's global stream."""
    import pymc as pm
    random.seed(seed)
    np.random.seed(seed)
    if not hasattr(pm, "_hf_draw"):
        pm._hf_draw, pm._hf_spp = pm.draw, pm.sample_prior_predictive
        pm.draw = lambda *a, **k: pm._hf_draw(*a, **{"random_seed": int(np.random.randint(2 ** 31)), **k})
        pm.sample_prior_predictive = lambda *a, **k: pm._hf_spp(*a, **{"random_seed": int(np.random.randint(2 ** 31)), **k})


def make(env_name):
    import importlib
    module, env_cls, goal_cls, params, *_ = ENVS[env_name]
    mod = importlib.import_module(f"boxing_gym.envs.{module}")
    env = getattr(mod, env_cls)(**params)
    env.include_prior = True
    return env, getattr(mod, goal_cls)(env)


def episode(call, env_name, seed, arm, checkpoints=(0, 5, 10), num_evals=10, window=None):
    import logging
    logging.getLogger("pymc").setLevel(logging.ERROR)
    from run_experiment import iterative_experiment  # BoxingGym's own loop
    _seed_everything(seed)
    env, goal = make(env_name)
    calls = {"n": 0}

    def counted(system, payload):
        """One retry of the same request after a transport failure; a second failure ends the episode."""
        calls["n"] += 1
        try:
            return call(system, payload)
        except Exception:  # noqa: BLE001 - the gateway's error type is not importable here
            calls["retries"] = calls.get("retries", 0) + 1
            return call(system, payload)

    scientist = gateway_experimenter(counted) if arm == "flat" else ForestScientist(counted, env_name, goal, seed, window)
    scientist.set_system_message(goal.get_system_message(True))
    results, queries, observations, successes, *_ = iterative_experiment(
        goal, scientist, list(checkpoints), num_evals, True, None, None, False, False)
    errors = [float(r[0][0]) for r in results]
    out = {"env": env_name, "seed": seed, "arm": arm, "checkpoints": list(checkpoints), "mse": errors,
           "standardized": [(e - goal.norm_mu) / goal.norm_sigma for e in errors],
           "queries": [str(q) for q in queries], "observations": [str(o) for o in observations],
           "valid_measurements": int(sum(map(bool, successes))), "calls": calls["n"],
           "transport_retries": calls.get("retries", 0)}
    if arm != "flat":
        X = [np.atleast_1d(np.asarray(p[0], dtype=float)) for p in goal.eval_points]
        gts = np.array([float(p[-1]) for p in goal.eval_points])
        forest_only = np.array([scientist.state.prediction(x) for x in X])
        out.update(forest_only_mse_final=float(np.mean((forest_only - gts) ** 2)), forest_log=scientist.log,
                   final_view=scientist.state.view(), forest_events=len(scientist.state.forest.log))
    return out


# ------------------------------------------------------------ stand-in model
FAMILIES = {1: [("a + b*x", {}), ("a + b*x + c*x**2", {}), ("a - b*c**x", {"c": 0.5}),
                ("exp(a + b*x + c*x**2 + d*x**3)", {"a": 3.0, "b": 0.0, "c": 0.0, "d": 0.0})],
            2: [("a", {}), ("a + b*x1 + c*x2", {}), ("b + a/(m + (x1-u)**2 + (x2-v)**2)", {"u": 0.0, "v": 0.0})]}


def scripted_call(system, payload):
    """No model: proposes fixed families once, answers with the forest's prediction, picks the middle input."""
    if "conversation" in payload:
        last = payload["conversation"][-1]["content"]
        return {"reply": "<thought>scripted</thought><answer>1.0</answer>" if "Predict" in last
                else "<thought>scripted</thought><observe>2.5</observe>" if "x1" not in system
                else "<thought>scripted</thought><observe>[0.5, 0.5]</observe>"}
    if "question" in payload:
        return {"answer": payload["forest"].get("credence_weighted_median_prediction", 1.0)}
    known = {row["law"] for row in payload["forest"]["laws"]} | set(payload["forest"]["contradicted"])
    dim = 2 if "x1, x2" in system else 1
    return {"laws": [{"expr": e, "init": i} for e, i in FAMILIES[dim] if e not in known and not payload["forest"]["laws"]]}


_CALL = None


def _one(job):
    try:
        return episode(_CALL, *job)
    except Exception as exc:  # noqa: BLE001 - an invalid episode is recorded and rerun, not fatal
        return {"env": job[0], "seed": job[1], "arm": job[2], "invalid": True, "error": f"{type(exc).__name__}: {exc}"[:300]}


def main(call=None, argv=None):
    global _CALL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", default="dugongs,peregrines,location_finding")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--arms", default="flat,delegated")
    ap.add_argument("--checkpoints", default="0,5,10")
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--boxing-gym", default="boxing-gym", help="path to a boxing-gym checkout (for run_experiment.py)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.boxing_gym)
    _CALL = call or scripted_call
    cps = tuple(int(c) for c in a.checkpoints.split(","))
    done = set()
    if a.out and Path(a.out).exists():
        done = {(e["env"], e["seed"], e["arm"]) for e in map(json.loads, open(a.out)) if not e.get("invalid")}
    jobs = [(e, s, arm, cps, a.evals, a.window) for e in a.envs.split(",")
            for s in range(a.seed0, a.seed0 + a.seeds) for arm in a.arms.split(",") if (e, s, arm) not in done]
    sink = open(a.out, "a") if a.out else None
    with multiprocessing.get_context("fork").Pool(a.jobs) as pool:
        for ep in pool.imap_unordered(_one, jobs):
            print(json.dumps({k: ep.get(k) for k in ("env", "seed", "arm", "standardized", "calls", "error")}), flush=True)
            if sink:
                sink.write(json.dumps(ep) + "\n")
                sink.flush()


if __name__ == "__main__":
    main()
