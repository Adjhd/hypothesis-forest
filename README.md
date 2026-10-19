# hypothesis-forest

A minimal, dependency-free implementation of the **hypothesis forest**: a shared, append-only state for
scientific hypotheses that agents propose, test, split, merge and transfer, and that a controller commits.
The whole library is about 700 lines of Python, written to be read and debugged end to end.

```
hforest/
  scope.py      where a claim holds; relation between two scopes
  forest.py     the forest, its 14 growth rules, the append-only log and replay
  merge.py      fork a forest for an agent; admit its proposals back into the canonical forest
  priors.py     turn the forest into role-labelled prior cards for a new task
  promotion.py  paired sign-test gate that authorizes a hypothesis for use
examples/steel_demo.py   formation, evolution and merge on a toy weathering-steel forest
tests/test_forest.py     the rules as executable checks
```

## Quickstart

```bash
python examples/steel_demo.py          # Python >= 3.10, no dependencies
python -m unittest discover tests
```

## The object

A forest holds hypotheses, plans, records, assessments and relations. Every change is an event in a log:
`{seq, parent, head, kind, body}` with `head = sha256(parent + event)[:12]`. The log *is* the forest:
`Forest.from_events(log)` replays it, re-validates every edit against the state at that point, and reproduces
the same head. A tampered or reordered log fails to replay.

* **Hypothesis**: statement, scope, predictions, falsifiers, mechanism, variables, endpoints, plans, status, revision,
  authorized scope.
* **Scope**: `dims` (a value, a set of values, or an interval `{"min", "max"}`), `exclusions`, `unknowns`.
  `relation(claim, target)` returns one of `equivalent, covers, overlaps, unknown, disjoint`, worst dimension first.
* **Record**: an observation with a tier `E0`-`E4` and a kind (`scientific`, `engineering`, `infrastructure`).
  Only an `E4` record, which admits an existing `E2`/`E3` scientific record, can move a hypothesis's status.
* **Assessment**: a verdict of one record on one hypothesis revision for an explicit target scope.

## Growth rules

Every edit names a trigger. The trigger fixes which operations are allowed, which statuses are reachable, and
whether admitted evidence is required. Anything else is rejected with `InvalidEdit`.

| trigger | operations | reachable status | admitted evidence |
|---|---|---|---|
| unexplained_observation | propose | - | no |
| refined_content | revise | revised | no |
| admitted_support_in_scope | attach_evidence | tested, supported | yes |
| admitted_contradiction_in_scope | attach_evidence, narrow_scope | tested, contradicted, scoped | yes |
| condition_dependent_outcome | split, narrow_scope | scoped | yes |
| unresolved_competition | link_competition | - | no |
| equivalent_hypotheses | merge | - | no |
| invalid_experiment | record_event | - | no |
| execution_blocked | record_event | blocked | no |
| stale_without_budget | suspend | frozen | no |
| relevant_new_evidence | reactivate | reactivated | no |
| cross_scope_request | propose_transfer | - | no |
| promotion_passed | authorize | - | yes |
| obsolete | retire | retired | no |

Two consequences follow directly from the table and are checked in the tests:

1. **No refutation by failure.** Only triggers that require admitted evidence can reach `contradicted` or `scoped`,
   so a failed or blocked experiment is recorded but never weakens a hypothesis.
2. **No inheritance by transfer.** `propose_transfer` adds a candidate relation with a bridge plan; it changes neither
   the status nor the authorized scope of anything.

## Three processes

**Formation.** An agent works in `fork(canonical)` and proposes freely there. `merge(fork, canonical)` replays the
fork's events onto the current canonical forest. Each proposal is resolved by content key
(mechanism + variables + endpoints) and scope: `new`, `refines`, `derived_from`, `alias` (same proposition and scope;
its later edits are redirected to the existing node) or `archive` (no new test or plan, or no identity to resolve).
Only actionable proposals, those tested in the fork or carrying a valid plan, enter.

**Evolution.** Admitted, in-scope evidence moves status (`proposed -> tested -> supported / contradicted -> scoped`).
Evidence whose scope only overlaps the claim needs an explicit `target_scope` or a written `scope_review`;
disjoint evidence is rejected. A condition-dependent outcome splits a hypothesis into children that partition one
dimension of its scope.

**Merge and use.** Agents never write the canonical forest: it accepts only edits committed by the controller,
and `merge` commits each admitted fork edit as the controller while keeping the agent as its `author`.
`merge(..., budget=n)` admits at most `n` new hypotheses per fork, so growth stays bounded. Edits from a fork are re-validated against the canonical state at merge time, so an edit made
against a stale base is rejected rather than silently applied. A hypothesis becomes usable as a direct prior only
after `paired_gate` accepts it and `authorize` sets an authorized scope inside its validity scope; `retrieve`
returns at most one card per role first (`direct_prior`, `negative_prior`, `in_scope_unvalidated`,
`transfer_candidate`, ...) and marks `inherits_conclusion` only for direct priors.

## Design rules for contributors

* The log is the source of truth; state is derived. Never mutate state outside `Forest._apply`.
* A new behaviour is a new row in `RULES` plus one `_check_<op>` and one branch in `_apply`, with a test.
* Keep it small: no dependencies, no I/O beyond JSONL, no domain code in `hforest/`.

## License

MIT, see `LICENSE`.
