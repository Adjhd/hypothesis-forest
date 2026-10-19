"""Hypothesis forests: scope-aware, evidence-gated hypothesis evolution."""
from .forest import RULES, TRANSITIONS, Assessment, Forest, Hypothesis, InvalidEdit, Record, content_key
from .merge import Fork, admission, fork, merge
from .policy import choose_experiment, rival_groups
from .priors import retrieve
from .promotion import paired_gate
from .scope import Scope, relation

__version__ = "0.1.0"
__all__ = ["Assessment", "Forest", "Fork", "Hypothesis", "InvalidEdit", "Record", "RULES", "Scope",
           "TRANSITIONS", "admission", "choose_experiment", "content_key", "fork", "merge", "paired_gate", "relation",
           "retrieve", "rival_groups"]
