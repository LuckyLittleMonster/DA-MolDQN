"""Reward strategy interface as a structural Protocol.

A ``Reward`` computes the per-molecule reward dict for one env step. Each concrete
reward (BDE_IP / QED / pLogP) owns its own predictor wiring and weights, so the
``MultiMolecules`` env only delegates to ``compute``.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reward(Protocol):
    """Strategy that turns a list of molecules into a reward dict.

    ``bde_cache`` is the LRU cache object used for BDE memoization (BDE_IP reward)
    or ``None`` for rewards that do not cache. The trainer reads
    ``environment.bde_cache`` for hit-rate logging in every reward mode.
    """

    bde_cache: object | None

    def compute(self, molecules, current_step, max_steps) -> dict:
        """Return the reward dict (keys depend on the concrete reward)."""
        ...
