"""Phase 2 smoke test: all moved modules import cleanly inside the src/ package."""
import importlib
import pytest

# Light modules (no torch/cenv side effects at import).
# NB: src.filter is an orphan (nobody imports it) and references a missing module
# `similarity_filter`; it is excluded here and removed in Phase 7 cleanup.
LIGHT = [
    "src.config",
    "src.utils",
    "src.eval",
    "src.shared_adam",
    "src.models.dqn",
]

# Heavy modules: importing them constructs the C++ cenv environment (environment.py)
# and pulls src.reward.bde (agent.py). Require the compiled csrc.cenv + torch runtime.
HEAVY = [
    "src.environment",
    "src.agent",
]


@pytest.mark.parametrize("mod", LIGHT)
def test_import_light(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod", HEAVY)
def test_import_heavy(mod):
    importlib.import_module(mod)
