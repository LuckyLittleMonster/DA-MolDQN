"""Enums replacing the project's magic values.

Each enum is the single authoritative definition AND the documentation for one
discrete parameter. The str->int mapping for the C++ ``cenv`` boundary lives here
(``ObservationType.cenv_fp_mode``) so it is defined in exactly one place.
"""
from __future__ import annotations

from enum import Enum, IntEnum


class ObservationType(Enum):
    """How the env emits the per-action fingerprint observation.

    Replaces the old ``observation_type`` string + ``use_cxx_incremental_fingerprint``
    int (0/1/2). ``cenv_fp_mode`` is the int that crosses the C++ boundary.
    """

    RDKIT = "rdkit"
    LIST = "list"
    NUMPY = "numpy"
    GNN = "gnn"      # frozen property-GNN teacher observation (see src/models/gnn_teacher.py)

    @property
    def cenv_fp_mode(self) -> int:
        """The old ``use_cxx_incremental_fingerprint`` int for cenv."""
        return {
            ObservationType.RDKIT: 0,
            ObservationType.LIST: 1,
            ObservationType.NUMPY: 2,
            # GNN needs the env to hand back RDKit Mols, same as RDKIT mode
            ObservationType.GNN: 0,
        }[self]


class FpFormat(IntEnum):
    """The old ``get_morgan_fingerprint`` 0/1/2 format selector."""

    NONE = 0
    LIST = 1
    NUMPY = 2


class RewardType(Enum):
    BDE_IP = "bde_ip"
    BDE_IP2 = "bde_ip2"   # multiplicative size desirability + clamped scalers
    QED = "qed"
    PLOGP = "plogp"


class Optimizer(Enum):
    """value == ``torch.optim`` attribute name (used as ``getattr(opt, value)``)."""

    ADAM = "Adam"
    SGD = "SGD"
    RMSPROP = "RMSprop"
    ADAMW = "AdamW"


class Backend(Enum):
    GLOO = "gloo"
    NCCL = "nccl"


class Starter(Enum):
    SINGLE = "single"
    FORK = "fork"
    SPAWN = "spawn"
    FORKSERVER = "forkserver"
    SLURM = "slurm"
    TORCHRUN = "torchrun"


class OHMode(Enum):
    """O-H bond maintenance policy (was the -2/-1/k sentinels)."""

    NO_LIMIT = "none"        # was -2: no limitation
    AT_LEAST_ONE = "exist"   # was -1: every mol must have >= 1 OH bond
    SAME_AS_INIT = "same"    # per-mol target = count_OH(init_mol_i)
    EXACT = "exact"          # target == a fixed count for every mol
