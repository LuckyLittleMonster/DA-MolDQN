"""Typed, single-source, validated config package.

Public API:
  - the enums (``ObservationType``, ``FpFormat``, ``RewardType``, ``Optimizer``,
    ``Backend``, ``Starter``, ``OHMode``);
  - the nested ``@dataclass`` config (``Config`` + the group cfgs);
  - ``parse_config(cfg) -> Config`` builds the typed Config from a Hydra DictConfig;
  - ``ENV_DEFAULTS`` exposes the import-time env constants (atom_types, ring sizes,
    fingerprint radius/length) sourced from the EnvCfg dataclass defaults.
"""
from src.config.enums import (
    Backend,
    FpFormat,
    ObservationType,
    Optimizer,
    OHMode,
    RewardType,
    Starter,
)
from src.config.schema import (
    BdeIpWeights,
    CacheCfg,
    CkptCfg,
    Config,
    DistCfg,
    EnvCfg,
    ENV_DEFAULTS,
    EtkdgCfg,
    ExperimentCfg,
    MaintainOH,
    MolsCfg,
    OptimCfg,
    QedWeights,
    RecordCfg,
    RewardCfg,
    RunCfg,
    TrainCfg,
)
from src.config.parse import parse_config, parse_maintain_oh

__all__ = [
    # enums
    "Backend", "FpFormat", "ObservationType", "Optimizer", "OHMode",
    "RewardType", "Starter",
    # schema
    "BdeIpWeights", "CacheCfg", "CkptCfg", "Config", "DistCfg", "EnvCfg",
    "ENV_DEFAULTS", "EtkdgCfg", "ExperimentCfg", "MaintainOH", "MolsCfg",
    "OptimCfg", "QedWeights", "RecordCfg", "RewardCfg", "RunCfg", "TrainCfg",
    # parse
    "parse_config", "parse_maintain_oh",
]
