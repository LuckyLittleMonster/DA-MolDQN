"""Shared entry logic for train / finetune / testing.

The Hydra ``DictConfig`` is parsed ONCE here into the typed :class:`Config`
(``src.config``); everything downstream (launchers, Trainer, env, rewards, agent)
uses the typed config, NOT ``cfg.X`` / ``hyp.X``.
"""
from omegaconf import OmegaConf

from src.config import parse_config
from src.launch import get_launcher
from src.trainer import Trainer


def _load_init_mols_from_path(config):
    with open(config.mols.init_mol_path) as f:
        mols = [line.strip() for line in f if line.strip()]
    start = config.mols.init_mol_start or 0
    return mols[start:]


def resolve_init_mols(config, rank, world_size):
    if config.mols.init_mol:
        init_mols = list(config.mols.init_mol)
    elif config.mols.init_mol_path:
        init_mols = _load_init_mols_from_path(config)
    else:
        init_mols = []
    num = config.mols.num_init_mol if config.mols.num_init_mol else len(init_mols)
    per = max(1, num // world_size)
    bid = min(rank * per, len(init_mols))
    eid = min((rank + 1) * per, len(init_mols))
    return init_mols[bid:eid]


def _worker(rank, world_size, config, mode, config_yaml):
    init_mols = resolve_init_mols(config, rank, world_size)
    Trainer(config, rank, world_size, init_mols, mode=mode,
            config_yaml=config_yaml).run()


def run_entry(cfg, mode):
    config = parse_config(cfg)
    # Keep the resolved Hydra yaml verbatim for the run's config.yaml artifact.
    config_yaml = OmegaConf.to_yaml(cfg)
    starter = config.dist.starter
    starter_name = starter.value if starter is not None else None
    launcher = get_launcher(starter_name, world_size=config.dist.mp_world_size,
                            master_port=str(config.dist.mp_master_port))
    launcher.run(lambda rank, ws: _worker(rank, ws, config, mode, config_yaml), config)
