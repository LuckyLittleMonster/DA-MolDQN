"""Shared entry logic for train / finetune / testing."""
from src.launch import get_launcher
from src.trainer import Trainer


def _load_init_mols_from_path(cfg):
    with open(cfg.init_mol_path) as f:
        mols = [line.strip() for line in f if line.strip()]
    start = cfg.init_mol_start or 0
    return mols[start:]


def resolve_init_mols(cfg, rank, world_size):
    if cfg.init_mol:
        init_mols = list(cfg.init_mol)
    elif cfg.init_mol_path:
        init_mols = _load_init_mols_from_path(cfg)
    else:
        init_mols = []
    num = cfg.num_init_mol if cfg.num_init_mol else len(init_mols)
    per = max(1, num // world_size)
    bid = min(rank * per, len(init_mols))
    eid = min((rank + 1) * per, len(init_mols))
    return init_mols[bid:eid]


def _worker(rank, world_size, cfg, mode):
    init_mols = resolve_init_mols(cfg, rank, world_size)
    Trainer(cfg, rank, world_size, init_mols, mode=mode).run()


def run_entry(cfg, mode):
    launcher = get_launcher(cfg.starter, world_size=cfg.mp_world_size,
                            master_port=str(cfg.mp_master_port))
    launcher.run(lambda rank, ws: _worker(rank, ws, cfg, mode), cfg)
