"""Fork/spawn/forkserver launcher: spawns ``world_size`` local worker processes.

Replaces the ``Worker(mp.Process)`` + ``__main__`` spawn block of main_hpc.py.
"""
from __future__ import annotations

import os

from src.launch.base import Launcher


def _fork_worker(fn, cfg, rank: int, world_size: int) -> None:
    launcher = ForkLauncher(world_size, start_method=cfg.get("starter", "fork"),
                            master_port=str(cfg.get("mp_master_port", "12355")))
    launcher.setup(
        rank, world_size,
        backend=cfg.get("backend", "gloo"),
        init_method=_init_method(cfg),
    )
    try:
        fn(rank, world_size)
    finally:
        launcher.cleanup()


def _init_method(cfg):
    base = cfg.get("init_method")
    if base is None:
        return None
    return f"{base}_{cfg.get('experiment')}_{cfg.get('trial')}"


class ForkLauncher(Launcher):
    name = "fork"

    def __init__(self, world_size: int, start_method: str = "fork", master_port: str = "12355"):
        self.world_size = world_size
        self.start_method = start_method
        self.master_port = master_port

    def resolve(self) -> tuple[int, int]:
        # rank is assigned per spawned process; not meaningful on the spawner.
        return (None, self.world_size)  # type: ignore[return-value]

    def run(self, fn, cfg) -> None:
        import torch.multiprocessing as mp

        mp.set_start_method(self.start_method, force=True)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ.setdefault("MASTER_PORT", self.master_port)
        procs = [
            mp.Process(target=_fork_worker, args=(fn, cfg, rank, self.world_size))
            for rank in range(self.world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
