"""Fork/spawn/forkserver launcher: spawns ``world_size`` local worker processes.

Replaces the ``Worker(mp.Process)`` + ``__main__`` spawn block of main_hpc.py.
"""
from __future__ import annotations

import os

from src.launch.base import Launcher, find_free_port


def _fork_worker(fn, config, rank: int, world_size: int) -> None:
    starter = config.dist.starter
    starter_name = starter.value if starter is not None else "fork"
    launcher = ForkLauncher(world_size, start_method=starter_name,
                            master_port=str(config.dist.mp_master_port))
    launcher.setup(
        rank, world_size,
        backend=config.dist.backend.value,
        init_method=_init_method(config),
    )
    try:
        fn(rank, world_size)
    finally:
        launcher.cleanup()


def _init_method(config):
    base = config.dist.init_method
    if base is None:
        return None
    return f"{base}_{config.experiment.experiment}_{config.experiment.trial}"


class ForkLauncher(Launcher):
    name = "fork"

    def __init__(self, world_size: int, start_method: str = "fork", master_port: str = "12355"):
        self.world_size = world_size
        self.start_method = start_method
        self.master_port = master_port

    def resolve(self) -> tuple[int, int]:
        # rank is assigned per spawned process; not meaningful on the spawner.
        return (None, self.world_size)  # type: ignore[return-value]

    def run(self, fn, config) -> None:
        import torch.multiprocessing as mp

        mp.set_start_method(self.start_method, force=True)
        os.environ["MASTER_ADDR"] = "localhost"
        # Parent picks a free port once; forked children inherit it (same group).
        # An explicit MASTER_PORT in the environment still takes precedence.
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        procs = [
            mp.Process(target=_fork_worker, args=(fn, config, rank, self.world_size))
            for rank in range(self.world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
