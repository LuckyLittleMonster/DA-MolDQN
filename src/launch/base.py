"""Dist launcher abstraction.

Extracts the per-starter ``(rank, world_size)`` resolution and process-group
setup that used to be inlined in ``main_hpc.main()`` / ``__main__``.

minimal-communication is unaffected: launchers only resolve identity and bring
up the process group; the once-per-episode DQN sync stays inside the Trainer.
"""
from __future__ import annotations

import datetime
import os
from abc import ABC, abstractmethod


class Launcher(ABC):
    name = "base"

    @abstractmethod
    def resolve(self) -> tuple[int, int]:
        """Return (rank, world_size) for this process."""

    def setup(self, rank: int, world_size: int, *, backend: str = "gloo",
              init_method: str | None = None, timeout_s: int = 600) -> None:
        import torch.distributed as dist
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=timeout_s),
        )
        dist.barrier()

    def cleanup(self) -> None:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

    def run(self, fn, cfg) -> None:
        """Single-process path (single/torchrun/slurm). fn(rank, world_size)."""
        rank, world_size = self.resolve()
        fn(rank, world_size)


class SingleLauncher(Launcher):
    name = "single"

    def resolve(self) -> tuple[int, int]:
        return (0, 1)


class TorchrunLauncher(Launcher):
    name = "torchrun"

    def resolve(self) -> tuple[int, int]:
        return (int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]))


class SlurmLauncher(Launcher):
    name = "slurm"

    def resolve(self) -> tuple[int, int]:
        return (int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NPROCS"]))
