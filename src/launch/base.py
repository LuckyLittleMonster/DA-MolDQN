"""Dist launcher abstraction.

Extracts the per-starter ``(rank, world_size)`` resolution and process-group
setup that used to be inlined in ``main_hpc.main()`` / ``__main__``.

minimal-communication is unaffected: launchers only resolve identity and bring
up the process group; the once-per-episode DQN sync stays inside the Trainer.
"""
from __future__ import annotations

import datetime
import os
import socket
from abc import ABC, abstractmethod


def find_free_port() -> int:
    """Ask the OS for a currently-free TCP port (for local env:// rendezvous)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Launcher(ABC):
    name = "base"

    @abstractmethod
    def resolve(self) -> tuple[int, int]:
        """Return (rank, world_size) for this process."""

    def setup(self, rank: int, world_size: int, *, backend: str = "gloo",
              init_method: str | None = None, timeout_s: int = 600) -> None:
        import torch.distributed as dist
        if init_method is None:
            # env:// rendezvous. For local runs (single) pick a FREE port so we never
            # collide on a hard-coded one; torchrun/slurm/fork already set MASTER_PORT,
            # so this only fires for a single-process run.
            os.environ.setdefault("MASTER_ADDR", "localhost")
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = str(find_free_port())
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

    def run(self, fn, config) -> None:
        """Single-process path (single/torchrun/slurm): setup -> fn -> cleanup."""
        rank, world_size = self.resolve()
        self.setup(rank, world_size, backend=config.dist.backend.value,
                   init_method=_resolved_init_method(config))
        try:
            fn(rank, world_size)
        finally:
            self.cleanup()


def _resolved_init_method(config):
    base = config.dist.init_method
    if not base:
        return None
    return f"{base}_{config.experiment.experiment}_{config.experiment.trial}"


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
        world = os.environ.get("SLURM_NPROCS") or os.environ["SLURM_NTASKS"]
        return (int(os.environ["SLURM_PROCID"]), int(world))
