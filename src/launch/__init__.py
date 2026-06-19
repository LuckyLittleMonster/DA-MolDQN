"""Launcher registry."""
from src.launch.base import Launcher, SingleLauncher, TorchrunLauncher, SlurmLauncher
from src.launch.fork import ForkLauncher


def get_launcher(name: str | None, **kwargs) -> Launcher:
    """Return a launcher by starter name.

    name in {None, 'single'} -> single process.
    'torchrun' / 'slurm'     -> env-resolved.
    'fork' / 'spawn' / 'forkserver' -> local multiprocess spawner (needs world_size).
    """
    if name in (None, "single"):
        return SingleLauncher()
    if name == "torchrun":
        return TorchrunLauncher()
    if name == "slurm":
        return SlurmLauncher()
    if name in ("fork", "spawn", "forkserver"):
        return ForkLauncher(
            world_size=kwargs["world_size"],
            start_method=name,
            master_port=str(kwargs.get("master_port", "12355")),
        )
    raise ValueError(f"unknown launcher/starter: {name!r}")


__all__ = ["get_launcher", "Launcher"]
