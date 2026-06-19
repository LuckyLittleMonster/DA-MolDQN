"""Unified run output.

All artifacts of one run live under ``{base}/{exp}_{trial}/``:

    {exp}_{trial}/
      config.yaml                 # resolved run config (replaces args.pickle)
      checkpoints/
        model_dqn.pth  model_target_dqn.pth
      _rank{r}_metrics.pickle     # transient per-rank shards (removed by merge)
      _rank{r}_paths.pickle
      {exp}_{trial}.pickle.gz     # final: all ranks' metrics + paths, gzip-compressed

``Recorder`` replaces the scattered ``pickle.dump`` calls in the old
``main_hpc.py`` and absorbs ``scripts/merge_pickles.py`` + ``scripts/pickle_to_gz.py``.
"""
from __future__ import annotations

import gzip
import os
import pickle
import glob


class Recorder:
    def __init__(self, base: str, exp: str, trial, rank: int, world_size: int):
        self.base = base
        self.exp = exp
        self.trial = str(trial)
        self.rank = rank
        self.world_size = world_size
        self.run_dir = os.path.join(base, f"{exp}_{self.trial}")
        os.makedirs(self.run_dir, exist_ok=True)
        self._metrics: dict | None = None
        self._paths: dict | None = None

    # ---- recording (in-memory, latest snapshot) ----
    def record_metrics(self, metrics: dict) -> None:
        """Store the latest full metrics snapshot (losses/rewards/times/cache rates)."""
        self._metrics = metrics

    def record_paths(self, top=None, last=None, all_smiles=None) -> None:
        self._paths = {"top": top, "last": last, "all": all_smiles}

    # ---- rank-0-only artifacts ----
    def save_checkpoint(self, dqn_state, target_state, eps_threshold, episode) -> None:
        if self.rank != 0:
            return
        import torch  # lazy: keep module import torch-free for the merge path

        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {"episode": episode, "eps_threshold": eps_threshold, "model_state_dict": dqn_state},
            os.path.join(ckpt_dir, "model_dqn.pth"),
        )
        torch.save(
            {"episode": episode, "eps_threshold": eps_threshold, "model_state_dict": target_state},
            os.path.join(ckpt_dir, "model_target_dqn.pth"),
        )

    def save_config(self, cfg_yaml: str) -> None:
        if self.rank != 0:
            return
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            f.write(cfg_yaml)

    # ---- per-rank shard ----
    def flush(self) -> None:
        if self._metrics is not None:
            with open(self._shard("metrics"), "wb") as f:
                pickle.dump(self._metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self._paths is not None:
            with open(self._shard("paths"), "wb") as f:
                pickle.dump(self._paths, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _shard(self, kind: str) -> str:
        return os.path.join(self.run_dir, f"_rank{self.rank}_{kind}.pickle")

    # ---- merge all rank shards -> single compressed pickle ----
    @staticmethod
    def merge(base: str, exp: str, trial, world_size: int) -> str:
        run_dir = os.path.join(base, f"{exp}_{trial}")
        merged: dict = {"metrics": {}, "paths": {}}
        shards: list[str] = []
        for kind in ("metrics", "paths"):
            for path in sorted(glob.glob(os.path.join(run_dir, f"_rank*_{kind}.pickle"))):
                fname = os.path.basename(path)
                rank = int(fname[len("_rank"):].split("_", 1)[0])
                with open(path, "rb") as f:
                    merged[kind][rank] = pickle.load(f)
                shards.append(path)

        out = os.path.join(run_dir, f"{exp}_{trial}.pickle.gz")
        with gzip.open(out, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

        for path in shards:
            os.remove(path)
        return out
