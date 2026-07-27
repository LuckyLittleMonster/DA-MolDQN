"""Compare arms on the FINAL molecule of the FINAL episode -- the deterministic output of
the converged policy -- instead of max-statistics over the whole run.

Why this is the stricter metric: `best` and `top100` are maxima over every molecule visited
during training, so they are dominated by whether an epsilon-greedy exploration step happened
to get lucky at some point. With eps_decay=0.968 epsilon is 0.0015 by episode 200, so the last
episode is effectively pure greedy: its final molecule is what the LEARNED POLICY produces
from each starting molecule, with no exploration randomness in it.

The reward of that molecule has to be RECOMPUTED, because the pipeline never scores it:
src/trainer.py:160-161 appends `rewards` after the step loop, and `find_reward()` evaluated
`self.states` at st = T-1, i.e. s_{T-1}. So metrics['rewards']['reward'][-1] and
paths[...]['last'][-1] both carry the PENULTIMATE molecule's score (task #6). Recomputing
also means the two arms are scored by one identical oracle call here, not by whatever their
own runs happened to record.
"""
import argparse
import gzip
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def final_molecules(trial, exp):
    """FINAL molecule of the FINAL recorded episode, one per molecule slot, all ranks."""
    p = os.path.join(HERE, "Experiments", f"{exp}_{trial}", f"{exp}_{trial}.pickle.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rb") as f:
        d = pickle.load(f)
    out = []
    for _, P in sorted(d["paths"].items()):
        last = P.get("last")
        if not last:
            continue
        mols_per_slot, _meta = last[-1]          # final recorded episode
        for path in mols_per_slot:
            if path:
                out.append(path[-1])             # the actual final molecule
    return out


def score(mols, device, reward_type, ip_ensemble):
    """Score with the production reward, one identical oracle call for every arm."""
    from omegaconf import OmegaConf
    from src.config import parse_config
    from src.reward.rewards import make_reward

    cfg = OmegaConf.load(os.path.join(HERE, "configs", "config.yaml"))
    OmegaConf.update(cfg, "reward.type", reward_type, force_add=True)
    OmegaConf.update(cfg, "reward.ip_ensemble", ip_ensemble, force_add=True)
    OmegaConf.update(cfg, "env.etkdg.threads", 2, force_add=True)
    config = parse_config(cfg)
    # rrab is relative to the episode's start molecule; scoring molecules out of episode
    # context, use each molecule as its own reference so rrab = 0 and the comparison is on
    # the BDE/IP part alone (bde_ip2 has no rrab term at all).
    r = make_reward(config, device, list(mols), None)
    return r.compute(list(mols), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="gnn32s")
    ap.add_argument("--arms", default="baseline:9100,9102,9104;gnn_distill:9101,9103,9105")
    ap.add_argument("--reward", default="bde_ip")
    ap.add_argument("--ip_ensemble", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    arms = {}
    for spec in a.arms.split(";"):
        name, trials = spec.split(":")
        per_seed = []
        for t in trials.split(","):
            mols = final_molecules(int(t), a.exp)
            if mols:
                per_seed.append(mols)
        arms[name] = per_seed
        print(f"{name}: {len(per_seed)} seed(s), "
              f"{[len(m) for m in per_seed]} final molecules each", flush=True)
    if not all(arms.values()):
        print("not enough completed runs"); return

    rows = {}
    for name, seeds in arms.items():
        per_seed = []
        for mols in seeds:
            out = score(mols, a.device, a.reward, a.ip_ensemble)
            r = np.asarray(out["reward"], dtype=float)
            ok = r > -999
            heavy = np.array([m.GetNumHeavyAtoms() for m in mols], dtype=float)
            ring3 = np.array([
                (min((len(x) for x in m.GetRingInfo().AtomRings()), default=0) == 3)
                for m in mols], dtype=float)
            per_seed.append({
                "final_reward_mean": float(r[ok].mean()) if ok.any() else float("nan"),
                "final_reward_median": float(np.median(r[ok])) if ok.any() else float("nan"),
                "final_reward_best": float(r[ok].max()) if ok.any() else float("nan"),
                "valid_frac": float(ok.mean()),
                "heavy_atoms": float(heavy.mean()),
                "ring3_frac": float(ring3.mean()),
                "n_unique": float(len({Chem.MolToSmiles(m) for m in mols})) / len(mols),
            })
        rows[name] = per_seed

    keys = ["final_reward_mean", "final_reward_median", "final_reward_best",
            "valid_frac", "heavy_atoms", "ring3_frac", "n_unique"]
    names = list(rows)
    print(f"\nFINAL molecule of the FINAL episode, reward RECOMPUTED ({a.reward})")
    print(f"{'metric':22s}" + "".join(f"{n:>20s}" for n in names) + f"{'delta':>10s}{'sep?':>13s}")
    print("-" * (22 + 20 * len(names) + 23))
    for k in keys:
        vals = {n: np.array([s[k] for s in rows[n]]) for n in names}
        line = f"{k:22s}" + "".join(f"{vals[n].mean():12.4f}±{vals[n].std():.4f}" for n in names)
        if len(names) == 2:
            a0, b0 = vals[names[0]], vals[names[1]]
            sep = "SEPARATED" if abs(b0.mean() - a0.mean()) > (a0.std() + b0.std()) else "overlapping"
            d = 100 * (b0.mean() - a0.mean()) / abs(a0.mean()) if a0.mean() else float("nan")
            line += f"{d:+9.1f}%{sep:>13s}"
        print(line)
    print("\nper-seed final_reward_mean:")
    for n in names:
        print(f"  {n:14s} " + "  ".join(f"{s['final_reward_mean']:.4f}" for s in rows[n]))


if __name__ == "__main__":
    main()
