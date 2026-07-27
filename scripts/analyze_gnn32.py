"""Aggregate the 32-rank / 128-molecule comparison across seeds.

IMPORTANT: the recorded arrays mix `reward_of_invalid_mol` sentinels (-1000) with IP values
that were never clipped to a physical range (observed span +-5e5). Aggregating them raw gives
population mean -33 and "best" 1448 -- i.e. the OPPOSITE conclusion. Everything here is
computed on the valid subset only, and the invalid fraction is reported as its own metric
because the distilled agent explores harder into molecules the oracles reject.
"""
import argparse
import glob
import gzip
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_paths = []


def load(trial, exp):
    p = os.path.join(HERE, "Experiments", f"{exp}_{trial}", f"{exp}_{trial}.pickle.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rb") as f:
        d = pickle.load(f)
    _paths.append(d.get("paths", {}))
    R, B, I = [], [], []
    for _, mm in sorted(d["metrics"].items()):
        r = mm["rewards"]
        R.append(np.asarray(r["reward"], dtype=float))
        B.append(np.asarray(r["BDE"], dtype=float))
        I.append(np.asarray(r["IP"], dtype=float))
    return np.concatenate(R, 1), np.concatenate(B, 1), np.concatenate(I, 1)


def ring3_frac(paths):
    """Fraction of recorded path molecules containing a 3-membered ring.

    #10 showed the invalid-rate rise is rrab REWARDING shrinkage: with atom count pushed
    down but a ring system to keep, the agent fuses/bridges/contracts rings, and those
    strained systems are exactly what ETKDG cannot embed. So the 3-ring fraction is the
    mechanism-level readout, not just the invalid rate.
    """
    from rdkit import Chem
    n = t3 = 0
    for _, P in paths.items():
        for _, meta in P.get("top", []):
            for m in meta["path"]:
                r = m.GetRingInfo().AtomRings()
                if r:
                    n += 1
                    t3 += int(min(len(x) for x in r) == 3)
    return (t3 / n) if n else float("nan")


def stats(t):
    R, B, I = t
    ok = (R > -999) & np.isfinite(R) & np.isfinite(I) & (I > 50) & (I < 350)
    r = R[ok]
    late = R[-100:]
    late_ok = late[(late > -999) & np.isfinite(late)]
    k = np.argsort(r)[-100:]
    return {
        "pop_mean": float(r.mean()),
        "median": float(np.median(r)),
        "best": float(r.max()),
        "top10": float(np.sort(r)[-10:].mean()),
        "top100": float(np.sort(r)[-100:].mean()),
        "late_pop": float(late_ok.mean()),
        "valid_frac": float(ok.mean()),
        "top100_BDE": float(B[ok][k].mean()),
        "top100_IP": float(I[ok][k].mean()),
        "ring3_frac": ring3_frac(_paths[-1]) if _paths else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="gnn32s")
    ap.add_argument("--base_trials", default="9100,9102,9104")
    ap.add_argument("--gnn_trials", default="9101,9103,9105")
    a = ap.parse_args()

    arms = {}
    for name, trials in (("baseline(FP)", a.base_trials), ("gnn_distill", a.gnn_trials)):
        S = []
        for t in trials.split(","):
            d = load(int(t), a.exp)
            if d is not None:
                S.append(stats(d))
        arms[name] = S
        print(f"{name}: {len(S)} seed(s) loaded")
    if not all(arms.values()):
        print("not enough completed runs yet"); return

    keys = ["pop_mean", "median", "best", "top10", "top100", "late_pop",
            "valid_frac", "ring3_frac", "top100_BDE", "top100_IP"]
    print(f"\n{'metric':16s} {'baseline(FP)':>18s} {'gnn_distill':>18s} {'delta':>10s} {'sep?':>12s}")
    print("-" * 80)
    for k in keys:
        va = np.array([s[k] for s in arms["baseline(FP)"]])
        vb = np.array([s[k] for s in arms["gnn_distill"]])
        sep = "SEPARATED" if abs(vb.mean() - va.mean()) > (va.std() + vb.std()) else "overlapping"
        d = 100 * (vb.mean() - va.mean()) / abs(va.mean()) if va.mean() else float("nan")
        print(f"{k:16s} {va.mean():10.4f}±{va.std():.4f} {vb.mean():10.4f}±{vb.std():.4f} "
              f"{d:+9.1f}% {sep:>12s}")
    print("\nper-seed population mean:")
    for name in arms:
        print(f"  {name:14s} " + "  ".join(f"{s['pop_mean']:.4f}" for s in arms[name]))


if __name__ == "__main__":
    main()
