"""BDE_IP2 reward: same BDE/IP oracles, but size handled MULTIPLICATIVELY and the
scaled properties clamped.

Three changes from ``BdeIpReward``, each motivated by a measured problem:

1. **Size becomes a bounded multiplicative desirability, not an additive penalty.**
   ``BdeIpReward`` adds ``0.5 * rrab`` with ``rrab = (n_init - n) / n_init``
   (``bde_ip.py:82-90``). That term (a) is *tradeable* — a big molecule can buy back its
   penalty with BDE/IP, (b) grows without bound as the molecule grows, (c) *rewards*
   shrinking (a molecule at n=8 from n_init=14 collects +0.214 even if its chemistry is
   unchanged), and (d) depends on the EPISODE'S START molecule, which is not part of the
   Q-network's observation (``src/features/fingerprints.py:17-26`` = fingerprint + step
   only), so the same molecule carries different rewards on different ranks while all ranks
   train one shared DDP Q-net.

   Here the property score is instead SCALED by ``d(size) in (0, 1]``. This is the design
   QED itself uses — Bickerton's QED contains molecular weight as one of its eight internal
   desirability functions, which is exactly why ``QedReward`` needs no external size term
   (``src/reward/rewards/qed.py:28``). A multiplicative factor cannot be traded away, adds
   no constant to the reward's mean, and is a pure function of the molecule.

2. **Single-sided window.** The goal is to stop molecules GROWING, not to push them toward
   some ideal size: ``Data/anti_400.txt`` spans 8-26 heavy atoms (p10 10, median 13, p90 19),
   so a two-sided window would penalise legitimately small antioxidants. Defaults
   ``n_max=26`` (the dataset maximum, i.e. the largest size chemists already accepted) and
   ``k=3`` give d = 0.98 at the dataset median, 0.50 at n=26, 0.21 at n=30, floor by n>=36.

3. **``bde_n``/``ip_n`` are clamped to [0, 1].** ``sklearn``'s ``MinMaxScaler`` defaults to
   ``clip=False`` (``src/reward/scaler.py:28-35`` fits it on two points), so
   ``BdeIpReward`` extrapolates linearly out of range: a diverged AIMNet IP of 304,727
   became ``ip_n = 3621`` and a reward of **1448**, which entered the replay buffer and the
   Bellman targets. Clamping closes that unbounded reward channel. Note this is a *backstop*
   — the real fix is rejecting the bad conformer upstream (tasks #1/#2).

The additive constant is NOT removed (unlike task #5's recommendation for ``bde_ip``): the
multiplicative form needs ``prop >= 0``, otherwise growing the molecule becomes a way to
shrink a negative reward toward zero. ``prop`` already spans [0, 1] with minimum 0, so there
is no removable offset. The ``2.0`` factor of ``bde_ip`` is dropped purely because it is
redundant, not for conditioning reasons -- scaling a reward is SNR-neutral.

``d_floor`` keeps a non-zero gradient in the oversized region. With ``d_floor=0`` an
oversized molecule scores exactly 0 no matter how good its chemistry is, making that region
an absorbing state the agent can never learn its way out of; at 0.1 the ORDERING among
oversized molecules survives, so there is still a path back.

This reward cannot enforce a HARD size limit — only make growth expensive. A hard cap
belongs in the action space (forbid ``atom_addition`` beyond ``n_max``).
"""
import math

from rdkit import Chem

from src.reward.rewards.bde_ip import BdeIpReward


class BdeIpReward2(BdeIpReward):
    """BDE_IP with a multiplicative size desirability and clamped property scalers."""

    def __init__(self, config, device, init_mols, bde_cache):
        super().__init__(config, device, init_mols, bde_cache)
        w = config.reward.bde_ip
        # size window: heavy-atom count, single-sided sigmoid
        self.size_n_max = getattr(w, "size_n_max", 26.0)
        self.size_k = getattr(w, "size_k", 3.0)
        self.size_floor = getattr(w, "size_floor", 0.1)
        # NOTE on the constant offset (task #5): the `2.0` factor in `bde_ip` is a pure
        # SCALE, and scaling is SNR-neutral -- SNR = spread * (1-gamma) / mean, so multiplying
        # the reward by any c leaves it unchanged. The removable constant in `bde_ip` is the
        # +1.6 coming from `2 * bde_weight * (1 - bde_n)`, i.e. from the `1`, not the `2`.
        #
        # That constant CANNOT be dropped here: the multiplicative size factor requires
        # prop >= 0. With prop in [-0.8, 0.2] a big molecule with bad chemistry would score
        # -0.8 * 0.1 = -0.08 while a small molecule with the same bad chemistry scores
        # -0.8 * 0.99 = -0.79, so GROWING would become a way to escape the penalty. As
        # written, prop = w_bde*(1-bde_n) + w_ip*ip_n already spans [0, 1] with minimum
        # exactly 0 -- there is no additive constant left to remove. Task #5 therefore
        # applies to the additive `bde_ip` only.

    def size_desirability(self, molecule):
        """d(size) in [size_floor, 1]: ~1 for dataset-sized molecules, decaying as they grow."""
        n = molecule.GetNumHeavyAtoms()
        z = (n - self.size_n_max) / self.size_k
        # logistic, guarded against overflow for very large molecules
        d = 0.0 if z > 60.0 else 1.0 / (1.0 + math.exp(z))
        return max(self.size_floor, d)

    def compute(self, molecules, current_step, max_steps):
        # Reuse the parent's prediction pipeline verbatim -- it owns the SMILES dedup, the
        # canonical round-trip that avoids an uncatchable RDKit EmbedMolecule SEGFAULT on
        # inconsistent stereo (bde_ip.py:96-115), and the drop-invalid-BDE-before-IP step.
        # Only the reward COMBINATION differs, so recompute it from the parent's raw outputs.
        base = super().compute(molecules, current_step, max_steps)
        bde_ps, ip_preds = base['BDE'], base['IP']

        rewards, sizes = [], []
        for molecule, bdep, ipp in zip(molecules, bde_ps, ip_preds):
            d = self.size_desirability(molecule)
            sizes.append(d)
            # the parent marks invalid entries with reward_of_invalid_mol in BOTH property
            # lists, so detect invalidity from the raw values rather than re-running oracles
            if bdep == self.reward_of_invalid_mol or ipp == self.reward_of_invalid_mol:
                rewards.append(self.reward_of_invalid_mol)
                continue
            # clamp: MinMaxScaler does NOT clip (src/reward/scaler.py:28-35 fits two points),
            # so out-of-range oracle output would otherwise extrapolate into unbounded reward
            # -- a diverged IP of 304,727 became ip_n = 3621 and a reward of 1448.
            bde = min(1.0, max(0.0, self.bde_scaler.transform([[bdep * self.bde_factor]])[0][0]))
            ip = min(1.0, max(0.0, self.ip_scaler.transform([[ipp * self.ip_factor]])[0][0]))
            # prop in [0, 1]; must stay non-negative or the multiplicative d inverts (see
            # the note in __init__). No 2.0 factor: it is a pure scale and SNR-neutral.
            prop = self.bed_weight * (1.0 - bde) + self.ip_weight * ip
            rewards.append(prop * d)

        return {'reward': rewards, 'BDE': bde_ps, 'IP': ip_preds, 'SIZE_D': sizes}
