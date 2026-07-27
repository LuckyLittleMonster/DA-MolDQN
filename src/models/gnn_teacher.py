"""Frozen property-GNN teacher for the DQN, ported from the rep_gnn study.

Why this exists (measured in docs/gnn_qnetwork_study.md, Follow-ups F-M; the scaffolding
that produced it lives in rep_gnn/ on the `rep` branch):

* The canonical fingerprint Q-network, AFTER Bellman training, ranks a step's candidate
  molecules at Spearman ~0 against the true reward -- no better than chance. The signal is
  destroyed by a signal-to-noise collapse: ``max_a' Q(s',a')`` is evaluated on the ~29/30 of
  candidates that never receive a gradient (replay only stores the CHOSEN molecule), the max
  turns that extrapolation error into a systematic positive bias, and Q inflates until the
  candidate-to-candidate reward differences that carry the ranking are ~10% of its magnitude.
* Training a GNN end-to-end as the Q-network makes it worse, not better: the same noise now
  corrupts the representation too, so a GINE encoder that ranks 0.84 as a property model ends
  up at 0.09 as a Q-network, for 6-8x the wall-clock.
* The fix is not a bigger network. It is to keep a property model OUT of the Bellman loop and
  inject its prediction into Q: as an observation feature, as a zero-initialised additive
  prior (so Q *starts* as the property ranker), and as a distillation target over the WHOLE
  candidate set (which is what repairs the coverage hole). Costs no extra oracle calls.

Measured effect in the standalone harness (QED, zinc-pool starts, 20 steps, 5 seeds):
population mean +40% over the canonical baseline and >=4x fewer oracle calls to a given
quality; on the exact production BDE_IP reward with anti_400 starts, top-10 +10%.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn

from src.features.gnn_featurize import mol_to_graph


class GNNTeacher:
    """Frozen encoder + frozen property head -> per-candidate observation.

    ``observe(mols, steps_left)`` returns ``[embedding, step, r_hat]`` where ``r_hat`` is the
    teacher's prediction of the environment's immediate reward. ``r_hat`` MUST reproduce the
    true reward: an uncalibrated one actively misleads the distillation (measured: population
    0.55 vs 0.92 when the applicability-domain factor was left out of it).
    """

    def __init__(self, ckpt_path, device, reward_kind="bde_ip", init_size=None):
        from torch_geometric.data import Batch  # local: only needed on the GNN path
        self._Batch = Batch
        from src.models.gnn_encoder import GNNQED

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = GNNQED(hidden=ck["hidden"], num_layers=ck["layers"],
                       bounded=ck.get("bounded", False), arch=ck.get("arch", "gine"))
        model.load_state_dict(ck["state_dict"])
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device
        self.reward_kind = reward_kind
        # de-standardisation of the head's output (labels were z-scored unless bounded)
        self.mu, self.sd = ck.get("mu"), ck.get("sd")
        if ck.get("bounded"):
            self.mu = self.sd = None
        self.emb_dim = self.model.encoder.out_dim
        self.obs_dim = self.emb_dim + 2          # + step + r_hat
        # Terms of the TRUE reward that the teacher was never trained on. These MUST be
        # added, or the better-ranking agent optimises only the modelled part and drifts on
        # the rest. Measured on gnn_qed.pt: r_hat correlates +0.968 with pure QED but only
        # +0.363 with the production reward 0.8*QED - 0.2*SA, and the resulting agent grew
        # molecules from 16.6 to 19.5 heavy atoms (SA rises, production reward -32%). On
        # BDE_IP the omitted +0.5*rrab REWARDS shrinking, so the agent went the other way,
        # to 9.89 heavy atoms with n_unique collapsing 0.586 -> 0.193.
        #
        # Both take an RDKit Mol. The trainer assigns them from make_corrections() after
        # construction. Episode-dependent parts (bde_ip's rrab is relative to the EPISODE's
        # start molecule) read self.init_size, which the trainer refreshes each episode --
        # so these corrections must never be cached alongside the molecule-only r_hat.
        self.scale_fn = None              # multiplicative: r_hat *= scale_fn(mol)
        self.extra_fn = None              # additive:       r_hat += extra_fn(mol)
        self.init_size = init_size

    @torch.no_grad()
    def _encode(self, mols):
        """Returns (embeddings [N,E], r_hat_molecule_part [N])."""
        graphs = [mol_to_graph(m) for m in mols]
        batch = self._Batch.from_data_list(graphs).to(self.device)
        emb = self.model.encoder(batch)
        out = self.model.head(emb).reshape(-1)
        if self.model.bounded:
            out = torch.sigmoid(out)
        r = out if self.sd is None else out * self.sd + self.mu
        return emb.cpu().numpy(), r.cpu().numpy()

    @torch.no_grad()
    def observe(self, mols, steps_left):
        emb, r = self._encode(mols)
        # order matters: the reward applies its size factor to the property score, so scale
        # first (bde_ip2: reward = prop * d(size)) then add episode-dependent terms
        # (bde_ip: reward = 2*prop + 0.5*rrab).
        if self.scale_fn is not None:
            r = r * np.asarray([self.scale_fn(m) for m in mols], dtype=np.float32)
        if self.extra_fn is not None:
            r = r + np.asarray([self.extra_fn(m) for m in mols], dtype=np.float32)
        step = np.full((len(mols), 1), float(steps_left), dtype=np.float32)
        return np.concatenate(
            [emb.astype(np.float32), step, r.reshape(-1, 1).astype(np.float32)], axis=1)


def make_corrections(reward_type, config, teacher):
    """Return (scale_fn, extra_fn) so that r_hat reproduces the FULL reward.

    r_hat is applied as ``r_hat * scale_fn(m) + extra_fn(m)``; either may be None.
    The teacher checkpoints predict only the property part, so every reward term the
    teacher never saw has to be reconstructed here -- otherwise the better-ranking agent
    optimises the modelled part and drifts on the rest.

    * ``bde_ip``  : additive + rrab_weight * (init_size - size(m)) / init_size, with size
                    counted as atoms + bonds exactly as bde_ip.py:82-90 does and init_size
                    the EPISODE's start molecule (refreshed per episode by the trainer).
    * ``bde_ip2`` : reward = prop * d(size) while gnn_bdeipprod.pt was trained on 2*prop, so
                    r_hat needs BOTH a 0.5 rescale and the multiplicative d(size). Measured
                    without it: r_hat is 2.08x too large, MAE 0.498.
    * ``qed``     : additive - sa_weight * SA(m), a pure molecule function.
    * ``plogp``   : nothing -- plogp_value already contains SA and the ring penalty and the
                    teacher is trained on the whole thing.
    """
    if reward_type == "bde_ip":
        w = config.reward.bde_ip.rrab

        def rrab(mol):
            n = mol.GetNumAtoms() + mol.GetNumBonds()
            base = teacher.init_size or n
            return w * float(base - n) / float(base)
        return None, rrab
    if reward_type == "bde_ip2":
        w = config.reward.bde_ip

        def d_size(mol):
            z = (mol.GetNumHeavyAtoms() - w.size_n_max) / w.size_k
            d = 0.0 if z > 60.0 else 1.0 / (1.0 + math.exp(z))
            # 0.5 undoes the teacher's 2*prop training target
            return 0.5 * max(w.size_floor, d)
        return d_size, None
    if reward_type == "qed":
        from src.reward.sa import sa_score
        w = config.reward.qed.sa

        def sa_pen(mol):
            return -w * float(sa_score(mol))
        return None, sa_pen
    return None, None


class MolDQNPrior(nn.Module):
    """Q-head that predicts a RESIDUAL around r_hat, the LAST observation feature.

    The final layer is zero-initialised, so at step 0 the Q-function is EXACTLY the frozen
    teacher's reward prediction: the agent begins already ranking candidates as well as the
    property model instead of at random, and Bellman training can only add a bounded
    correction on top. Removing this and keeping only the distillation loss measured WORSE
    than doing nothing (rho -0.08 vs +0.24) -- the head then has to memorise the teacher with
    the same weights the Bellman loss is fighting over.
    """

    def __init__(self, input_length, output_length=1, hidden=512):
        super().__init__()
        last = nn.Linear(hidden // 2, output_length)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(
            nn.Linear(input_length, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
            last,
        )

    def forward(self, x):
        return self.net(x) + x[:, -1:]

    def training_step(self, *a, **kw):
        pass
