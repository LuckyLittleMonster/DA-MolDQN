"""BDE_IP reward: weighted combine of scaled BDE + IP + RRAB.

Owns the ``BDEPredictor`` / ``IPPredictor`` wiring (wrapped with ``cached`` for
generic dedup + index-mapping) and the reward-combination math. BDE is memoized
by canonical SMILES via the shared ``bde_cache``; IP is never cached (its value
depends on a random ETKDG conformer).
"""
from rdkit import Chem

from src.cache import cached, LRUCache
from src.reward.bde.predictor import BDEPredictor
from src.reward.ip.predictor import IPPredictor


class BdeIpReward:
    """docstring for BdeIpReward"""

    def __init__(self, config, device, init_mols, bde_cache):
        self.device = device
        self.init_mols = init_mols
        self.bde_cache = bde_cache

        reward_cfg = config.reward
        env_cfg = config.env
        self.bde_factor = reward_cfg.bde_factor
        self.ip_factor = reward_cfg.ip_factor
        self.ip_ensemble = reward_cfg.ip_ensemble
        self.reward_of_invalid_mol = reward_cfg.reward_of_invalid_mol

        # Named reward weights (no list-length magic).
        weights = reward_cfg.bde_ip
        self.bed_weight = weights.bde
        self.ip_weight = weights.ip
        self.rrab_weight = weights.rrab

        self.use_bde_cache = env_cfg.cache.bde
        # IP caching is OFF by default (cache.ip=false): IP depends on a random
        # ETKDG conformer, so a single cached value is a frozen noisy sample
        # (see docs/ip-stability-verification.md). cache.ip=true enables it for
        # the cached-vs-fresh A/B study only.
        self.use_ip_cache = env_cfg.cache.ip
        self.etkdg_max_attempts_cache = env_cfg.etkdg.max_attempts_cache
        self.etkdg_max_attempts_uncache = env_cfg.etkdg.max_attempts_uncache

        # Intra-rank ETKDG threading: RDKit's EmbedMolecule releases the GIL,
        # so a thread pool parallelizes 3D embedding within one rank
        # (~1.75x at 2 threads). Lets 36 ranks x 2 threads keep 72-core etkdg
        # throughput while halving GPU contexts (fits memory + MPS works).
        self.etkdg_threads = env_cfg.etkdg.threads
        # Identity-based ETKDG seed makes IP reproducible per molecule, which is
        # what makes cache.ip exact (cache == recompute). See etkdg.py.
        self.etkdg_deterministic_seed = env_cfg.etkdg.deterministic_seed

        # Pure predictors, wrapped by ``cached`` for generic dedup +
        # index-mapping. BDE gets a swappable cache (LRU); IP gets cache=None
        # (never cached) and call_on_empty=False.
        self.bde_pred = BDEPredictor(device=self.device, bde_factor=self.bde_factor)
        self.bde_scaler = self.bde_pred.bde_scaler
        self.bde_model = self.bde_pred.bde_model
        self.bde = cached(
            self.bde_pred.predict_BDE, cache=self.bde_cache,
            invalid_value=self.reward_of_invalid_mol)

        self.ip_pred = IPPredictor(
            device=self.device,
            ip_factor=self.ip_factor,
            ip_ensemble=self.ip_ensemble,
            etkdg_threads=self.etkdg_threads,
            etkdg_max_attempts_cache=self.etkdg_max_attempts_cache,
            etkdg_max_attempts_uncache=self.etkdg_max_attempts_uncache,
            etkdg_deterministic_seed=self.etkdg_deterministic_seed)
        self.ip_scaler = self.ip_pred.ip_scaler
        self.ip_model = self.ip_pred.ip_model
        _ip_attempts = self.etkdg_max_attempts_uncache
        self.ip_cache = (LRUCache(env_cfg.lru_cache_capacity * len(self.init_mols))
                         if self.use_ip_cache else None)
        self.ip = cached(
            lambda keys, mols: self.ip_pred.predict_IP(mols, _ip_attempts),
            cache=self.ip_cache, call_on_empty=False,
            invalid_value=self.reward_of_invalid_mol)

        self.init_mols_n = [m.GetNumAtoms() + m.GetNumBonds() for m in self.init_mols]

    def calc_rrabs(self, molecules):
        rrabs = []
        for molecule, init_mol_n in zip(molecules, self.init_mols_n):
            n = molecule.GetNumAtoms() + molecule.GetNumBonds()
            rrab = float(init_mol_n - n) / float(init_mol_n)
            rrabs.append(rrab)
        return rrabs

    def compute(self, molecules, current_step, max_steps):

        # remove duplicated smiles.
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        smiles_p = {}
        for i, s in enumerate(smiles):
            if s in smiles_p:
                smiles_p[s][1].append(i)
            else:
                # Rebuild the molecule from its canonical SMILES before AddHs +
                # ETKDG. cenv's C++ reaction editing can leave a double bond in an
                # inconsistent stereo state (marked STEREOZ but with an empty
                # stereoatoms list); RDKit's EmbedMolecule then dereferences the
                # empty stereoatoms array and SEGFAULTS — an uncatchable crash that
                # kills the whole (possibly multi-node) job. A fresh MolFromSmiles
                # re-perceives stereochemistry consistently and avoids it
                # (SanitizeMol alone does NOT — it doesn't touch stereoatoms).
                # For valid molecules this is a canonical round-trip identity, so
                # rewards match the old path; unparseable mols are dropped (invalid).
                m_clean = Chem.MolFromSmiles(s)
                if m_clean is None:
                    continue
                mol_with_H = Chem.AddHs(m_clean)
                smiles_p[s] = mol_with_H, [i]

        bde_ps, bde_vs = self.bde(smiles, smiles_p, use_cache=self.use_bde_cache)
        # ignore mols without valid BDE while predicting IP
        for s, v in zip(smiles, bde_vs):
            if (not v) and s in smiles_p:
                del smiles_p[s]
        # IP: use_cache=False (default) -> fresh ETKDG conformer every visit;
        # use_cache=True (cache.ip) -> frozen first-visit single-conformer IP.
        ip_preds, ip_vs = self.ip(smiles, smiles_p, use_cache=self.use_ip_cache)

        rrabs = self.calc_rrabs(molecules)

        rewards = []
        for bdep, bdev, ipp, ipv, rrab in zip(bde_ps, bde_vs, ip_preds, ip_vs, rrabs):
            if bdev and ipv:
                bden = self.bde_scaler.transform([[bdep * self.bde_factor]])
                ipn = self.ip_scaler.transform([[ipp * self.ip_factor]])
                bde = bden[0][0]
                ip = ipn[0][0]
                r = 2.0 * (self.bed_weight * (1.0 - bde) + self.ip_weight * ip) + self.rrab_weight * rrab
                rewards.append(r)
            else:
                rewards.append(self.reward_of_invalid_mol)

        return {'reward':rewards, 'BDE':bde_ps, 'IP':ip_preds, 'RRAB': rrabs}
