import logging
import random
import threading

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import preprocessing

from .base import RewardCalculator


def ev2kcal_per_mol(ev):
    return ev * 23.0609


def _get_scaler_cached(path):
    if 'bde' in path:
        data = np.array([[96.58618528], [59.79533261]])
        return preprocessing.MinMaxScaler().fit(data)
    elif 'ip' in path:
        data = np.array([[178.1623553], [110.8306396]])
        return preprocessing.MinMaxScaler().fit(data)
    import csv
    real = []
    with open(path) as f:
        s = csv.reader(f, delimiter="\t")
        next(s)
        for r in s:
            if r[1] != '':
                real.append([float(r[1])])
    return preprocessing.MinMaxScaler().fit(real)


class AimnetNseModel:
    """Picklable wrapper around EnsembleCalculator."""
    def __init__(self, path, device):
        from eval import load_models
        self.path = path
        self.device = device
        self.model = load_models([path]).to(device)

    def __setstate__(self, state):
        from eval import load_models
        self.path = state['path']
        self.device = state['device']
        self.model = load_models([self.path]).to(self.device)

    def __getstate__(self):
        return dict(path=self.path, device=self.device)


class BDEIPReward(RewardCalculator):
    def __init__(self, device, args, init_mols):
        self.device = device
        self.discount_factor = args.discount_factor
        self.bde_factor = args.reward.bde_factor
        self.ip_factor = args.reward.ip_factor
        self.reward_of_invalid_mol = args.reward.reward_of_invalid_mol

        self.bed_weight = 0.8
        self.ip_weight = 0.2
        self.rrab_weight = 0.5

        self.use_bde_cache = 'bde' in args.cache
        self.use_ip_cache = 'ip' in args.cache

        # ETKDG settings from etkdg config
        self.etkdg_max_iterations = args.etkdg.max_iterations
        self.etkdg_timeout = args.etkdg.timeout
        self.etkdg_max_attempts_cache = args.etkdg.max_attempts_cache
        self.etkdg_max_attempts_uncache = args.etkdg.max_attempts_uncache

        # Reward weights
        rw = list(args.reward.reward_weight)
        if len(rw) == 1:
            self.bed_weight = rw[0]
            self.ip_weight = 1.0 - self.bed_weight
        elif len(rw) == 2:
            self.bed_weight = rw[0]
            self.ip_weight = rw[1]
        elif len(rw) >= 3:
            self.bed_weight = rw[0]
            self.ip_weight = rw[1]
            self.rrab_weight = rw[2]

        # BDE model
        self.bde_scaler = _get_scaler_cached('./Data/anti-bde.csv')
        from bde_predictor.predict import BDEModel
        self.bde_model = BDEModel('bde_predictor/weights/bde_db2_model3.npz', device=str(self.device))

        # IP model (AIMNet)
        self.ip_scaler = _get_scaler_cached('./Data/anti-ip.csv')
        self.ip_model = AimnetNseModel('aimnetnse-models/aimnet-nse-cv4.jpt', self.device)

        self.init_mols_n = [m.GetNumAtoms() + m.GetNumBonds() for m in init_mols]

    @property
    def reward_keys(self):
        return ['BDE', 'IP', 'RRAB', 'IP_Probs']

    def compute(self, molecules, bde_cache=None, ip_cache=None, **kwargs):
        return self._find_bde_ip_reward_cache(molecules, bde_cache, ip_cache)

    def compute_overlap(self, molecules, prefetch_fn=None, cleanup_fn=None,
                        bde_cache=None, ip_cache=None):
        return self._find_bde_ip_reward_overlap(
            molecules, prefetch_fn=prefetch_fn, cleanup_fn=cleanup_fn,
            bde_cache=bde_cache, ip_cache=ip_cache)

    # --- BDE prediction ---

    def read_bde_from_df(self, pred, mol=None):
        if len(pred.bde_pred) < 1:
            return 0.0, False
        return min(pred.bde_pred), True

    def predict_BDE(self, smiles, mols):
        return self.bde_model.predict_oh_bde(smiles)

    def predict_BDE_cache(self, smiles, smiles_p, useCache, bde_cache):
        bde_ps = [self.reward_of_invalid_mol for _ in smiles]
        bde_vs = [False for _ in smiles]

        smiles_uncached = []
        mols_uncached = []
        for s, (mol_with_H, ids) in smiles_p.items():
            if useCache:
                p, v = bde_cache.get(s)
                if v:
                    for i in ids:
                        bde_ps[i] = p
                        bde_vs[i] = True
                    continue
            smiles_uncached.append(s)
            mols_uncached.append(mol_with_H)

        if len(smiles_uncached) > 0:
            ucps, ucvs = self.predict_BDE(smiles_uncached, mols_uncached)
            for i, s in enumerate(smiles_uncached):
                p = ucps[i]
                v = ucvs[i]
                if v:
                    if useCache:
                        bde_cache.put(s, p)
                    for mol_idx in smiles_p[s][1]:
                        bde_ps[mol_idx] = p
                        bde_vs[mol_idx] = True
        return bde_ps, bde_vs

    # --- ETKDG conformer generation ---

    def rwmol2data_atts(self, mols, maxAttempts):
        try:
            return self._rwmol2data_atts_parallel(mols, maxAttempts)
        except Exception:
            pass
        return self._rwmol2data_atts_serial(mols, maxAttempts)

    def _rwmol2data_atts_parallel(self, mols, maxAttempts):
        import src.cenv as cenv
        nThreads = min(72, len(mols) * maxAttempts)
        coords_list, success_list, probs_list, natoms_list = cenv.embed_molecules_parallel(
            mols, maxAttempts, nThreads,
            self.etkdg_max_iterations, self.etkdg_timeout)

        data = [dict() for _ in mols]
        success = list(success_list)
        prob = list(probs_list)

        charge = torch.tensor([1, 0, -1]).to(self.device)
        mult = torch.tensor([2, 1, 2]).to(self.device)

        for i in range(len(mols)):
            if success[i]:
                coords = torch.tensor(coords_list[i], dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(self.device)
                numbers = [a.GetAtomicNum() for a in mols[i].GetAtoms()]
                numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(self.device)
                data[i] = dict(coord=coords, numbers=numbers, charge=charge, mult=mult)

        return data, success, prob

    def _rwmol2data_atts_serial(self, mols, maxAttempts):
        data = [dict() for _ in mols]
        success = [False for _ in mols]
        prob = [-1 for _ in mols]

        for i, mol in enumerate(mols):
            try:
                s = 0
                for _ in range(maxAttempts):
                    cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=1)
                    if cid >= 0:
                        s += 1
                        if not success[i]:
                            success[i] = True
                            coords = mol.GetConformer(cid).GetPositions()
                            coords = torch.tensor(coords, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(self.device)
                            numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
                            numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(self.device)
                            charge = torch.tensor([1, 0, -1]).to(self.device)
                            mult = torch.tensor([2, 1, 2]).to(self.device)
                            data[i] = dict(coord=coords, numbers=numbers, charge=charge, mult=mult)
                prob[i] = s / maxAttempts
            except Exception as e:
                logging.warning(f"IP Exception: {e}")
        return data, success, prob

    # --- IP prediction ---

    def predict_IP(self, molecules, maxAttempts):
        ds, vs, probs = self.rwmol2data_atts(molecules, maxAttempts)
        preds = [0.0] * len(molecules)

        groups = {}
        for i, (data, valid) in enumerate(zip(ds, vs)):
            if valid:
                N = data['numbers'].shape[1]
                groups.setdefault(N, []).append((i, data))

        if not groups:
            return preds, vs, probs

        ip_model = self.ip_model
        charge_single = torch.tensor([1, 0, -1], device=self.device)
        mult_single = torch.tensor([2, 1, 2], device=self.device)

        with torch.jit.optimized_execution(False), torch.no_grad():
            for N, items in groups.items():
                indices, datas = zip(*items)
                B = len(items)
                batch_coord = torch.stack([d['coord'] for d in datas], dim=0).reshape(B * 3, N, 3)
                batch_numbers = torch.stack([d['numbers'] for d in datas], dim=0).reshape(B * 3, N)
                batch_charge = charge_single.repeat(B)
                batch_mult = mult_single.repeat(B)
                batch_data = dict(coord=batch_coord, numbers=batch_numbers,
                                  charge=batch_charge, mult=batch_mult)
                pred = ip_model.model(batch_data)
                energy = pred['energy'].detach().cpu().numpy().reshape(B, 3)
                ip_kcal = (energy[:, 0] - energy[:, 1]) * 23.0609
                for batch_i, mol_i in enumerate(indices):
                    preds[mol_i] = float(ip_kcal[batch_i])

        return preds, vs, probs

    def predict_IP_cache(self, smiles, smiles_p, useCache, ip_cache):
        ip_preds = [self.reward_of_invalid_mol for _ in smiles]
        ip_vs = [False for _ in smiles]
        ip_probs = [-1 for _ in smiles]

        maxAttempts = self.etkdg_max_attempts_cache if useCache else self.etkdg_max_attempts_uncache

        smiles_uncached = []
        mols_uncached = []

        for s, (mol_with_H, ids) in smiles_p.items():
            if useCache:
                p, v = ip_cache.get(s)
                if v:
                    (pred, prob) = p
                    for i in ids:
                        if random.random() <= prob:
                            ip_preds[i] = pred
                            ip_vs[i] = True
                            ip_probs[i] = prob
                    continue
            smiles_uncached.append(s)
            mols_uncached.append(mol_with_H)

        if len(smiles_uncached) > 0:
            ucps, ucvs, probs = self.predict_IP(mols_uncached, maxAttempts)
            for i, s in enumerate(smiles_uncached):
                pred = ucps[i]
                v = ucvs[i]
                prob = probs[i]
                if v:
                    if useCache:
                        ip_cache.put(s, (pred, prob))
                    for mol_idx in smiles_p[s][1]:
                        ip_preds[mol_idx] = pred
                        ip_vs[mol_idx] = True
                        ip_probs[mol_idx] = prob
        return ip_preds, ip_vs, ip_probs

    # --- RRAB calculation ---

    def calc_rrabs(self, molecules):
        rrabs = []
        for molecule, init_mol_n in zip(molecules, self.init_mols_n):
            n = molecule.GetNumAtoms() + molecule.GetNumBonds()
            rrab = float(init_mol_n - n) / float(init_mol_n)
            rrabs.append(rrab)
        return rrabs

    # --- Reward assembly ---

    def _compute_reward(self, bde_ps, bde_vs, ip_preds, ip_vs, ip_probs, rrabs):
        rewards = []
        for bdep, bdev, ipp, ipv, ip_prob, rrab in zip(bde_ps, bde_vs, ip_preds, ip_vs, ip_probs, rrabs):
            if bdev and ipv:
                bden = self.bde_scaler.transform([[bdep * self.bde_factor]])
                ipn = self.ip_scaler.transform([[ipp * self.ip_factor]])
                bde = bden[0][0]
                ip = ipn[0][0]
                r = 2.0 * (self.bed_weight * (1.0 - bde) + self.ip_weight * ip) + self.rrab_weight * rrab
                rewards.append(r)
            else:
                rewards.append(self.reward_of_invalid_mol)
        return rewards

    def _deduplicate_smiles(self, molecules):
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        smiles_p = {}
        for i, s in enumerate(smiles):
            if s in smiles_p:
                smiles_p[s][1].append(i)
            else:
                mol_with_H = Chem.AddHs(molecules[i])
                smiles_p[s] = mol_with_H, [i]
        return smiles, smiles_p

    def _find_bde_ip_reward_cache(self, molecules, bde_cache, ip_cache):
        smiles, smiles_p = self._deduplicate_smiles(molecules)

        bde_ps, bde_vs = self.predict_BDE_cache(smiles, smiles_p, useCache=self.use_bde_cache, bde_cache=bde_cache)
        # Filter out mols without valid BDE before IP prediction
        for s, v in zip(smiles, bde_vs):
            if (not v) and s in smiles_p:
                del smiles_p[s]
        ip_preds, ip_vs, ip_probs = self.predict_IP_cache(smiles, smiles_p, useCache=self.use_ip_cache, ip_cache=ip_cache)

        rrabs = self.calc_rrabs(molecules)
        rewards = self._compute_reward(bde_ps, bde_vs, ip_preds, ip_vs, ip_probs, rrabs)
        return {'reward': rewards, 'BDE': bde_ps, 'IP': ip_preds, 'RRAB': rrabs, 'IP_Probs': ip_probs}

    # --- Overlap version ---

    def _run_etkdg_for_ip(self, smiles_p, useCache, ip_cache):
        maxAttempts = self.etkdg_max_attempts_cache if useCache else self.etkdg_max_attempts_uncache
        ip_uncached_smiles = []
        ip_uncached_mols = []
        ip_cached_results = {}

        for s, (mol_with_H, ids) in smiles_p.items():
            if useCache:
                p, v = ip_cache.get(s)
                if v:
                    ip_cached_results[s] = p
                    continue
            ip_uncached_smiles.append(s)
            ip_uncached_mols.append(mol_with_H)

        etkdg_data = None
        if ip_uncached_mols:
            etkdg_data = self.rwmol2data_atts(ip_uncached_mols, maxAttempts)

        return ip_uncached_smiles, ip_uncached_mols, etkdg_data, ip_cached_results, maxAttempts

    def _assemble_ip_results(self, smiles, smiles_p, ip_uncached_smiles,
                             ip_cached_results, aimnet_preds, aimnet_vs, aimnet_probs,
                             ip_cache):
        ip_preds = [self.reward_of_invalid_mol for _ in smiles]
        ip_vs = [False for _ in smiles]
        ip_probs = [-1 for _ in smiles]

        for s, (pred, prob) in ip_cached_results.items():
            if s in smiles_p:
                for i in smiles_p[s][1]:
                    if random.random() <= prob:
                        ip_preds[i] = pred
                        ip_vs[i] = True
                        ip_probs[i] = prob

        if aimnet_preds is not None:
            for j, s in enumerate(ip_uncached_smiles):
                pred = aimnet_preds[j]
                v = aimnet_vs[j]
                prob = aimnet_probs[j]
                if v:
                    if self.use_ip_cache:
                        ip_cache.put(s, (pred, prob))
                    if s in smiles_p:
                        for i in smiles_p[s][1]:
                            ip_preds[i] = pred
                            ip_vs[i] = True
                            ip_probs[i] = prob

        return ip_preds, ip_vs, ip_probs

    def _find_bde_ip_reward_overlap(self, molecules, prefetch_fn=None, cleanup_fn=None,
                                     bde_cache=None, ip_cache=None):
        smiles, smiles_p_all = self._deduplicate_smiles(molecules)

        # PRE-PHASE: BDE prep + ETKDG cache check
        bde_ps = [self.reward_of_invalid_mol for _ in smiles]
        bde_vs = [False for _ in smiles]
        bde_uncached_smiles = []
        for s, (mol_with_H, ids) in smiles_p_all.items():
            if self.use_bde_cache:
                p, v = bde_cache.get(s)
                if v:
                    for i in ids:
                        bde_ps[i] = p
                        bde_vs[i] = True
                    continue
            bde_uncached_smiles.append(s)

        bde_batch = bde_graphs = bde_graph_indices = bde_partial_results = None
        if bde_uncached_smiles:
            bde_batch, bde_graphs, bde_graph_indices, bde_partial_results = \
                self.bde_model.prep_batch(bde_uncached_smiles)

        maxAttempts = self.etkdg_max_attempts_cache if self.use_ip_cache else self.etkdg_max_attempts_uncache
        ip_uncached_smiles = []
        ip_uncached_mols = []
        ip_cached_results = {}
        for s, (mol_with_H, ids) in smiles_p_all.items():
            if self.use_ip_cache:
                p, v = ip_cache.get(s)
                if v:
                    ip_cached_results[s] = p
                    continue
            ip_uncached_smiles.append(s)
            ip_uncached_mols.append(mol_with_H)

        # LEVEL 1: BDE_gpu || ETKDG_embed(C++) || cleanup
        bde_raw_pred = [None]
        etkdg_raw = [None]

        def run_bde_gpu():
            bde_raw_pred[0] = self.bde_model.forward_batch(bde_batch)

        def run_etkdg_embed():
            if ip_uncached_mols:
                import src.cenv as cenv
                nThreads = min(72, len(ip_uncached_mols) * maxAttempts)
                etkdg_raw[0] = cenv.embed_molecules_parallel(
                    ip_uncached_mols, maxAttempts, nThreads,
                    self.etkdg_max_iterations, self.etkdg_timeout)

        threads = []
        if bde_batch is not None:
            threads.append(threading.Thread(target=run_bde_gpu))
        if ip_uncached_mols:
            threads.append(threading.Thread(target=run_etkdg_embed))
        if cleanup_fn is not None:
            threads.append(threading.Thread(target=cleanup_fn))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # POST-PHASE 1: BDE post + ETKDG tensor assembly
        if bde_uncached_smiles:
            ucps, ucvs = self.bde_model.postprocess_oh_bde(
                bde_raw_pred[0], bde_graphs, bde_graph_indices,
                bde_partial_results, bde_uncached_smiles)
            for j, s in enumerate(bde_uncached_smiles):
                p, v = ucps[j], ucvs[j]
                if v:
                    if self.use_bde_cache:
                        bde_cache.put(s, p)
                    for i in smiles_p_all[s][1]:
                        bde_ps[i] = p
                        bde_vs[i] = True

        smiles_p_filtered = dict(smiles_p_all)
        for s, v in zip(smiles, bde_vs):
            if (not v) and s in smiles_p_filtered:
                del smiles_p_filtered[s]
        bde_valid_smiles = set(smiles_p_filtered.keys())

        etkdg_data = None
        if etkdg_raw[0] is not None:
            coords_list, success_list, probs_list, natoms_list = etkdg_raw[0]
            ds = [dict() for _ in ip_uncached_mols]
            vs = list(success_list)
            etkdg_probs = list(probs_list)
            charge = torch.tensor([1, 0, -1]).to(self.device)
            mult = torch.tensor([2, 1, 2]).to(self.device)
            for idx in range(len(ip_uncached_mols)):
                if vs[idx]:
                    coords = torch.tensor(coords_list[idx], dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(self.device)
                    numbers = [a.GetAtomicNum() for a in ip_uncached_mols[idx].GetAtoms()]
                    numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(self.device)
                    ds[idx] = dict(coord=coords, numbers=numbers, charge=charge, mult=mult)
                s = ip_uncached_smiles[idx]
                if s not in bde_valid_smiles:
                    vs[idx] = False
            etkdg_data = (ds, vs, etkdg_probs)

        # AIMNet prep
        aimnet_batches = []
        aimnet_preds = None
        aimnet_vs = None
        aimnet_probs = None
        if etkdg_data is not None:
            ds, vs, probs = etkdg_data
            aimnet_preds = [0.0] * len(ds)
            aimnet_vs = vs
            aimnet_probs = probs
            groups = {}
            for i, (data, valid) in enumerate(zip(ds, vs)):
                if valid:
                    N = data['numbers'].shape[1]
                    groups.setdefault(N, []).append((i, data))
            charge_single = torch.tensor([1, 0, -1], device=self.device)
            mult_single = torch.tensor([2, 1, 2], device=self.device)
            for N, items in groups.items():
                indices, datas = zip(*items)
                B = len(items)
                batch_coord = torch.stack([d['coord'] for d in datas], dim=0).reshape(B * 3, N, 3)
                batch_numbers = torch.stack([d['numbers'] for d in datas], dim=0).reshape(B * 3, N)
                batch_charge = charge_single.repeat(B)
                batch_mult = mult_single.repeat(B)
                batch_data = dict(coord=batch_coord, numbers=batch_numbers,
                                  charge=batch_charge, mult=batch_mult)
                aimnet_batches.append((list(indices), batch_data, B))

        # LEVEL 2: AIMNet_gpu || prefetch_fn(CVA, C++)
        aimnet_raw_results = []
        prefetch_result = [None]

        def run_aimnet_gpu():
            if not aimnet_batches:
                return
            ip_model = self.ip_model
            with torch.jit.optimized_execution(False), torch.no_grad():
                for indices, batch_data, B in aimnet_batches:
                    pred = ip_model.model(batch_data)
                    energy = pred['energy'].detach().cpu().numpy().reshape(B, 3)
                    aimnet_raw_results.append((indices, energy, B))

        def run_prefetch():
            if prefetch_fn is not None:
                try:
                    prefetch_result[0] = prefetch_fn()
                except Exception as e:
                    logging.warning(f"prefetch_fn failed: {e}")
                    prefetch_result[0] = None

        t_aimnet = threading.Thread(target=run_aimnet_gpu)
        t_prefetch = threading.Thread(target=run_prefetch)
        t_aimnet.start()
        t_prefetch.start()
        t_aimnet.join()
        t_prefetch.join()

        # POST-PHASE 2: AIMNet post + assemble rewards
        if aimnet_preds is not None:
            for indices, energy, B in aimnet_raw_results:
                ip_kcal = (energy[:, 0] - energy[:, 1]) * 23.0609
                for batch_i, mol_i in enumerate(indices):
                    aimnet_preds[mol_i] = float(ip_kcal[batch_i])

        ip_preds, ip_vs, ip_probs = self._assemble_ip_results(
            smiles, smiles_p_filtered, ip_uncached_smiles,
            ip_cached_results, aimnet_preds, aimnet_vs, aimnet_probs,
            ip_cache)

        rrabs = self.calc_rrabs(molecules)
        rewards = self._compute_reward(bde_ps, bde_vs, ip_preds, ip_vs, ip_probs, rrabs)

        reward_dict = {'reward': rewards, 'BDE': bde_ps, 'IP': ip_preds,
                       'RRAB': rrabs, 'IP_Probs': ip_probs}
        return reward_dict, prefetch_result[0]
