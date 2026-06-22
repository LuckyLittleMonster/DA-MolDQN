"""RL environment: ``MultiMolecules``.

A thin environment over the ``Molecule`` base. It composes a ``BDEPredictor`` and
an ``IPPredictor`` (for the BDE_IP reward) and only handles reward *combination*:
scaler transform, weighted combine, and RRAB math. BDE/IP prediction itself is
delegated to the composed predictors. QED and pLogP rewards live here too.
"""
from rdkit import Chem

from src import config_defaults as hyp
from src.environment import Molecule
from src.reward.qed import qed_value
from src.reward.sa import sa_score
from src.reward.plogp import plogp_value
from src.cache import LRUCache, CachedPredictor
from src.reward.bde_predictor.predictor import BDEPredictor
from src.reward.ip_predictor.predictor import IPPredictor


def count_OH(mol):
    OH_count = 0;
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetNumImplicitHs() > 0: # 8 for 'O'
            OH_count += 1
    return OH_count


class MultiMolecules(Molecule):
    """docstring for DistributedMolecules"""
    def __init__(self, args, device, **kwargs):
        super(MultiMolecules, self).__init__(
            args = args,
            **kwargs)

        # BDE is memoized by canonical SMILES (deterministic). IP is NOT cached —
        # its value depends on a random ETKDG conformer. The trainer reads
        # environment.bde_cache.hit_rate() for logging, so keep the object here.
        self.bde_cache = LRUCache(hyp.lru_cache_capacity * len(self.init_mols))
        self.discount_factor = args.discount_factor
        self.device = device

        self.bde_ip_reward = False
        self.qed_reward = False
        self.plogp_reward = False

        # parser.add_argument('--maintain_OH', type=str, default=None, help=
        #     "default: None or 'None': no limitation\n"
        #     "same: The number of OH bonds are always same to the initial molecules.\n"
        #     "exist: All molecules must have one or more OH bonds.\n"
        #     "n: all mols should have the n of OH bonds\n")

        # maintail_OH:
        #     -2: no limitation
        #     -1: at least 1 OH bond
        #     0 ~ N: has the number of OH bonds, it is the same as the initial mol

        if args.maintain_OH is None:
            self.maintain_OH_flags = [-2 for _ in self.init_mols]
        elif args.maintain_OH == 'same':
            self.maintain_OH_flags = [count_OH(m) for m in self.init_mols]
        elif args.maintain_OH == 'exist':
            self.maintain_OH_flags = [-1 for _ in self.init_mols]
        else:
            self.maintain_OH_flags = [int(args.maintain_OH) for _ in self.init_mols]

        # if args.maintain_OH is None:
        #     self.maintain_OH_flags = [count_OH(m) for m in self.init_mols]
        # else :
        #     self.maintain_OH_flags = [args.maintain_OH for _ in self.init_mols]

        if args.reward.lower() == "BDE_IP".lower():
            self.bde_ip_reward = True
            self.bde_factor = hyp.bde_factor
            self.ip_factor = hyp.ip_factor

            self.bed_weight = 0.8
            self.ip_weight = 0.2
            self.rrab_weight = 0.5

            self.use_bde_cache = 'bde' in args.cache
            self.etkdg_max_attempts_cache = args.etkdg_max_attempts_cache
            self.etkdg_max_attempts_uncache = args.etkdg_max_attempts_uncache

            # Intra-rank ETKDG threading: RDKit's EmbedMolecule releases the GIL,
            # so a thread pool parallelizes 3D embedding within one rank
            # (~1.75x at 2 threads). Lets 36 ranks x 2 threads keep 72-core etkdg
            # throughput while halving GPU contexts (fits memory + MPS works).
            self.etkdg_threads = int(getattr(args, 'etkdg_threads', 1))

            if len(args.reward_weight) == 0:
                # use default weights
                pass
            elif len(args.reward_weight) == 1:
                # assume that the one value is bde weight, which is the same as main_multi.py
                self.bed_weight = args.reward_weight[0]
                self.ip_weight = 1.0 - self.bed_weight

            elif len(args.reward_weight) == 2:
                self.bed_weight = args.reward_weight[0]
                self.ip_weight = args.reward_weight[1]
            else :
                self.bed_weight = args.reward_weight[0]
                self.ip_weight = args.reward_weight[1]
                self.rrab_weight = args.reward_weight[2]

            # Pure predictors, wrapped by a CachedPredictor for generic
            # dedup + index-mapping. BDE gets a swappable cache (LRU); IP gets
            # cache=None (never cached) and call_on_empty=False.
            self.bde_predictor = BDEPredictor(device=self.device)
            self.bde_scaler = self.bde_predictor.bde_scaler
            self.bde_model = self.bde_predictor.bde_model
            self.bde = CachedPredictor(
                self.bde_predictor.predict_BDE, cache=self.bde_cache,
                invalid_value=hyp.reward_of_invalid_mol)

            self.ip_predictor = IPPredictor(
                device=self.device,
                etkdg_threads=self.etkdg_threads,
                etkdg_max_attempts_cache=self.etkdg_max_attempts_cache,
                etkdg_max_attempts_uncache=self.etkdg_max_attempts_uncache)
            self.ip_scaler = self.ip_predictor.ip_scaler
            self.ip_model = self.ip_predictor.ip_model
            _ip_attempts = self.etkdg_max_attempts_uncache
            self.ip = CachedPredictor(
                lambda keys, mols: self.ip_predictor.predict_IP(mols, _ip_attempts),
                cache=None, call_on_empty=False,
                invalid_value=hyp.reward_of_invalid_mol)

            self.init_mols_n = [m.GetNumAtoms() + m.GetNumBonds() for m in self.init_mols]

        elif args.reward.lower() == "qed":
            self.qed_reward = True
            self.qed_weight = 0.8
            self.sa_weight = 0.2
            if len(args.reward_weight) == 0:
                # use default weights
                pass
            elif len(args.reward_weight) == 1:
                # assume that the one value is bde weight, which is the same as main_multi.py
                self.qed_weight = args.reward_weight[0]
                self.sa_weight = 1.0 - self.qed_weight

            elif len(args.reward_weight) == 2:
                self.qed_weight = args.reward_weight[0]
                self.sa_weight = args.reward_weight[1]

        elif args.reward.lower() == "plogp":
            self.plogp_reward = True
            # self.plogp_weight = 1.0

        self.init_rewards = self.find_reward(self.init_mols)

    def calc_rrabs(self, molecules):
        rrabs = []
        for molecule, init_mol_n in zip(molecules, self.init_mols_n):
            n = molecule.GetNumAtoms() + molecule.GetNumBonds()
            rrab = float(init_mol_n - n) / float(init_mol_n)
            rrabs.append(rrab)
        return rrabs

    def find_bde_ip_reward_cache(self, molecules):

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

        bde_ps, bde_vs = self.bde.predict(smiles, smiles_p, use_cache=self.use_bde_cache)
        # ignore mols without valid BDE while predicting IP
        for s, v in zip(smiles, bde_vs):
            if (not v) and s in smiles_p:
                del smiles_p[s]
        # IP is never cached (random conformer); use_cache=False -> dedup + map only.
        ip_preds, ip_vs = self.ip.predict(smiles, smiles_p, use_cache=False)

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
                rewards.append(hyp.reward_of_invalid_mol)

        return {'reward':rewards, 'BDE':bde_ps, 'IP':ip_preds, 'RRAB': rrabs}

    def find_qed_reward(self, molecules):
        rs = []
        qeds = []
        sas = []
        for molecule in molecules:
            qed = qed_value(molecule)
            qeds.append(qed)
            SA_score = sa_score(molecule)
            sas.append(SA_score)
            reward = (qed * self.qed_weight - self.sa_weight * SA_score) * self.discount_factor ** (self.max_steps-self.current_step)
            rs.append(reward)
        return {'reward': rs, 'QED':qeds, 'SA_score':sas}

    def find_plogp_reward(self, molecules):
        rs = []
        sims = []
        for mol in molecules:
            score = plogp_value(mol)
            sims.append(-1)
            reward = score * self.discount_factor ** (self.max_steps-self.current_step)
            rs.append(reward)
        return {'reward': rs, 'plogp':rs, 'sim': sims}


    def find_reward(self, molecules = None):
        if molecules is None:
            molecules = self.states
        if self.bde_ip_reward:
            return self.find_bde_ip_reward_cache(molecules)
        elif self.qed_reward:
            return self.find_qed_reward(molecules)
        elif self.plogp_reward:
            return self.find_plogp_reward(molecules)
