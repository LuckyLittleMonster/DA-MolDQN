"""Molecular optimization environments.

BaseEnvironment   — abstract base with action generation, stepping, observation
                    computation. Subclasses implement calc_reward().
QEDEnvironment    — QED + SA_score reward.
DockingEnvironment — stub for future docking-based reward.
"""

from abc import abstractmethod
import os
import sys
import numpy as np

import hyp
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class BaseEnvironment:
    """Base environment for reaction-based molecular optimization.

    Handles action generation (reaction predictors + optional atom-level ops),
    stepping, fingerprint computation, observation computation, and reward
    shaping.  Subclasses only need to implement ``calc_reward(mol)``.
    """

    def __init__(self, init_mols, discount_factor=0.9, max_steps=20,
                 use_hypergraph=True, hypergraph_top_k=20, device='cuda',
                 reaction_only=False, reactant_method='hypergraph',
                 product_num_beams=1,
                 expand_fragments=False, mw_penalty=0.0,
                 mw_threshold=150.0, co_reactant_oversample=1,
                 filter_products=True, filter_min_tanimoto=0.2,
                 filter_max_mw_delta=200.0,
                 reward_tanimoto_bonus=0.0, reward_mw_penalty=0.0,
                 model_top_k=20):

        self.init_mols = [Chem.RWMol(Chem.MolFromSmiles(s)) if isinstance(s, str)
                          else Chem.RWMol(s) for s in init_mols]
        self.discount_factor = discount_factor
        self.max_steps = max_steps
        self.device = device
        self.reaction_only = reaction_only

        # Reward shaping
        self.reward_tanimoto_bonus = reward_tanimoto_bonus
        self.reward_mw_penalty = reward_mw_penalty

        # Atom-level action config (used when reaction_only=False)
        self.atom_types = set(['C', 'N', 'O'])
        self.allow_removal = True
        self.allow_no_modification = True

        # Fingerprint generator
        from rdkit.Chem import rdFingerprintGenerator
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            fpSize=hyp.fingerprint_length, radius=hyp.fingerprint_radius)

        # Reaction predictor
        self.use_hypergraph = use_hypergraph
        self.hypergraph_predictor = None  # legacy
        self.reaction_predictor = None    # new pipeline
        self.reactant_method = reactant_method
        self.hypergraph_top_k = hypergraph_top_k

        if use_hypergraph:
            self._init_reaction_predictor(
                reactant_method=reactant_method,
                hypergraph_top_k=hypergraph_top_k,
                device=device,
                product_num_beams=product_num_beams,
                expand_fragments=expand_fragments,
                mw_penalty=mw_penalty,
                mw_threshold=mw_threshold,
                co_reactant_oversample=co_reactant_oversample,
                filter_products=filter_products,
                filter_min_tanimoto=filter_min_tanimoto,
                filter_max_mw_delta=filter_max_mw_delta,
                model_top_k=model_top_k,
            )

        self.reset()

    # ------------------------------------------------------------------
    # Reaction predictor factory
    # ------------------------------------------------------------------

    def _init_reaction_predictor(self, *, reactant_method, hypergraph_top_k,
                                 device, product_num_beams,
                                 expand_fragments, mw_penalty,
                                 mw_threshold, co_reactant_oversample,
                                 filter_products, filter_min_tanimoto,
                                 filter_max_mw_delta, model_top_k):
        if reactant_method in ('hypergraph', 'fingerprint'):
            try:
                from model_reactions.reaction_predictor import ReactionPredictor
                from model_reactions.config import ReactionPredictorConfig
                config = ReactionPredictorConfig(
                    reactant_method=reactant_method,
                    reactant_top_k=hypergraph_top_k,
                    expand_fragments=expand_fragments,
                    mw_penalty_alpha=mw_penalty,
                    mw_penalty_threshold=mw_threshold,
                    co_reactant_oversample=co_reactant_oversample,
                    filter_products=filter_products,
                    filter_min_tanimoto=filter_min_tanimoto,
                    filter_max_mw_delta=filter_max_mw_delta,
                )
                config.product_config.num_beams = product_num_beams
                config.product_config.num_return_sequences = 1
                self.reaction_predictor = ReactionPredictor(config=config, device=device)
                self.reaction_predictor.load()
                print(f"ReactionPredictor initialized (method={reactant_method})")
            except Exception as e:
                print(f"Failed to initialize ReactionPredictor: {e}")
                import traceback; traceback.print_exc()
        elif reactant_method == 'aio':
            try:
                from hypergraph.action_generator import AIOActionGenerator
                self.reaction_predictor = AIOActionGenerator(
                    device=device,
                    top_k=hypergraph_top_k,
                    filter_products=filter_products,
                    filter_min_tanimoto=filter_min_tanimoto,
                    filter_max_mw_delta=filter_max_mw_delta,
                )
                self.reaction_predictor.load()
                print(f"AIOActionGenerator initialized (directed hypergraph)")
            except Exception as e:
                print(f"Failed to initialize AIOActionGenerator: {e}")
                import traceback; traceback.print_exc()
        elif reactant_method == 'hybrid':
            try:
                from model_reactions.hybrid_predictor import HybridReactionPredictor
                self.reaction_predictor = HybridReactionPredictor(
                    device=device,
                    top_k=hypergraph_top_k,
                    filter_products=filter_products,
                    filter_min_tanimoto=filter_min_tanimoto,
                    filter_max_mw_delta=filter_max_mw_delta,
                )
                self.reaction_predictor.load()
                print(f"HybridReactionPredictor initialized (AIO + V3 re-rank)")
            except Exception as e:
                print(f"Failed to initialize HybridReactionPredictor: {e}")
                import traceback; traceback.print_exc()
        elif reactant_method == 'template':
            try:
                from template import TemplateReactionPredictor
                n_workers = 8
                self.reaction_predictor = TemplateReactionPredictor(
                    top_k=hypergraph_top_k,
                    num_workers=n_workers,
                )
                self.reaction_predictor.load()
                print(f"TemplateReactionPredictor initialized "
                      f"(71 templates + 10K blocks, {n_workers} workers)")
            except Exception as e:
                print(f"Failed to initialize TemplateReactionPredictor: {e}")
                import traceback; traceback.print_exc()
        elif reactant_method == 'template_2model':
            try:
                from model_reactions.hybrid_template_predictor import HybridTemplate2ModelPredictor
                self.reaction_predictor = HybridTemplate2ModelPredictor(
                    device=device,
                    top_k=hypergraph_top_k,
                    model_top_k=model_top_k,
                    reactant_method='hypergraph',
                    filter_products=filter_products,
                    filter_min_tanimoto=filter_min_tanimoto,
                    filter_max_mw_delta=filter_max_mw_delta,
                    product_num_beams=product_num_beams,
                )
                self.reaction_predictor.load()
                print(f"HybridTemplate2ModelPredictor initialized "
                      f"(model_top_k={model_top_k}, total_top_k={hypergraph_top_k})")
            except Exception as e:
                print(f"Failed to initialize HybridTemplate2ModelPredictor: {e}")
                import traceback; traceback.print_exc()
        elif reactant_method == 'legacy':
            try:
                from hypergraph.hypergraph_neighbor_predictor import HypergraphNeighborPredictor
                self.hypergraph_predictor = HypergraphNeighborPredictor(
                    data_dir="Data/uspto",
                    top_k=hypergraph_top_k,
                    max_index_mols=10000,
                    device=device
                )
                print("HypergraphNeighborPredictor initialized (legacy)")
            except Exception as e:
                print(f"Failed to initialize HypergraphNeighborPredictor: {e}")

    # ------------------------------------------------------------------
    # Reset / fingerprint
    # ------------------------------------------------------------------

    def reset(self):
        """Reset environment to initial state."""
        self.states = [Chem.RWMol(mol) for mol in self.init_mols]
        self.step_count = 0
        return self.states

    def get_fingerprint(self, mol):
        """Get Morgan fingerprint as numpy array."""
        try:
            Chem.SanitizeMol(mol)
            fp = self.fp_gen.GetFingerprint(mol)
            arr = np.zeros(hyp.fingerprint_length, dtype=np.float32)
            for idx in fp.GetOnBits():
                arr[idx] = 1.0
            return arr
        except Exception:
            return np.zeros(hyp.fingerprint_length, dtype=np.float32)

    def get_fingerprint_from_smiles(self, smi):
        """Get Morgan fingerprint from SMILES string (thread-safe)."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(hyp.fingerprint_length, dtype=np.float32)
        return self.get_fingerprint(mol)

    # ------------------------------------------------------------------
    # Reward (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def calc_reward(self, mol):
        """Calculate reward for a molecule.

        Returns:
            (reward: float, info: dict) where info contains property values
            keyed by property name, e.g. {'QED': 0.8, 'SA_score': 3.2}.
        """
        ...

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def _add_reaction_actions(self, mol, smiles, valid_actions, action_info,
                              max_actions=5):
        """Add reaction-based actions using reaction_predictor or legacy."""
        predictor = self.reaction_predictor or self.hypergraph_predictor
        if predictor is None:
            return

        try:
            co_reacts, products, scores = predictor.get_valid_actions(
                mol, top_k=max_actions)
            for idx, prod_smiles in enumerate(products[:max_actions]):
                try:
                    prod_mol = Chem.MolFromSmiles(prod_smiles)
                    if prod_mol is None:
                        continue
                    Chem.SanitizeMol(prod_mol)
                    _ = self.fp_gen.GetFingerprint(prod_mol)
                    valid_actions.append(Chem.RWMol(prod_mol))
                    action_info.append({
                        'source': 'reaction',
                        'action_type': 'predicted_reaction',
                        'reactant': smiles,
                        'product': prod_smiles,
                        'co_reactant': co_reacts[idx] if idx < len(co_reacts) else None,
                        'reaction_score': float(scores[idx]) if idx < len(scores) else None,
                    })
                except Exception:
                    continue
        except Exception:
            pass

    def get_valid_actions(self, mol, return_info=False):
        """Get valid actions for a molecule."""
        valid_actions = []
        action_info = []
        smiles = Chem.MolToSmiles(mol)

        if self.reaction_only:
            # Always include current molecule as action (early stopping)
            valid_actions.append(Chem.RWMol(mol))
            action_info.append({
                'source': 'no_modification',
                'action_type': 'keep_same',
                'reactant': smiles,
                'product': smiles,
                'co_reactant': None,
                'reaction_score': None,
            })
            self._add_reaction_actions(mol, smiles, valid_actions, action_info,
                                       max_actions=10)
            if return_info:
                return valid_actions, action_info
            return valid_actions

        # Mixed mode: reactions + atom/bond modifications
        if self.allow_no_modification:
            valid_actions.append(Chem.RWMol(mol))
            action_info.append({
                'source': 'no_modification',
                'action_type': 'keep_same',
                'reactant': smiles,
                'product': smiles,
                'co_reactant': None,
                'reaction_score': None,
            })

        # Add atoms
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetImplicitValence() == 0:
                continue
            for atom_type in self.atom_types:
                try:
                    new_mol = Chem.RWMol(mol)
                    new_atom_idx = new_mol.AddAtom(Chem.Atom(atom_type))
                    new_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                    Chem.SanitizeMol(new_mol)
                    new_smiles = Chem.MolToSmiles(new_mol)
                    if new_smiles != smiles:
                        valid_actions.append(new_mol)
                        atom_symbol = (atom_type if isinstance(atom_type, str)
                                       else Chem.GetPeriodicTable().GetElementSymbol(atom_type))
                        action_info.append({
                            'source': 'add_atom',
                            'action_type': f'add_{atom_symbol}_to_atom_{atom_idx}',
                            'reactant': smiles,
                            'product': new_smiles,
                            'co_reactant': None,
                            'reaction_score': None,
                        })
                except Exception:
                    continue

        # Add bonds between existing atoms
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                if mol.GetBondBetweenAtoms(i, j) is None:
                    try:
                        new_mol = Chem.RWMol(mol)
                        new_mol.AddBond(i, j, Chem.BondType.SINGLE)
                        Chem.SanitizeMol(new_mol)
                        new_smiles = Chem.MolToSmiles(new_mol)
                        if new_smiles != smiles:
                            valid_actions.append(new_mol)
                            action_info.append({
                                'source': 'add_bond',
                                'action_type': f'add_bond_{i}_{j}',
                                'reactant': smiles,
                                'product': new_smiles,
                                'co_reactant': None,
                                'reaction_score': None,
                            })
                    except Exception:
                        continue

        # Remove bonds
        if self.allow_removal:
            for bond in mol.GetBonds():
                if bond.GetIsAromatic():
                    continue
                try:
                    new_mol = Chem.RWMol(mol)
                    bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    new_mol.RemoveBond(bi, bj)
                    frags = Chem.GetMolFrags(new_mol)
                    if len(frags) == 1:
                        Chem.SanitizeMol(new_mol)
                        new_smiles = Chem.MolToSmiles(new_mol)
                        if new_smiles != smiles:
                            valid_actions.append(new_mol)
                            action_info.append({
                                'source': 'remove_bond',
                                'action_type': f'remove_bond_{bi}_{bj}',
                                'reactant': smiles,
                                'product': new_smiles,
                                'co_reactant': None,
                                'reaction_score': None,
                            })
                except Exception:
                    continue

        # Reaction-based actions
        self._add_reaction_actions(mol, smiles, valid_actions, action_info,
                                   max_actions=5)

        if not valid_actions:
            valid_actions = [Chem.RWMol(mol)]
            action_info.append({
                'source': 'fallback',
                'action_type': 'keep_same',
                'reactant': smiles,
                'product': smiles,
                'co_reactant': None,
                'reaction_score': None,
            })

        if return_info:
            return valid_actions, action_info
        return valid_actions

    def get_valid_actions_batch(self, mols, return_info=False):
        """Batch valid actions for multiple molecules."""
        if not self.reaction_only or self.reaction_predictor is None:
            all_valid = []
            all_info = []
            for mol in mols:
                if return_info:
                    va, ai = self.get_valid_actions(mol, return_info=True)
                    all_valid.append(va)
                    all_info.append(ai)
                else:
                    all_valid.append(self.get_valid_actions(mol))
            if return_info:
                return all_valid, all_info
            return all_valid

        smiles_list = [Chem.MolToSmiles(m) for m in mols]

        try:
            batch_results = self.reaction_predictor.get_valid_actions_batch(
                smiles_list, top_k=self.hypergraph_top_k)
        except Exception:
            all_valid = []
            all_info = []
            for mol in mols:
                if return_info:
                    va, ai = self.get_valid_actions(mol, return_info=True)
                    all_valid.append(va)
                    all_info.append(ai)
                else:
                    all_valid.append(self.get_valid_actions(mol))
            if return_info:
                return all_valid, all_info
            return all_valid

        all_valid_actions = []
        all_action_info = []

        for mol_idx, (co_reacts, products, scores) in enumerate(batch_results):
            valid_actions = []
            action_info = []

            valid_actions.append(Chem.RWMol(mols[mol_idx]))
            action_info.append({
                'source': 'no_modification',
                'action_type': 'keep_same',
                'reactant': smiles_list[mol_idx],
                'product': smiles_list[mol_idx],
                'co_reactant': None,
                'reaction_score': None,
            })

            for idx, prod_smiles in enumerate(products):
                try:
                    prod_mol = Chem.MolFromSmiles(prod_smiles)
                    if prod_mol is None:
                        continue
                    Chem.SanitizeMol(prod_mol)
                    _ = self.fp_gen.GetFingerprint(prod_mol)
                    valid_actions.append(Chem.RWMol(prod_mol))
                    action_info.append({
                        'source': 'reaction',
                        'action_type': 'predicted_reaction',
                        'reactant': smiles_list[mol_idx],
                        'product': prod_smiles,
                        'co_reactant': co_reacts[idx] if idx < len(co_reacts) else None,
                        'reaction_score': float(scores[idx]) if idx < len(scores) else None,
                    })
                except Exception:
                    continue

            all_valid_actions.append(valid_actions)
            all_action_info.append(action_info)

        if return_info:
            return all_valid_actions, all_action_info
        return all_valid_actions

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions):
        """Take a step: apply actions, compute rewards with shaping."""
        prev_states = self.states
        self.states = actions
        self.step_count += 1

        rewards = []
        property_info = {}  # key → list of per-mol values

        for idx, mol in enumerate(self.states):
            r, info = self.calc_reward(mol)

            # Accumulate property values
            for key, val in info.items():
                property_info.setdefault(key, []).append(val)

            # Reward shaping: Tanimoto similarity bonus
            if self.reward_tanimoto_bonus > 0 and prev_states is not None:
                prev_mol = prev_states[idx]
                try:
                    fp_prev = self.fp_gen.GetFingerprint(prev_mol)
                    fp_cur = self.fp_gen.GetFingerprint(mol)
                    from rdkit import DataStructs
                    tani = DataStructs.TanimotoSimilarity(fp_prev, fp_cur)
                    r += tani * self.reward_tanimoto_bonus
                except Exception:
                    pass

            # Reward shaping: MW change penalty
            if self.reward_mw_penalty > 0 and prev_states is not None:
                prev_mol = prev_states[idx]
                try:
                    from rdkit.Chem import Descriptors
                    mw_prev = Descriptors.ExactMolWt(prev_mol)
                    mw_cur = Descriptors.ExactMolWt(mol)
                    mw_delta = abs(mw_cur - mw_prev)
                    r -= max(0.0, mw_delta - 100.0) * self.reward_mw_penalty
                except Exception:
                    pass

            # Discount
            r = r * (self.discount_factor ** (self.max_steps - self.step_count))
            rewards.append(r)

        done = self.step_count >= self.max_steps

        result = {'reward': rewards, 'done': done}
        result.update(property_info)
        return result

    # ------------------------------------------------------------------
    # Observation computation
    # ------------------------------------------------------------------

    def compute_observations(self, valid_actions, step, gnn_dqn=None):
        """Compute observations for a list of action molecules."""
        step_frac = step / self.max_steps
        if gnn_dqn is not None:
            smiles_list = [Chem.MolToSmiles(m) for m in valid_actions]
            return gnn_dqn.encode_molecules(smiles_list, step_fraction=step_frac)
        else:
            observations = []
            for action_mol in valid_actions:
                fp = self.get_fingerprint(action_mol)
                obs = np.append(fp, step_frac)
                observations.append(obs)
            return np.array(observations)

    def compute_observations_batch(self, all_valid_actions, step, gnn_dqn=None):
        """Compute observations for all molecules' valid actions in one batch.

        Flattens, deduplicates SMILES for fingerprint computation, splits back.
        """
        step_frac = step / self.max_steps

        flat_actions = []
        split_sizes = []
        for actions in all_valid_actions:
            split_sizes.append(len(actions))
            flat_actions.extend(actions)

        if gnn_dqn is not None:
            smiles_list = [Chem.MolToSmiles(m) for m in flat_actions]
            flat_obs = gnn_dqn.encode_molecules(smiles_list, step_fraction=step_frac)
        else:
            n_total = len(flat_actions)
            if n_total >= 32:
                smiles_list = [Chem.MolToSmiles(m) for m in flat_actions]
                unique_smiles = list(dict.fromkeys(smiles_list))
                fp_cache = {}
                for smi in unique_smiles:
                    fp_cache[smi] = self.get_fingerprint_from_smiles(smi)

                flat_obs = np.empty((n_total, hyp.fingerprint_length + 1),
                                    dtype=np.float32)
                for i, smi in enumerate(smiles_list):
                    flat_obs[i, :hyp.fingerprint_length] = fp_cache[smi]
                    flat_obs[i, hyp.fingerprint_length] = step_frac
            else:
                flat_obs = []
                for action_mol in flat_actions:
                    fp = self.get_fingerprint(action_mol)
                    obs = np.append(fp, step_frac)
                    flat_obs.append(obs)
                flat_obs = np.array(flat_obs)

        result = []
        offset = 0
        for size in split_sizes:
            result.append(flat_obs[offset:offset + size])
            offset += size
        return result


# ======================================================================
# QEDEnvironment
# ======================================================================

class QEDEnvironment(BaseEnvironment):
    """QED + SA_score reward environment."""

    def __init__(self, init_mols, qed_weight=0.8, sa_weight=0.2, **kwargs):
        self.qed_weight = qed_weight
        self.sa_weight = sa_weight

        # SA scorer
        from rdkit import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        self.sascorer = sascorer

        super().__init__(init_mols, **kwargs)

    def calc_reward(self, mol):
        """Calculate QED + SA reward.

        Returns:
            (reward, {'QED': qed, 'SA_score': sa})
        """
        try:
            from rdkit.Chem import QED as QEDModule
            Chem.SanitizeMol(mol)
            qed = QEDModule.qed(mol)
            sa = self.sascorer.calculateScore(mol)
            sa_normalized = (10 - sa) / 9
            reward = self.qed_weight * qed + self.sa_weight * sa_normalized
            return reward, {'QED': qed, 'SA_score': sa}
        except Exception:
            return -0.5, {'QED': 0.0, 'SA_score': 10.0}


# ======================================================================
# DockingEnvironment (stub)
# ======================================================================

class DockingEnvironment(BaseEnvironment):
    """Stub environment for docking-based reward (future extension)."""

    def __init__(self, init_mols, target_pdb=None, **kwargs):
        self.target_pdb = target_pdb
        super().__init__(init_mols, **kwargs)

    def calc_reward(self, mol):
        raise NotImplementedError("Override with your docking scorer.")
