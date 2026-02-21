"""Hierarchical DQN predictor for template-based reaction selection.

Replaces random sampling in TemplateReactionPredictor with two learned
Q-networks that select (template, building_block) pairs:

  Q1 (TemplatePolicyNetwork): mol_state -> template scores
  Q2 (BlockScoringNetwork):   (mol_state, template) -> block scores

Action generation flow:
  1. Compute molecule fingerprint
  2. Check which templates match (SubstructMatch) -> template_mask
  3. Q1 scores templates -> select top-K1 (e.g., 10)
  4. For uni-molecular templates: run reaction directly
  5. For bi-molecular templates:
     a. Q2 scores all compatible blocks -> select top-K2 per template
     b. Run reactions for (template, block) pairs
  6. Collect products, rank by Q-score, return top-K

Training:
  - Store transitions: (state_fp, template_id, block_id, reward, ...)
  - DQN update with experience replay and target networks
  - Q1 trained on template selection, Q2 on block selection
  - Both share the same reward signal from the RL environment

API compatible with TemplateReactionPredictor (drop-in replacement).
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .template_predictor import TemplateReactionPredictor
from .policy_network import TemplatePolicyNetwork, BlockScoringNetwork


class DQNTemplatePredictor:
    """Hierarchical DQN predictor for template + building block selection.

    Drop-in replacement for TemplateReactionPredictor with learned scoring.
    Uses Q1 for template selection and Q2 for building block selection.

    Args:
        device: 'cuda' or 'cpu'.
        top_k: Total number of actions to return.
        top_k_templates: Number of templates to consider per molecule.
        top_k_blocks: Number of blocks to consider per selected template.
        fp_dim: Morgan fingerprint dimension.
        fp_radius: Morgan fingerprint radius.
        hidden_dim: MLP hidden dimension for Q-networks.
        template_emb_dim: Template embedding dimension.
        block_emb_dim: Block embedding dimension for bilinear scoring.
        lr: Learning rate for Adam optimizer.
        gamma: Discount factor for DQN.
        polyak: Soft update coefficient for target networks.
        replay_size: Maximum replay buffer capacity.
        num_workers: Thread workers for parallel template matching.
        seed: Random seed.
        template_path: Path to reaction templates file.
        building_block_path: Path to building blocks SMILES file.
    """

    def __init__(
        self,
        device='cuda',
        top_k=20,
        top_k_templates=10,
        top_k_blocks=5,
        fp_dim=4096,
        fp_radius=3,
        hidden_dim=256,
        template_emb_dim=128,
        block_emb_dim=64,
        lr=1e-4,
        gamma=0.9,
        polyak=0.995,
        replay_size=50000,
        num_workers=8,
        seed=42,
        template_path=None,
        building_block_path=None,
        block_emb_refresh_freq=200,
        **kwargs,
    ):
        self.device = device
        self.top_k = top_k
        self.top_k_templates = top_k_templates
        self.top_k_blocks = top_k_blocks
        self.fp_dim = fp_dim
        self.fp_radius = fp_radius
        self.hidden_dim = hidden_dim
        self.template_emb_dim = template_emb_dim
        self.block_emb_dim = block_emb_dim
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.num_workers = num_workers
        self.seed = seed
        self.template_path = template_path
        self.building_block_path = building_block_path
        self.block_emb_refresh_freq = block_emb_refresh_freq

        self.rng = random.Random(seed)

        # Will be populated by load()
        self.tp = None  # TemplateReactionPredictor (for data)
        self.n_templates = 0
        self.n_blocks = 0
        self.template_q = None
        self.template_q_target = None
        self.block_q = None
        self.block_q_target = None
        self.optimizer = None

        # Lookup tables
        self.uni_by_template = {}
        self.bi_by_template = {}
        self.is_uni = None  # (n_templates,) bool
        self.is_bi = None   # (n_templates,) bool
        self.template_compat = None  # (n_templates, n_blocks) bool GPU
        self.block_fps = None  # (n_blocks, fp_dim) GPU tensor

        # Replay buffer
        self.replay_buffer = []
        self.replay_pos = 0

        # Cached state from last get_valid_actions_batch (for observe_step)
        self._last_fps_np = None
        self._last_template_masks = None
        self._last_batch_metadata = None

        self._loaded = False
        self._executor = None
        self._fp_gen = None
        self._train_steps = 0

    def load(self):
        """Initialize templates, blocks, networks, and precompute embeddings."""
        if self._loaded:
            return

        t0 = time.perf_counter()

        # 1. Create and load TemplateReactionPredictor for data
        self.tp = TemplateReactionPredictor(
            template_path=self.template_path,
            building_block_path=self.building_block_path,
            top_k=self.top_k,
            num_workers=0,  # We handle parallelism ourselves
        )
        self.tp.load()

        # 2. Extract template info
        all_indices = set()
        for rxn in self.tp.uni_reactions:
            all_indices.add(rxn.index)
        for rxn in self.tp.bi_reactions:
            all_indices.add(rxn.index)
        self.n_templates = max(all_indices) + 1 if all_indices else 0
        self.n_blocks = len(self.tp.bb_library)

        # 3. Build lookup tables: template_idx -> reaction objects
        self.uni_by_template = {}
        for rxn in self.tp.uni_reactions:
            self.uni_by_template[rxn.index] = rxn

        self.bi_by_template = {}
        for rxn in self.tp.bi_reactions:
            self.bi_by_template.setdefault(rxn.index, []).append(rxn)

        self.is_uni = np.zeros(self.n_templates, dtype=bool)
        for rxn in self.tp.uni_reactions:
            self.is_uni[rxn.index] = True

        self.is_bi = np.zeros(self.n_templates, dtype=bool)
        for rxn in self.tp.bi_reactions:
            self.is_bi[rxn.index] = True

        # bi_rxn object -> list index (for per-orientation compat lookup)
        self._bi_rxn_to_idx = {
            id(rxn): idx for idx, rxn in enumerate(self.tp.bi_reactions)}

        t1 = time.perf_counter()

        # 4. Dense compatibility masks on GPU
        # Per-bi_rxn: (n_bi_reactions, n_blocks) - orientation-specific
        n_bi = len(self.tp.bi_reactions)
        self.bi_compat_dense = torch.zeros(
            n_bi, self.n_blocks, dtype=torch.bool, device=self.device)
        for bi_idx, bi_rxn in enumerate(self.tp.bi_reactions):
            compat = self.tp.bi_compat.get(bi_idx)
            if compat is not None and len(compat) > 0:
                self.bi_compat_dense[bi_idx,
                                     torch.from_numpy(compat).long()] = True

        # Per-template union: (n_templates, n_blocks) - for Q2 target approx
        self.template_compat = torch.zeros(
            self.n_templates, self.n_blocks,
            dtype=torch.bool, device=self.device)
        for bi_idx, bi_rxn in enumerate(self.tp.bi_reactions):
            self.template_compat[bi_rxn.index] |= self.bi_compat_dense[bi_idx]

        # Count compatible pairs
        n_compat = int(self.template_compat.sum().item())

        t2 = time.perf_counter()

        # 5. Compute block fingerprints -> GPU
        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            fpSize=self.fp_dim, radius=self.fp_radius)
        self.block_fps = self._compute_block_fps()

        t3 = time.perf_counter()

        # 6. Create Q-networks
        self.template_q = TemplatePolicyNetwork(
            fp_dim=self.fp_dim, n_templates=self.n_templates,
            hidden_dim=self.hidden_dim, emb_dim=self.template_emb_dim,
        ).to(self.device)

        self.template_q_target = TemplatePolicyNetwork(
            fp_dim=self.fp_dim, n_templates=self.n_templates,
            hidden_dim=self.hidden_dim, emb_dim=self.template_emb_dim,
        ).to(self.device)
        self.template_q_target.load_state_dict(self.template_q.state_dict())
        self.template_q_target.requires_grad_(False)

        self.block_q = BlockScoringNetwork(
            fp_dim=self.fp_dim, template_emb_dim=self.template_emb_dim,
            hidden_dim=self.hidden_dim, block_emb_dim=self.block_emb_dim,
        ).to(self.device)

        self.block_q_target = BlockScoringNetwork(
            fp_dim=self.fp_dim, template_emb_dim=self.template_emb_dim,
            hidden_dim=self.hidden_dim, block_emb_dim=self.block_emb_dim,
        ).to(self.device)
        self.block_q_target.load_state_dict(self.block_q.state_dict())
        self.block_q_target.requires_grad_(False)

        # 7. Precompute block embeddings
        self.block_q.precompute_block_embeddings(self.block_fps)
        self.block_q_target.precompute_block_embeddings(self.block_fps)

        # 8. Optimizer (Q1 + Q2 params)
        self.optimizer = torch.optim.Adam(
            list(self.template_q.parameters())
            + list(self.block_q.parameters()),
            lr=self.lr,
        )

        # 9. Thread pool for parallel template matching
        if self.num_workers > 0:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

        t4 = time.perf_counter()

        n_q1_params = sum(p.numel() for p in self.template_q.parameters())
        n_q2_params = sum(p.numel() for p in self.block_q.parameters())

        self._loaded = True
        print(f"  DQNTemplatePredictor ready ({t4 - t0:.1f}s total)")
        print(f"    Templates: {self.n_templates} "
              f"({sum(self.is_uni)} uni + {sum(self.is_bi)} bi)")
        print(f"    Blocks: {self.n_blocks}, compat pairs: {n_compat}")
        print(f"    Q1 params: {n_q1_params:,}, Q2 params: {n_q2_params:,}")
        print(f"    Block FP: {t3 - t2:.1f}s, "
              f"compat mask: {t2 - t1:.1f}s")

    # ------------------------------------------------------------------
    # Fingerprint computation
    # ------------------------------------------------------------------

    def _compute_mol_fp(self, mol):
        """Compute Morgan FP for one molecule -> numpy float32 array."""
        try:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return np.zeros(self.fp_dim, dtype=np.float32)
            Chem.SanitizeMol(mol)
            fp = self._fp_gen.GetFingerprint(mol)
            arr = np.zeros(self.fp_dim, dtype=np.float32)
            for idx in fp.GetOnBits():
                arr[idx] = 1.0
            return arr
        except Exception:
            return np.zeros(self.fp_dim, dtype=np.float32)

    def _compute_fps_batch(self, mols):
        """Compute FPs for multiple molecules -> GPU tensor (n, fp_dim)."""
        fps_np = np.zeros((len(mols), self.fp_dim), dtype=np.float32)

        def _compute(args):
            i, mol = args
            fps_np[i] = self._compute_mol_fp(mol)

        if self._executor and len(mols) > 1:
            list(self._executor.map(_compute, enumerate(mols)))
        else:
            for i, mol in enumerate(mols):
                _compute((i, mol))

        return torch.from_numpy(fps_np).to(self.device)

    def _compute_block_fps(self):
        """Compute Morgan FPs for all building blocks -> GPU tensor."""
        t0 = time.perf_counter()
        fps = np.zeros((self.n_blocks, self.fp_dim), dtype=np.float32)

        for i in range(self.n_blocks):
            _smi, mol = self.tp.bb_library[i]
            try:
                fp = self._fp_gen.GetFingerprint(mol)
                for bit in fp.GetOnBits():
                    fps[i, bit] = 1.0
            except Exception:
                pass

        print(f"    Block FPs computed ({self.n_blocks} x {self.fp_dim}) "
              f"in {time.perf_counter() - t0:.1f}s")
        return torch.from_numpy(fps).to(self.device)

    # ------------------------------------------------------------------
    # Template matching
    # ------------------------------------------------------------------

    def _match_templates_extended(self, mols):
        """Check template applicability + record matching bi-reaction indices.

        Parallelized with ThreadPoolExecutor (RDKit GIL-free SubstructMatch).

        Returns:
            masks: (n_mols, n_templates) bool numpy array.
            mol_bi_matches: list[dict[int, list[int]]] per molecule:
                template_id -> list of matching bi_reaction list indices.
        """
        n_mols = len(mols)
        masks = np.zeros((n_mols, self.n_templates), dtype=bool)
        mol_bi_matches = [dict() for _ in range(n_mols)]

        def _check(args):
            i, mol = args
            if mol is None:
                return
            for rxn in self.tp.uni_reactions:
                if rxn.is_reactant(mol):
                    masks[i, rxn.index] = True
            for bi_idx, rxn in enumerate(self.tp.bi_reactions):
                if rxn.is_mol_reactant(mol):
                    masks[i, rxn.index] = True
                    mol_bi_matches[i].setdefault(rxn.index, []).append(bi_idx)

        if self._executor and n_mols > 1:
            list(self._executor.map(_check, enumerate(mols)))
        else:
            for i, mol in enumerate(mols):
                _check((i, mol))

        return masks, mol_bi_matches

    # Keep simple version for observe_step (doesn't need bi_matches)
    def _match_templates_batch(self, mols):
        """Simple template mask (no orientation info)."""
        masks, _ = self._match_templates_extended(mols)
        return masks

    # ------------------------------------------------------------------
    # Action generation (API compatible with TemplateReactionPredictor)
    # ------------------------------------------------------------------

    def get_valid_actions(self, mol, top_k=None, epsilon=0.0):
        """Generate scored actions for one molecule."""
        results = self.get_valid_actions_batch(
            [mol], top_k=top_k, epsilon=epsilon)
        return results[0]

    def get_valid_actions_batch(self, mols, top_k=None, epsilon=0.0):
        """Generate scored actions using batched GPU + parallel CPU reactions.

        Pipeline:
          Phase 1: Parallel FP computation (CPU threads)
          Phase 2: Parallel template matching with orientation tracking
          Phase 3: Batch Q1 scoring (single GPU call)
          Phase 4: Template selection + batch compat mask building
          Phase 5: Batch Q2 scoring (single GPU call, ALL mol*template pairs)
          Phase 6: Block selection (GPU topk)
          Phase 7: Parallel RDKit reactions (CPU thread pool)
          Phase 8: Assemble and rank results per molecule
        """
        if top_k is None:
            top_k = self.top_k

        # Parse molecules
        mol_objects = []
        for m in mols:
            if isinstance(m, str):
                parsed = Chem.MolFromSmiles(m)
                mol_objects.append(parsed if parsed else None)
            else:
                mol_objects.append(m)
        n_mols = len(mol_objects)

        # Precompute SMILES (thread-safe, for dedup in reaction results)
        mol_smiles = [
            Chem.MolToSmiles(m) if m is not None else ''
            for m in mol_objects]

        # Phase 1: Parallel FP computation -> GPU
        fps = self._compute_fps_batch(mol_objects)

        # Phase 2: Extended template matching (parallel, records bi_rxn indices)
        template_masks, mol_bi_matches = self._match_templates_extended(
            mol_objects)
        template_masks_gpu = torch.from_numpy(template_masks).to(self.device)

        # Phase 3: Batch Q1 scoring (single GPU forward)
        with torch.no_grad():
            template_scores = self.template_q(fps, template_masks_gpu)
            template_scores_np = template_scores.cpu().numpy()

        # Cache for observe_step
        self._last_fps_np = fps.cpu().numpy()
        self._last_template_masks = template_masks
        self._last_batch_metadata = [[] for _ in range(n_mols)]

        k1 = min(self.top_k_templates, self.n_templates)
        k2 = self.top_k_blocks

        # Phase 4: Template selection per mol + collect Q2 tasks
        # selected_templates[i] = list of template_ids for molecule i
        selected_templates = [[] for _ in range(n_mols)]
        uni_tasks = []   # (mol_idx, t_id, rxn)
        q2_tasks = []    # (mol_idx, t_id, matching_bi_indices)

        for i in range(n_mols):
            if mol_objects[i] is None:
                continue
            valid_ids = np.where(template_masks[i])[0]
            if len(valid_ids) == 0:
                continue

            t_scores = template_scores_np[i]
            scored = sorted(
                ((t_id, t_scores[t_id]) for t_id in valid_ids),
                key=lambda x: x[1], reverse=True)

            if self.rng.random() < epsilon:
                self.rng.shuffle(scored)

            sel = [t_id for t_id, _ in scored[:k1]]
            selected_templates[i] = sel

            for t_id in sel:
                if self.is_uni[t_id]:
                    rxn = self.uni_by_template.get(t_id)
                    if rxn:
                        uni_tasks.append((i, int(t_id), rxn))
                if self.is_bi[t_id]:
                    bi_indices = mol_bi_matches[i].get(t_id, [])
                    if bi_indices:
                        q2_tasks.append((i, int(t_id), bi_indices))

        # Phase 5: Batch Q2 scoring — single GPU call for ALL (mol, template)
        # Build batch tensors
        n_q2 = len(q2_tasks)
        q2_block_top = {}  # (mol_idx, t_id) -> (top_block_ids, block_scores)

        if n_q2 > 0:
            q2_fps = fps[[t[0] for t in q2_tasks]]  # (n_q2, fp_dim) GPU
            q2_t_ids = torch.tensor(
                [t[1] for t in q2_tasks], dtype=torch.long, device=self.device)
            q2_t_embs = self.template_q.get_template_embedding(
                q2_t_ids)  # (n_q2, emb_dim)

            # Build per-task orientation-specific compat masks
            q2_compat = torch.zeros(
                n_q2, self.n_blocks, dtype=torch.bool, device=self.device)
            for j, (mol_idx, t_id, bi_indices) in enumerate(q2_tasks):
                for bi_idx in bi_indices:
                    q2_compat[j] |= self.bi_compat_dense[bi_idx]

            # Single batch Q2 forward
            with torch.no_grad():
                q2_scores_all = self.block_q(
                    q2_fps, q2_t_embs, q2_compat)  # (n_q2, n_blocks)

            # Phase 6: Select top blocks per task (GPU topk)
            sample_k = k2 * 3  # oversample to handle reaction failures
            for j in range(n_q2):
                mol_idx, t_id, bi_indices = q2_tasks[j]
                n_compat = int(q2_compat[j].sum().item())
                if n_compat == 0:
                    continue
                actual_k = min(sample_k, n_compat)

                if self.rng.random() < epsilon:
                    compat_idx = q2_compat[j].nonzero(as_tuple=True)[0]
                    perm = torch.randperm(
                        len(compat_idx), device=self.device)[:actual_k]
                    top_ids = compat_idx[perm].cpu().numpy()
                else:
                    _, top_ids = q2_scores_all[j].topk(actual_k)
                    top_ids = top_ids.cpu().numpy()

                scores_np = q2_scores_all[j].cpu().numpy()
                q2_block_top[(mol_idx, t_id)] = (top_ids, scores_np)

        # Phase 7: Parallel reactions — dispatch ALL to thread pool
        # Build flat task list
        rxn_tasks = []

        # Uni-molecular tasks
        for mol_idx, t_id, rxn in uni_tasks:
            rxn_tasks.append({
                'type': 'uni',
                'mol_idx': mol_idx,
                't_id': t_id,
                'rxn': rxn,
                'mol': mol_objects[mol_idx],
                'mol_smi': mol_smiles[mol_idx],
                'q_score': float(template_scores_np[mol_idx, t_id]),
            })

        # Bi-molecular tasks (one per block candidate)
        for j, (mol_idx, t_id, bi_indices) in enumerate(q2_tasks):
            block_info = q2_block_top.get((mol_idx, t_id))
            if block_info is None:
                continue
            top_ids, scores_np = block_info
            matching_rxns = [self.tp.bi_reactions[bi] for bi in bi_indices]

            for blk_idx in top_ids:
                blk_smi, blk_mol = self.tp.bb_library[int(blk_idx)]
                rxn_tasks.append({
                    'type': 'bi',
                    'mol_idx': mol_idx,
                    't_id': t_id,
                    'blk_idx': int(blk_idx),
                    'blk_smi': blk_smi,
                    'blk_mol': blk_mol,
                    'matching_rxns': matching_rxns,
                    'mol': mol_objects[mol_idx],
                    'mol_smi': mol_smiles[mol_idx],
                    'q_score': float(scores_np[blk_idx]),
                })

        # Execute reactions in parallel
        def _run_reaction(task):
            if task['type'] == 'uni':
                prods = task['rxn'].forward_smiles(task['mol'])
                for prod in prods:
                    if prod and prod != task['mol_smi']:
                        return {
                            'mol_idx': task['mol_idx'],
                            'co_reactant': '',
                            'product': prod,
                            'q_score': task['q_score'],
                            'template_id': task['t_id'],
                            'block_id': -1,
                        }
                return None
            else:  # bi
                for bi_rxn in task['matching_rxns']:
                    if not bi_rxn.is_block_reactant(task['blk_mol']):
                        continue
                    prods = bi_rxn.forward_smiles(task['mol'], task['blk_mol'])
                    for prod in prods:
                        if prod and prod != task['mol_smi']:
                            return {
                                'mol_idx': task['mol_idx'],
                                'co_reactant': task['blk_smi'],
                                'product': prod,
                                'q_score': task['q_score'],
                                'template_id': task['t_id'],
                                'block_id': task['blk_idx'],
                            }
                return None

        if self._executor and len(rxn_tasks) > 1:
            rxn_results = list(self._executor.map(_run_reaction, rxn_tasks))
        else:
            rxn_results = [_run_reaction(t) for t in rxn_tasks]

        # Phase 8: Assemble results per molecule
        per_mol_actions = [[] for _ in range(n_mols)]
        for res in rxn_results:
            if res is not None:
                per_mol_actions[res['mol_idx']].append(res)

        results = []
        for i in range(n_mols):
            actions = per_mol_actions[i]

            # Deduplicate by product SMILES, keep best score
            seen = {}
            for act in actions:
                prod = act['product']
                if prod not in seen or act['q_score'] > seen[prod]['q_score']:
                    seen[prod] = act
            unique = sorted(
                seen.values(), key=lambda x: x['q_score'], reverse=True)
            unique = unique[:top_k]

            co_reactants = [a['co_reactant'] for a in unique]
            products = [a['product'] for a in unique]
            scores = np.array(
                [a['q_score'] for a in unique], dtype=np.float32)
            metadata = [{'template_id': a['template_id'],
                         'block_id': a['block_id'],
                         'q_score': a['q_score']}
                        for a in unique]

            results.append((co_reactants, products, scores))
            self._last_batch_metadata[i] = metadata

        return results

    # ------------------------------------------------------------------
    # Training: transition storage and DQN update
    # ------------------------------------------------------------------

    def store_transition(self, mol_fp, template_id, block_id, reward,
                         next_mol_fp, done, template_mask,
                         next_template_mask):
        """Store one transition in the replay buffer.

        Args:
            mol_fp: numpy (fp_dim,) float32.
            template_id: int (which template was selected).
            block_id: int (-1 for uni-molecular).
            reward: float.
            next_mol_fp: numpy (fp_dim,) float32.
            done: bool.
            template_mask: numpy (n_templates,) bool.
            next_template_mask: numpy (n_templates,) bool.
        """
        transition = {
            'mol_fp': mol_fp,
            'template_id': template_id,
            'block_id': block_id,
            'reward': reward,
            'next_mol_fp': next_mol_fp,
            'done': done,
            'template_mask': template_mask,
            'next_template_mask': next_template_mask,
        }
        if len(self.replay_buffer) < self.replay_size:
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[self.replay_pos] = transition
        self.replay_pos = (self.replay_pos + 1) % self.replay_size

    def observe_step(self, action_indices, rewards, next_mols, dones):
        """Store transitions using cached metadata from get_valid_actions_batch.

        Call after env.step() to record what happened.

        Args:
            action_indices: list of int (which action was selected per mol).
            rewards: list of float.
            next_mols: list of Mol objects (next states).
            dones: list of bool.
        """
        if self._last_fps_np is None:
            return

        # Compute next-state info
        next_fps = self._compute_fps_batch(next_mols).cpu().numpy()
        next_masks = self._match_templates_batch(next_mols)

        for i, (idx, reward, done) in enumerate(
                zip(action_indices, rewards, dones)):
            if i >= len(self._last_batch_metadata):
                continue
            meta_list = self._last_batch_metadata[i]
            if idx < 0 or idx >= len(meta_list):
                continue
            meta = meta_list[idx]
            if meta['template_id'] < 0:
                continue

            self.store_transition(
                mol_fp=self._last_fps_np[i],
                template_id=meta['template_id'],
                block_id=meta['block_id'],
                reward=reward,
                next_mol_fp=next_fps[i],
                done=done,
                template_mask=self._last_template_masks[i],
                next_template_mask=next_masks[i],
            )

    def train_step(self, batch_size=64):
        """One DQN gradient step for both Q1 and Q2.

        Returns:
            dict with loss_q1, loss_q2, q1_mean, q2_mean (empty if not enough data).
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = random.sample(self.replay_buffer, batch_size)

        # Collate batch -> GPU tensors
        mol_fps = torch.tensor(
            np.array([t['mol_fp'] for t in batch]),
            dtype=torch.float32, device=self.device)
        template_ids = torch.tensor(
            [t['template_id'] for t in batch],
            dtype=torch.long, device=self.device)
        block_ids = torch.tensor(
            [t['block_id'] for t in batch],
            dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [t['reward'] for t in batch],
            dtype=torch.float32, device=self.device)
        next_mol_fps = torch.tensor(
            np.array([t['next_mol_fp'] for t in batch]),
            dtype=torch.float32, device=self.device)
        dones = torch.tensor(
            [float(t['done']) for t in batch],
            dtype=torch.float32, device=self.device)
        template_masks = torch.tensor(
            np.array([t['template_mask'] for t in batch]),
            dtype=torch.bool, device=self.device)
        next_template_masks = torch.tensor(
            np.array([t['next_template_mask'] for t in batch]),
            dtype=torch.bool, device=self.device)

        # ---- Q1 Update ----
        q1_all = self.template_q(mol_fps, template_masks)
        q1_current = q1_all.gather(
            1, template_ids.unsqueeze(1)).squeeze(1)  # (batch,)

        with torch.no_grad():
            q1_target_all = self.template_q_target(
                next_mol_fps, next_template_masks)
            # Handle molecules with NO valid next templates
            q1_next_max = q1_target_all.max(dim=1).values
            no_valid = ~next_template_masks.any(dim=1)
            q1_next_max = q1_next_max.masked_fill(no_valid, 0.0)
            q1_target = rewards + (1 - dones) * self.gamma * q1_next_max

        loss_q1 = F.smooth_l1_loss(q1_current, q1_target)

        # ---- Q2 Update (only for bi-molecular actions) ----
        bi_mask = block_ids >= 0
        loss_q2 = torch.tensor(0.0, device=self.device)

        if bi_mask.any():
            bi_idx = bi_mask.nonzero(as_tuple=True)[0]
            bi_mol_fps = mol_fps[bi_idx]
            bi_template_ids = template_ids[bi_idx]
            bi_block_ids = block_ids[bi_idx]
            bi_rewards = rewards[bi_idx]
            bi_next_fps = next_mol_fps[bi_idx]
            bi_dones = dones[bi_idx]
            bi_next_masks = next_template_masks[bi_idx]

            # Template embeddings for selected templates
            t_embs = self.template_q.get_template_embedding(
                bi_template_ids)  # (bi_batch, emb_dim)

            # Block FPs for selected blocks (for gradient flow)
            bi_block_fps = self.block_fps[bi_block_ids]  # (bi_batch, fp_dim)

            # Current Q2 with gradient through both sides
            q2_current = self.block_q.score_single_block(
                bi_mol_fps, t_embs.detach(), bi_block_fps)

            # Target Q2
            with torch.no_grad():
                # Best template for next state via Q1
                q1_next = self.template_q_target(bi_next_fps, bi_next_masks)
                has_valid = bi_next_masks.any(dim=1)
                # Default to template 0 for mols with no valid templates
                best_next_t = torch.zeros(
                    len(bi_idx), dtype=torch.long, device=self.device)
                if has_valid.any():
                    best_next_t[has_valid] = q1_next[has_valid].argmax(dim=1)

                # Template embeddings and compat for best next templates
                t_embs_next = self.template_q_target.get_template_embedding(
                    best_next_t)
                compat_next = self.template_compat[best_next_t]

                # Q2 target: max over compatible blocks
                q2_all_next = self.block_q_target(
                    bi_next_fps, t_embs_next, compat_next)
                q2_next_max = q2_all_next.max(dim=1).values

                # Handle: no valid template or no compatible blocks
                no_blocks = ~compat_next.any(dim=1) | ~has_valid
                q2_next_max = q2_next_max.masked_fill(no_blocks, 0.0)

                q2_target = (bi_rewards
                             + (1 - bi_dones) * self.gamma * q2_next_max)

            loss_q2 = F.smooth_l1_loss(q2_current, q2_target)

        # Combined optimization
        loss = loss_q1 + loss_q2
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.template_q.parameters())
            + list(self.block_q.parameters()),
            max_norm=10.0)
        self.optimizer.step()

        # Soft update target networks
        with torch.no_grad():
            for p, pt in zip(self.template_q.parameters(),
                             self.template_q_target.parameters()):
                pt.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)
            for p, pt in zip(self.block_q.parameters(),
                             self.block_q_target.parameters()):
                pt.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)

        self._train_steps += 1

        # Periodically refresh block embeddings
        if (self._train_steps % self.block_emb_refresh_freq == 0):
            self.block_q.precompute_block_embeddings(self.block_fps)
            self.block_q_target.precompute_block_embeddings(self.block_fps)

        return {
            'loss_q1': loss_q1.item(),
            'loss_q2': loss_q2.item(),
            'q1_mean': q1_current.mean().item(),
            'q2_mean': (q2_current.mean().item()
                        if bi_mask.any() else 0.0),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        """Save Q-networks, optimizer, and replay buffer stats."""
        torch.save({
            'template_q': self.template_q.state_dict(),
            'template_q_target': self.template_q_target.state_dict(),
            'block_q': self.block_q.state_dict(),
            'block_q_target': self.block_q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self._train_steps,
            'replay_size_current': len(self.replay_buffer),
        }, path)

    def load_checkpoint(self, path):
        """Load Q-networks and optimizer from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.template_q.load_state_dict(ckpt['template_q'])
        self.template_q_target.load_state_dict(ckpt['template_q_target'])
        self.block_q.load_state_dict(ckpt['block_q'])
        self.block_q_target.load_state_dict(ckpt['block_q_target'])
        if 'optimizer' in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self._train_steps = ckpt.get('train_steps', 0)
        # Refresh block embeddings with loaded weights
        self.block_q.precompute_block_embeddings(self.block_fps)
        self.block_q_target.precompute_block_embeddings(self.block_fps)
        print(f"  Loaded checkpoint ({self._train_steps} train steps)")

    def close(self):
        """Shut down thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
