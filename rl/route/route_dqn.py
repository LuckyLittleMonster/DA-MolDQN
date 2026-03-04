"""Route-DQN agent: hierarchical Q-learning for synthesis route optimization.

Two-level action selection:
  Q_pos: Select which route position to modify (L1 mask).
  Q_bb:  Select which building block to use (L2+L3 mask).

Route state is encoded via RouteStateEncoder (mean pooling over step features).
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# ── Prioritized Experience Replay ─────────────────────────────────────

class SumTree:
    """Binary tree where each leaf stores a priority value.

    Parent nodes store the sum of children, enabling O(log n) proportional
    sampling via tree traversal.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        end = self.capacity - 1 + self.size
        if self.size == 0:
            return 1.0
        return float(self.tree[self.capacity - 1:end].max())

    def add(self, priority: float, data):
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self._update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update(self, idx: int, priority: float):
        self._update(idx, priority)

    def get(self, cumsum: float):
        """Sample a leaf by cumulative sum. Returns (tree_idx, priority, data)."""
        idx = 0
        while idx < self.capacity - 1:  # not a leaf
            left = 2 * idx + 1
            if cumsum <= self.tree[left]:
                idx = left
            else:
                cumsum -= self.tree[left]
                idx = left + 1
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree.

    Args:
        capacity: Maximum buffer size.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta_start: Initial IS weight exponent.
        beta_end: Final IS weight exponent (annealed over training).
        eps: Small constant added to TD-error for non-zero priority.
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 eps: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start
        self.eps = eps
        self.capacity = capacity

    def __len__(self) -> int:
        return self.tree.size

    def anneal_beta(self, progress: float):
        """Anneal beta from beta_start to beta_end. progress in [0, 1]."""
        self.beta = self.beta_start + progress * (self.beta_end - self.beta_start)

    def add(self, transition):
        """Add transition with max priority (ensures new samples get replayed)."""
        priority = max(self.tree.max_priority, 1.0) ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int):
        """Sample batch proportional to priorities.

        Returns:
            batch: list of transitions
            tree_indices: list of SumTree indices (for priority update)
            is_weights: np.ndarray of importance sampling weights
        """
        batch = []
        tree_indices = []
        priorities = []

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            cumsum = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(cumsum)
            if data is None:
                # Edge case: empty slot, resample
                cumsum = random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(cumsum)
            batch.append(data)
            tree_indices.append(idx)
            priorities.append(priority)

        # Importance sampling weights
        priorities = np.array(priorities, dtype=np.float64)
        probs = priorities / (self.tree.total + 1e-10)
        is_weights = (self.tree.size * probs) ** (-self.beta)
        is_weights /= is_weights.max()  # normalize

        return batch, tree_indices, is_weights.astype(np.float32)

    def update_priorities(self, tree_indices: list[int],
                          td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, td_err in zip(tree_indices, td_errors):
            priority = (abs(td_err) + self.eps) ** self.alpha
            self.tree.update(idx, priority)

from .route import (
    SynthesisRoute, cascade_validate, update_route,
    extend_route, truncate_route,
    _validate_one_candidate_fast, _mp_validate_route,
)
from .policy_network import (
    RouteStateEncoder,
    PositionPolicyNetwork,
    RouteBBScoringNetwork,
)


def _mp_scan_applicable(smi):
    """mp.Pool worker: find applicable bi-reactions for a SMILES.

    Uses route._mp_tp (set before Pool creation via fork).
    Returns (smi, list[(rxn_idx, l2_array)]).
    """
    from .route import _mp_tp
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, []
    applicable = []
    bi_reactions = _mp_tp.bi_reactions
    bi_compat = _mp_tp.bi_compat
    for idx in range(len(bi_reactions)):
        if bi_reactions[idx].is_mol_reactant(mol):
            l2 = bi_compat.get(idx)
            if l2 is not None and len(l2) > 0:
                applicable.append((idx, l2))
    return smi, applicable


class RouteDQN:
    """Route-DQN agent for synthesis route optimization.

    Args:
        tp: Loaded TemplateReactionPredictor.
        device: 'cuda' or 'cpu'.
        fp_dim: Morgan fingerprint dimension.
        fp_radius: Morgan fingerprint radius.
        template_emb_dim: Template embedding dimension.
        block_emb_dim: Block embedding dimension.
        route_emb_dim: Route state embedding dimension.
        max_route_len: Maximum steps in a route.
        hidden_dim: MLP hidden dimension.
        lr: Learning rate.
        gamma: Discount factor.
        polyak: Soft update coefficient for target networks.
        replay_size: Maximum replay buffer capacity.
        block_emb_refresh_freq: How often to refresh block embeddings.
        cascade_workers: Thread workers for cascade validation.
    """

    def __init__(
        self,
        tp,
        device='cuda',
        fp_dim=4096,
        fp_radius=3,
        template_emb_dim=128,
        block_emb_dim=64,
        route_emb_dim=256,
        max_route_len=10,
        hidden_dim=256,
        lr=1e-4,
        gamma=0.9,
        polyak=0.995,
        replay_size=50000,
        block_emb_refresh_freq=200,
        cascade_workers=8,
        mp_pool=None,
        enable_variable_length=True,
        min_route_len=2,
        enable_exp=False,
        subsample_k=512,
    ):
        self.tp = tp
        self.device = device
        self.fp_dim = fp_dim
        self.fp_radius = fp_radius
        self.template_emb_dim = template_emb_dim
        self.block_emb_dim = block_emb_dim
        self.route_emb_dim = route_emb_dim
        self.max_route_len = max_route_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.block_emb_refresh_freq = block_emb_refresh_freq
        self._mp_pool = mp_pool

        # Variable-length route support
        self.enable_variable_length = enable_variable_length
        self.min_route_len = min_route_len
        self.EXTEND_POS = max_route_len
        self.TRUNCATE_POS = max_route_len + 1
        self.n_positions = max_route_len + 2

        self.n_blocks = len(tp.bb_library)
        self.n_templates = max(
            max((r.index for r in tp.uni_reactions), default=0),
            max((r.index for r in tp.bi_reactions), default=0),
        ) + 1

        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            fpSize=fp_dim, radius=fp_radius)

        # Template embeddings (learned, shared between Q_pos context and Q_bb)
        self.template_embs = nn.Embedding(
            self.n_templates, template_emb_dim).to(device)

        # Block embeddings lookup (for route state encoding)
        # We use a small projection from block FP
        self._block_fp_proj = nn.Linear(fp_dim, block_emb_dim).to(device)

        # Route state encoder
        self.route_encoder = RouteStateEncoder(
            fp_dim=fp_dim,
            template_emb_dim=template_emb_dim,
            block_emb_dim=block_emb_dim,
            step_hidden=hidden_dim,
            route_emb_dim=route_emb_dim,
        ).to(device)

        # Q_pos: position selection
        self.q_pos = PositionPolicyNetwork(
            route_emb_dim=route_emb_dim,
            max_route_len=max_route_len,
            hidden_dim=hidden_dim // 2,
        ).to(device)

        # Q_bb: building block selection
        self.q_bb = RouteBBScoringNetwork(
            route_emb_dim=route_emb_dim,
            fp_dim=fp_dim,
            template_emb_dim=template_emb_dim,
            hidden_dim=hidden_dim,
            block_emb_dim=block_emb_dim,
        ).to(device)

        # Target networks
        self.q_pos_target = PositionPolicyNetwork(
            route_emb_dim=route_emb_dim,
            max_route_len=max_route_len,
            hidden_dim=hidden_dim // 2,
        ).to(device)
        self.q_pos_target.load_state_dict(self.q_pos.state_dict())
        self.q_pos_target.requires_grad_(False)

        self.q_bb_target = RouteBBScoringNetwork(
            route_emb_dim=route_emb_dim,
            fp_dim=fp_dim,
            template_emb_dim=template_emb_dim,
            hidden_dim=hidden_dim,
            block_emb_dim=block_emb_dim,
        ).to(device)
        self.q_bb_target.load_state_dict(self.q_bb.state_dict())
        self.q_bb_target.requires_grad_(False)

        # Route encoder target (for encoding next-state routes)
        self.route_encoder_target = RouteStateEncoder(
            fp_dim=fp_dim,
            template_emb_dim=template_emb_dim,
            block_emb_dim=block_emb_dim,
            step_hidden=hidden_dim,
            route_emb_dim=route_emb_dim,
        ).to(device)
        self.route_encoder_target.load_state_dict(
            self.route_encoder.state_dict())
        self.route_encoder_target.requires_grad_(False)

        # Precompute block FPs
        self.block_fps = self._compute_all_block_fps()

        # Precompute block embeddings for Q_bb inference
        self.q_bb.precompute_block_embeddings(self.block_fps)
        self.q_bb_target.precompute_block_embeddings(self.block_fps)

        # Optimizer (all trainable params)
        self._all_params = (
            list(self.template_embs.parameters())
            + list(self._block_fp_proj.parameters())
            + list(self.route_encoder.parameters())
            + list(self.q_pos.parameters())
            + list(self.q_bb.parameters())
        )
        self.optimizer = torch.optim.Adam(self._all_params, lr=lr)

        # Experimental improvements
        self.enable_exp = enable_exp
        self.subsample_k = subsample_k

        # Replay buffer (PER when enable_exp, else uniform)
        if enable_exp:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_size, alpha=0.6,
                beta_start=0.4, beta_end=1.0)
            self._uniform_replay = False
            print(f"  [EXP] PER enabled (alpha=0.6, beta=0.4->1.0)")
            print(f"  [EXP] Double DQN enabled")
            print(f"  [EXP] Action subsampling: k={subsample_k}")
            print(f"  [EXP] Reward shaping: F=gamma*QED(s')-QED(s)")
        else:
            self._uniform_replay = True
        self.replay_buffer_list: list[dict] = []  # uniform fallback
        self.replay_pos = 0

        # Cascade validation: prefer mp_pool (true parallelism) over threads
        if mp_pool is not None:
            self._cascade_executor = None
        else:
            self._cascade_executor = (
                ThreadPoolExecutor(max_workers=cascade_workers)
                if cascade_workers > 0 else None)

        self._train_steps = 0

        # Extend optimization: applicability cache (mp_pool reused from cascade)
        self._applicable_cache: dict[str, list[tuple]] = {}

        # Log param counts
        n_params = sum(p.numel() for p in self._all_params)
        print(f"  RouteDQN: {n_params:,} trainable params")

    # ------------------------------------------------------------------
    # Fingerprint helpers
    # ------------------------------------------------------------------

    def _compute_fp(self, smi: str) -> np.ndarray:
        """Compute Morgan FP for one SMILES -> numpy float32."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.zeros(self.fp_dim, dtype=np.float32)
        try:
            fp = self._fp_gen.GetFingerprint(mol)
            arr = np.zeros(self.fp_dim, dtype=np.float32)
            for idx in fp.GetOnBits():
                arr[idx] = 1.0
            return arr
        except Exception:
            return np.zeros(self.fp_dim, dtype=np.float32)

    def _compute_fp_tensor(self, smi: str) -> torch.Tensor:
        """Compute Morgan FP for one SMILES -> GPU tensor."""
        return torch.from_numpy(
            self._compute_fp(smi)).to(self.device)

    def _compute_all_block_fps(self) -> torch.Tensor:
        """Compute FPs for all building blocks -> (n_blocks, fp_dim) GPU."""
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
        elapsed = time.perf_counter() - t0
        print(f"  Block FPs: {self.n_blocks} x {self.fp_dim} in {elapsed:.1f}s")
        return torch.from_numpy(fps).to(self.device)

    # ------------------------------------------------------------------
    # Route encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_route(self, route: SynthesisRoute,
                     use_target: bool = False) -> torch.Tensor:
        """Encode a single route into a state embedding.

        Returns: (route_emb_dim,) tensor.
        """
        return self.encode_routes([route], use_target=use_target).squeeze(0)

    @torch.no_grad()
    def encode_routes(self, routes: list[SynthesisRoute],
                      use_target: bool = False) -> torch.Tensor:
        """Encode multiple routes into state embeddings.

        Returns: (batch, route_emb_dim) tensor.
        """
        encoder = self.route_encoder_target if use_target else self.route_encoder

        # Collect step features
        all_fps = []
        all_template_ids = []
        all_block_fps = []
        route_lengths = []

        block_idx_list = []  # collect for GPU indexing

        for route in routes:
            n_steps = len(route.steps)
            route_lengths.append(n_steps)

            for step in route.steps:
                # Intermediate FP (CPU, then batch transfer)
                all_fps.append(self._compute_fp(step.intermediate_smi))
                # Template ID
                all_template_ids.append(step.template_idx)
                # Block index for GPU-side lookup (-1 => zero FP)
                if step.is_uni or step.block_idx < 0:
                    block_idx_list.append(-1)
                else:
                    block_idx_list.append(step.block_idx)

        if not all_fps:
            return torch.zeros(
                len(routes), self.route_emb_dim, device=self.device)

        step_fps = torch.tensor(
            np.array(all_fps), dtype=torch.float32, device=self.device)
        template_ids = torch.tensor(
            all_template_ids, dtype=torch.long, device=self.device)
        # Clamp -1 to 0 for indexing, then zero out invalid entries
        safe_template_ids = template_ids.clamp(min=0)
        step_template_embs = self.template_embs(safe_template_ids)
        invalid_template_mask = (template_ids < 0).unsqueeze(-1)
        step_template_embs = step_template_embs.masked_fill(
            invalid_template_mask, 0.0)

        # GPU-side block FP lookup (avoids per-step CPU↔GPU transfers)
        block_indices_t = torch.tensor(
            block_idx_list, dtype=torch.long, device=self.device)
        # Clamp -1 to 0 for indexing, then zero out uni-molecular entries
        safe_indices = block_indices_t.clamp(min=0)
        block_fps_tensor = self.block_fps[safe_indices]  # (total_steps, fp_dim)
        uni_mask = (block_indices_t < 0).unsqueeze(-1)  # (total_steps, 1)
        block_fps_tensor = block_fps_tensor.masked_fill(uni_mask, 0.0)
        step_block_embs = self._block_fp_proj(block_fps_tensor)

        lengths = torch.tensor(
            route_lengths, dtype=torch.long, device=self.device)

        return encoder(step_fps, step_template_embs, step_block_embs, lengths)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_position(self, route: SynthesisRoute,
                        epsilon: float) -> int:
        """Select which position to modify using Q_pos + epsilon-greedy.

        Args:
            route: Current route.
            epsilon: Exploration probability.

        Returns:
            Position index.
        """
        valid_positions = [
            i for i, m in enumerate(route.modifiable_mask) if m]
        if not valid_positions:
            return 0  # fallback (shouldn't happen with well-formed routes)

        if random.random() < epsilon:
            return random.choice(valid_positions)

        route_state = self.encode_route(route)
        position_mask = torch.zeros(
            self.n_positions, dtype=torch.bool, device=self.device)
        for p in valid_positions:
            if p < self.n_positions:
                position_mask[p] = True
        # Add EXTEND/TRUNCATE if variable-length is enabled
        if self.enable_variable_length:
            if len(route) < self.max_route_len:
                position_mask[self.EXTEND_POS] = True
            if len(route) > self.min_route_len:
                position_mask[self.TRUNCATE_POS] = True

        q_pos = self.q_pos(
            route_state.unsqueeze(0), position_mask.unsqueeze(0))
        return q_pos.squeeze(0).argmax().item()

    def get_valid_blocks(self, route: SynthesisRoute,
                         position: int) -> list[int]:
        """Get L2+L3 validated candidate block indices."""
        step = route.steps[position]
        if step.is_uni:
            return []

        # L2: template compatibility
        l2_candidates = self.tp.bi_compat.get(step.bi_rxn_idx)
        if l2_candidates is None or len(l2_candidates) == 0:
            return []

        # L3: cascade validation (parallel if executor available)
        return cascade_validate(
            route, position, l2_candidates.tolist(), self.tp,
            executor=self._cascade_executor)

    def select_block(self, route: SynthesisRoute, position: int,
                     valid_block_indices: list[int],
                     epsilon: float) -> int:
        """Select BB from validated candidates using Q_bb + epsilon-greedy.

        Args:
            route: Current route.
            position: Step index being modified.
            valid_block_indices: L2+L3 validated BB indices.
            epsilon: Exploration probability.

        Returns:
            Selected block index.
        """
        if not valid_block_indices:
            return route.steps[position].block_idx  # keep current

        if random.random() < epsilon:
            return random.choice(valid_block_indices)

        route_state = self.encode_route(route)
        step = route.steps[position]

        # Position's input FP
        if position == 0:
            pos_fp = self._compute_fp_tensor(route.init_mol_smi)
        else:
            pos_fp = self._compute_fp_tensor(
                route.steps[position - 1].intermediate_smi)

        safe_tidx = max(step.template_idx, 0)
        template_emb = self.template_embs(
            torch.tensor(safe_tidx, device=self.device))

        # Build mask
        compat_mask = torch.zeros(
            self.n_blocks, dtype=torch.bool, device=self.device)
        compat_mask[valid_block_indices] = True

        q_bb = self.q_bb(
            route_state.unsqueeze(0),
            pos_fp.unsqueeze(0),
            template_emb.unsqueeze(0),
            compat_mask.unsqueeze(0))

        return q_bb.squeeze(0).argmax().item()

    def act(self, route: SynthesisRoute, epsilon: float):
        """Full action: select position then select BB.

        Returns:
            (position, block_idx, valid_blocks) tuple.
        """
        position = self.select_position(route, epsilon)

        # Ensure position is valid
        if route.steps[position].is_uni:
            valid_positions = [
                i for i, m in enumerate(route.modifiable_mask) if m]
            if valid_positions:
                position = random.choice(valid_positions)
            else:
                return position, route.steps[position].block_idx, []

        valid_blocks = self.get_valid_blocks(route, position)

        if not valid_blocks:
            # No valid candidates at this position, keep current BB
            return position, route.steps[position].block_idx, []

        block_idx = self.select_block(route, position, valid_blocks, epsilon)
        return position, block_idx, valid_blocks

    # ------------------------------------------------------------------
    # Batch action selection
    # ------------------------------------------------------------------

    def act_batch(self, routes: list[SynthesisRoute],
                  epsilon: float, cascade_top_k: int = 64) -> dict:
        """Batch action selection for all routes.

        Optimized flow: Q_bb scores first (cheap matmul with L2 mask),
        then cascade-validates only top-k candidates per route.

        Supports three action types:
          - swap: Replace BB at selected position (original behavior)
          - extend: Append a new reaction step at route end
          - truncate: Remove the last route step

        Returns dict with:
            positions: list[int]
            block_indices: list[int]
            valid_blocks_list: list[list[int]]
            action_types: list[str] ('swap', 'extend', 'truncate')
            extend_bi_rxn_indices: list[int] (bi_rxn_idx for extend, -1 otherwise)
            route_states: np.ndarray (n, route_emb_dim)
            position_masks: np.ndarray (n, n_positions) bool
            pos_fps: list[np.ndarray]
            template_indices: list[int]
            timings: dict[str, float] (phase durations in seconds)
        """
        n = len(routes)
        timings = {}

        # --- Phase 1: Batch encode all routes (1 GPU forward) ---
        t0 = time.perf_counter()
        route_states = self.encode_routes(routes)  # (n, route_emb_dim)
        timings['encode'] = time.perf_counter() - t0

        # --- Phase 2: Batch Q_pos (1 GPU forward) ---
        t0 = time.perf_counter()
        pos_masks_np = np.zeros((n, self.n_positions), dtype=bool)
        valid_pos_lists = []
        for i, route in enumerate(routes):
            vp = [j for j, m in enumerate(route.modifiable_mask)
                  if m and j < self.max_route_len]
            if self.enable_variable_length:
                if len(route) < self.max_route_len:
                    vp.append(self.EXTEND_POS)
                    pos_masks_np[i, self.EXTEND_POS] = True
                if len(route) > self.min_route_len:
                    vp.append(self.TRUNCATE_POS)
                    pos_masks_np[i, self.TRUNCATE_POS] = True
            valid_pos_lists.append(vp)
            for p in vp:
                if p < self.max_route_len:
                    pos_masks_np[i, p] = True

        pos_masks = torch.tensor(
            pos_masks_np, dtype=torch.bool, device=self.device)

        # Per-route epsilon-greedy
        explore = np.random.random(n) < epsilon

        with torch.no_grad():
            q_pos_all = self.q_pos(route_states, pos_masks)
        q_pos_argmax = q_pos_all.argmax(dim=1).cpu().numpy()

        positions = []
        for i in range(n):
            vp = valid_pos_lists[i]
            if not vp:
                positions.append(0)
            elif explore[i]:
                positions.append(random.choice(vp))
            else:
                pos = int(q_pos_argmax[i])
                positions.append(pos if pos in vp else vp[0])

        # Ensure swap positions are not uni
        for i, route in enumerate(routes):
            pos = positions[i]
            if pos < len(route.steps) and route.steps[pos].is_uni:
                swap_vp = [p for p in valid_pos_lists[i]
                           if p < self.max_route_len]
                if swap_vp:
                    positions[i] = random.choice(swap_vp)
        timings['q_pos'] = time.perf_counter() - t0

        # --- Classify action types ---
        action_types = []
        for i in range(n):
            pos = positions[i]
            if pos == self.EXTEND_POS:
                action_types.append('extend')
            elif pos == self.TRUNCATE_POS:
                action_types.append('truncate')
            else:
                action_types.append('swap')

        # --- Phase 3: Q_bb scoring with L2 mask only (for swap routes) ---
        t0 = time.perf_counter()
        pos_fps_list = []
        template_idx_list = []
        l2_masks = torch.zeros(
            n, self.n_blocks, dtype=torch.bool, device=self.device)

        for i, route in enumerate(routes):
            pos = positions[i]
            atype = action_types[i]

            if atype == 'swap':
                step = route.steps[pos]
                if pos == 0:
                    pos_fps_list.append(
                        self._compute_fp(route.init_mol_smi))
                else:
                    pos_fps_list.append(self._compute_fp(
                        route.steps[pos - 1].intermediate_smi))
                template_idx_list.append(step.template_idx)
                if not step.is_uni:
                    l2 = self.tp.bi_compat.get(step.bi_rxn_idx)
                    if l2 is not None and len(l2) > 0:
                        # Action subsampling: subsample L2 BBs if too many
                        if (self.enable_exp
                                and len(l2) > self.subsample_k):
                            subset = np.random.choice(
                                l2, size=self.subsample_k, replace=False)
                            l2_masks[i, subset] = True
                        else:
                            l2_masks[i, l2] = True
            elif atype == 'extend':
                # Use final product FP as position context
                pos_fps_list.append(
                    self._compute_fp(route.final_product_smi))
                template_idx_list.append(-1)  # placeholder
            else:  # truncate
                pos_fps_list.append(
                    self._compute_fp(route.final_product_smi))
                template_idx_list.append(-1)

        pos_fps_t = torch.tensor(
            np.array(pos_fps_list), dtype=torch.float32, device=self.device)
        tmpl_ids_t = torch.tensor(
            template_idx_list, dtype=torch.long, device=self.device)
        safe_tmpl_ids = tmpl_ids_t.clamp(min=0)
        tmpl_embs_t = self.template_embs(safe_tmpl_ids)
        tmpl_embs_t = tmpl_embs_t.masked_fill(
            (tmpl_ids_t < 0).unsqueeze(-1), 0.0)

        with torch.no_grad():
            q_bb_all = self.q_bb(
                route_states, pos_fps_t, tmpl_embs_t, l2_masks)
        timings['q_bb'] = time.perf_counter() - t0

        # --- Phase 4: Top-k selection + cascade validation (swap only) ---
        t0 = time.perf_counter()
        topk_per_route = []
        for i in range(n):
            if action_types[i] != 'swap':
                topk_per_route.append([])
                continue
            n_l2 = l2_masks[i].sum().item()
            if n_l2 == 0:
                topk_per_route.append([])
                continue
            k = min(cascade_top_k, n_l2)
            _, topk_idx = q_bb_all[i].topk(k)
            topk_per_route.append(topk_idx.cpu().tolist())

        valid_blocks_list = self._cascade_validate_batch(
            routes, positions, topk_per_route)
        timings['cascade'] = time.perf_counter() - t0

        # --- Phase 5: Select BB / handle extend / truncate ---
        t0 = time.perf_counter()
        q_bb_cpu = q_bb_all.cpu()

        # Collect extend route indices and batch-process them
        extend_indices = [i for i in range(n) if action_types[i] == 'extend']
        extend_results = {}
        if extend_indices:
            extend_results = self._handle_extend_batch(
                extend_indices, routes, route_states, explore)

        block_indices = []
        extend_bi_rxn_indices = []
        for i in range(n):
            atype = action_types[i]

            if atype == 'truncate':
                block_indices.append(-1)
                extend_bi_rxn_indices.append(-1)
            elif atype == 'extend':
                bi_rxn_idx, blk_idx = extend_results.get(i, (-1, -1))
                block_indices.append(blk_idx)
                extend_bi_rxn_indices.append(bi_rxn_idx)
            else:  # swap
                vb = valid_blocks_list[i]
                if not vb:
                    block_indices.append(
                        routes[i].steps[positions[i]].block_idx)
                elif explore[i]:
                    block_indices.append(random.choice(vb))
                else:
                    best_idx = max(
                        vb, key=lambda idx: q_bb_cpu[i, idx].item())
                    block_indices.append(best_idx)
                extend_bi_rxn_indices.append(-1)
        timings['extend'] = time.perf_counter() - t0

        return {
            'positions': positions,
            'block_indices': block_indices,
            'valid_blocks_list': valid_blocks_list,
            'action_types': action_types,
            'extend_bi_rxn_indices': extend_bi_rxn_indices,
            'route_states': route_states.cpu().numpy(),
            'position_masks': pos_masks_np,
            'pos_fps': pos_fps_list,
            'template_indices': template_idx_list,
            'timings': timings,
        }

    def _handle_extend_batch(
        self, extend_indices: list[int],
        routes: list[SynthesisRoute],
        route_states: torch.Tensor,
        explore: np.ndarray,
    ) -> dict[int, tuple[int, int]]:
        """Batch extend handling for all routes choosing 'extend'.

        Optimized over per-route _handle_extend():
        1. Thread-parallel HasSubstructMatch (RDKit C++ releases GIL)
        2. Dedup: same final_product scanned only once
        3. Single batched Q_bb GPU call for all (route, reaction) pairs

        Args:
            extend_indices: Global indices of routes choosing 'extend'.
            routes: All routes in the batch.
            route_states: (n, route_emb_dim) on GPU.
            explore: (n,) bool array, True = epsilon-greedy exploration.

        Returns:
            dict mapping global_route_idx -> (bi_rxn_idx, block_idx).
        """
        bi_reactions = self.tp.bi_reactions
        bi_compat = self.tp.bi_compat
        n_bi = len(bi_reactions)

        # --- Step 1: HasSubstructMatch (mp.Pool or sequential, with caching) ---
        smi_list = [routes[i].final_product_smi for i in extend_indices]
        smi_to_applicable = {}
        uncached_smis = []
        for smi in set(smi_list):
            if smi in self._applicable_cache:
                smi_to_applicable[smi] = self._applicable_cache[smi]
            else:
                uncached_smis.append(smi)

        if uncached_smis:
            if self._mp_pool is not None and len(uncached_smis) > 1:
                # mp.Pool: true multiprocessing (GIL not released by RDKit)
                results_list = self._mp_pool.map(
                    _mp_scan_applicable, uncached_smis)
                for smi, applicable in results_list:
                    smi_to_applicable[smi] = applicable
                    self._applicable_cache[smi] = applicable
            else:
                # Sequential fallback
                for smi in uncached_smis:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        applicable = []
                    else:
                        applicable = []
                        for idx in range(n_bi):
                            if bi_reactions[idx].is_mol_reactant(mol):
                                l2 = bi_compat.get(idx)
                                if l2 is not None and len(l2) > 0:
                                    applicable.append((idx, l2))
                    smi_to_applicable[smi] = applicable
                    self._applicable_cache[smi] = applicable

        # --- Step 2: Handle exploration + collect scoring tasks ---
        results = {}  # global_idx -> (bi_rxn_idx, blk_idx)
        scoring_tasks = []  # (global_i, applicable_rxns)

        for global_i, smi in zip(extend_indices, smi_list):
            applicables = smi_to_applicable[smi]
            if not applicables:
                results[global_i] = (-1, -1)
                continue
            if explore[global_i]:
                rxn_idx, l2 = random.choice(applicables)
                blk_idx = int(random.choice(l2))
                results[global_i] = (rxn_idx, blk_idx)
            else:
                scoring_tasks.append((global_i, applicables))

        if not scoring_tasks:
            for gi in extend_indices:
                if gi not in results:
                    results[gi] = (-1, -1)
            return results

        # --- Step 3: Batch Q_bb scoring ---
        all_route_states = []
        all_final_fps = []
        all_tmpl_ids = []
        all_l2_arrays = []
        scoring_map = []  # (global_i, rxn_idx) per batch entry

        fp_cache = {}
        for global_i, applicables in scoring_tasks:
            smi = routes[global_i].final_product_smi
            if smi not in fp_cache:
                fp_cache[smi] = self._compute_fp_tensor(smi)
            final_fp = fp_cache[smi]
            rs = route_states[global_i]

            for rxn_idx, l2 in applicables:
                all_route_states.append(rs)
                all_final_fps.append(final_fp)
                all_tmpl_ids.append(max(bi_reactions[rxn_idx].index, 0))
                all_l2_arrays.append(l2)
                scoring_map.append((global_i, rxn_idx))

        total_pairs = len(scoring_map)
        if total_pairs == 0:
            for gi in extend_indices:
                if gi not in results:
                    results[gi] = (-1, -1)
            return results

        # Build batched tensors
        batch_rs = torch.stack(all_route_states)
        batch_fp = torch.stack(all_final_fps)
        batch_te = self.template_embs(
            torch.tensor(all_tmpl_ids, dtype=torch.long, device=self.device))

        # Build compat masks via scatter (vectorized, avoids Python loop)
        row_indices = []
        col_indices = []
        for j, l2 in enumerate(all_l2_arrays):
            n_l2 = len(l2)
            row_indices.append(np.full(n_l2, j, dtype=np.int64))
            col_indices.append(np.asarray(l2, dtype=np.int64))
        row_cat = np.concatenate(row_indices)
        col_cat = np.concatenate(col_indices)
        batch_cm = torch.zeros(
            total_pairs, self.n_blocks, dtype=torch.bool, device=self.device)
        batch_cm[row_cat, col_cat] = True

        # Chunked Q_bb inference + vectorized post-processing
        MAX_BATCH = 4096
        # Pre-build route group indices for vectorized argmax
        scoring_global_ids = torch.tensor(
            [gi for gi, _ in scoring_map], dtype=torch.long)
        scoring_rxn_ids = [rxn for _, rxn in scoring_map]
        best_per_route = {}  # global_i -> (score, rxn_idx, blk_idx)

        for cs in range(0, total_pairs, MAX_BATCH):
            ce = min(cs + MAX_BATCH, total_pairs)
            with torch.no_grad():
                q_chunk = self.q_bb(
                    batch_rs[cs:ce], batch_fp[cs:ce],
                    batch_te[cs:ce], batch_cm[cs:ce])

            # Vectorized: get max score and argmax per row on GPU
            max_scores, max_blk_ids = q_chunk.max(dim=1)
            max_scores_cpu = max_scores.cpu()
            max_blk_cpu = max_blk_ids.cpu()

            for j in range(ce - cs):
                global_i = scoring_global_ids[cs + j].item()
                score = max_scores_cpu[j].item()
                if (global_i not in best_per_route
                        or score > best_per_route[global_i][0]):
                    best_per_route[global_i] = (
                        score, scoring_rxn_ids[cs + j], max_blk_cpu[j].item())

        for global_i, (_, rxn_idx, blk_idx) in best_per_route.items():
            results[global_i] = (rxn_idx, blk_idx)

        # Fill remaining with defaults
        for gi in extend_indices:
            if gi not in results:
                results[gi] = (-1, -1)

        return results

    def _cascade_validate_batch(
        self, routes: list[SynthesisRoute], positions: list[int],
        candidates_per_route: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """Cascade validation for all routes in parallel.

        Args:
            routes: List of synthesis routes.
            positions: Position index to modify per route.
            candidates_per_route: Per-route candidate block indices to validate.
                If None, uses full L2 candidates from bi_compat (slow).

        Uses (in priority order):
        1. multiprocessing.Pool — true parallelism, per-route granularity
        2. ThreadPoolExecutor — per-candidate parallelism (GIL limited)
        3. Sequential fallback
        """
        n = len(routes)

        # Prepare per-route info: (route, position, candidates, input_smi)
        route_tasks: list[tuple | None] = []
        for i, (route, pos) in enumerate(zip(routes, positions)):
            # Skip virtual positions (EXTEND/TRUNCATE) — not swap actions
            if pos >= len(route.steps):
                route_tasks.append(None)
                continue
            step = route.steps[pos]
            if step.is_uni:
                route_tasks.append(None)
                continue

            # Use provided candidates or fall back to full L2
            if candidates_per_route is not None:
                cands = candidates_per_route[i]
            else:
                l2 = self.tp.bi_compat.get(step.bi_rxn_idx)
                cands = l2.tolist() if l2 is not None and len(l2) > 0 else []

            if not cands:
                route_tasks.append(None)
                continue

            if pos == 0:
                input_smi = route.init_mol_smi
            else:
                input_smi = route.steps[pos - 1].intermediate_smi
            route_tasks.append((route, pos, cands, input_smi))

        # --- Method 1: multiprocessing Pool (best: bypasses GIL) ---
        if self._mp_pool is not None:
            mp_tasks = []
            task_indices = []
            for i, task in enumerate(route_tasks):
                if task is not None:
                    mp_tasks.append(task)
                    task_indices.append(i)

            results: list[list[int]] = [[] for _ in range(n)]
            if mp_tasks:
                mp_results = self._mp_pool.map(
                    _mp_validate_route, mp_tasks)
                for idx, valid_blocks in zip(task_indices, mp_results):
                    results[idx] = valid_blocks
            return results

        # --- Method 2: ThreadPoolExecutor (GIL limited) ---
        if self._cascade_executor is not None:
            all_futures: list[tuple[int, 'Future']] = []
            for i, task in enumerate(route_tasks):
                if task is None:
                    continue
                _route, _pos, l2_list, input_smi = task
                input_mol = Chem.MolFromSmiles(input_smi)
                if input_mol is None:
                    continue
                for blk_idx in l2_list:
                    f = self._cascade_executor.submit(
                        _validate_one_candidate_fast, blk_idx,
                        routes[i], positions[i], input_mol, self.tp)
                    all_futures.append((i, f))

            results = [[] for _ in range(n)]
            for route_idx, f in all_futures:
                r = f.result()
                if r is not None:
                    results[route_idx].append(r)
            return results

        # --- Method 3: Sequential fallback ---
        results = [[] for _ in range(n)]
        for i, task in enumerate(route_tasks):
            if task is None:
                continue
            _route, _pos, l2_list, input_smi = task
            input_mol = Chem.MolFromSmiles(input_smi)
            if input_mol is None:
                continue
            for blk_idx in l2_list:
                if _validate_one_candidate_fast(
                        blk_idx, routes[i], positions[i],
                        input_mol, self.tp) is not None:
                    results[i].append(blk_idx)
        return results

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def store_transition(self, transition: dict):
        """Store a transition in replay buffer."""
        if self.enable_exp:
            self.replay_buffer.add(transition)
        else:
            if len(self.replay_buffer_list) < self.replay_size:
                self.replay_buffer_list.append(transition)
            else:
                self.replay_buffer_list[self.replay_pos] = transition
            self.replay_pos = (self.replay_pos + 1) % self.replay_size

    @property
    def replay_len(self) -> int:
        if self.enable_exp:
            return len(self.replay_buffer)
        return len(self.replay_buffer_list)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch_size: int = 32) -> dict:
        """One DQN gradient step for both Q_pos and Q_bb.

        When enable_exp=True, applies:
        - Prioritized Experience Replay (PER) with IS weights
        - Double DQN (online selects action, target evaluates)
        - Action subsampling IS correction in Q_bb targets

        Transitions contain:
            route_state: (route_emb_dim,) numpy
            position: int
            position_mask: (max_route_len,) bool numpy
            block_idx: int
            block_mask: (n_blocks,) bool numpy (L2+L3 validated)
            position_fp: (fp_dim,) numpy
            template_idx: int
            reward: float
            next_route_state: (route_emb_dim,) numpy
            next_position_mask: (max_route_len,) bool numpy
            done: bool
        """
        if self.replay_len < batch_size:
            return {}

        # --- Sample from replay buffer ---
        tree_indices = None
        is_weights_t = None

        if self.enable_exp:
            batch, tree_indices, is_weights = self.replay_buffer.sample(
                batch_size)
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device)
        else:
            batch = random.sample(self.replay_buffer_list, batch_size)

        # Collate (with padding for backward-compat position masks)
        route_states = torch.tensor(
            np.array([t['route_state'] for t in batch]),
            dtype=torch.float32, device=self.device)
        positions = torch.tensor(
            [t['position'] for t in batch],
            dtype=torch.long, device=self.device)

        # Pad position_mask / next_position_mask to n_positions if needed
        def _pad_mask(mask_arr):
            if len(mask_arr) >= self.n_positions:
                return mask_arr[:self.n_positions]
            padded = np.zeros(self.n_positions, dtype=bool)
            padded[:len(mask_arr)] = mask_arr
            return padded

        position_masks = torch.tensor(
            np.array([_pad_mask(t['position_mask']) for t in batch]),
            dtype=torch.bool, device=self.device)
        block_ids = torch.tensor(
            [t['block_idx'] for t in batch],
            dtype=torch.long, device=self.device)
        position_fps = torch.tensor(
            np.array([t['position_fp'] for t in batch]),
            dtype=torch.float32, device=self.device)
        template_ids = torch.tensor(
            [t['template_idx'] for t in batch],
            dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [t['reward'] for t in batch],
            dtype=torch.float32, device=self.device)
        next_route_states = torch.tensor(
            np.array([t['next_route_state'] for t in batch]),
            dtype=torch.float32, device=self.device)
        next_position_masks = torch.tensor(
            np.array([_pad_mask(t['next_position_mask']) for t in batch]),
            dtype=torch.bool, device=self.device)
        dones = torch.tensor(
            [float(t['done']) for t in batch],
            dtype=torch.float32, device=self.device)

        # --- Q_pos loss ---
        q_pos_all = self.q_pos(route_states, position_masks)
        q_pos_current = q_pos_all.gather(
            1, positions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            has_valid_pos = next_position_masks.any(dim=1)

            if self.enable_exp:
                # Double DQN: online network selects action, target evaluates
                q_pos_next_online = self.q_pos(
                    next_route_states, next_position_masks)
                best_actions = q_pos_next_online.argmax(dim=1, keepdim=True)
                q_pos_next_target = self.q_pos_target(
                    next_route_states, next_position_masks)
                q_pos_next_max = q_pos_next_target.gather(
                    1, best_actions).squeeze(1)
            else:
                q_pos_next = self.q_pos_target(
                    next_route_states, next_position_masks)
                q_pos_next_max = q_pos_next.max(dim=1).values

            q_pos_next_max = q_pos_next_max.masked_fill(~has_valid_pos, 0.0)
            q_pos_target_vals = (
                rewards + (1 - dones) * self.gamma * q_pos_next_max)

        td_errors_pos = (q_pos_current - q_pos_target_vals).detach()

        if self.enable_exp:
            # PER: weight loss by importance sampling weights
            loss_pos = (is_weights_t
                        * F.smooth_l1_loss(q_pos_current, q_pos_target_vals,
                                           reduction='none')).mean()
        else:
            loss_pos = F.smooth_l1_loss(q_pos_current, q_pos_target_vals)

        # --- Q_bb loss ---
        # Truncate actions have block_idx=-1; exclude from Q_bb loss
        valid_bb = (block_ids >= 0)
        safe_tmpl = template_ids.clamp(min=0)
        template_embs = self.template_embs(safe_tmpl)
        template_embs = template_embs.masked_fill(
            (template_ids < 0).unsqueeze(-1), 0.0)
        safe_block_ids = block_ids.clamp(min=0)
        block_fps_selected = self.block_fps[safe_block_ids]

        q_bb_current = self.q_bb.score_single_block(
            route_states, position_fps, template_embs.detach(),
            block_fps_selected)

        with torch.no_grad():
            q_bb_target_vals = (
                rewards + (1 - dones) * self.gamma * q_pos_next_max)

        if valid_bb.any():
            if self.enable_exp:
                elementwise = F.smooth_l1_loss(
                    q_bb_current[valid_bb], q_bb_target_vals[valid_bb],
                    reduction='none')
                loss_bb = (is_weights_t[valid_bb] * elementwise).mean()
            else:
                loss_bb = F.smooth_l1_loss(
                    q_bb_current[valid_bb], q_bb_target_vals[valid_bb])
        else:
            loss_bb = torch.tensor(0.0, device=self.device)

        # Combined loss
        loss = loss_pos + loss_bb
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._all_params, max_norm=10.0)
        self.optimizer.step()

        # PER: update priorities with TD-errors
        if self.enable_exp and tree_indices is not None:
            td_errors = td_errors_pos.abs().cpu().numpy()
            self.replay_buffer.update_priorities(tree_indices, td_errors)

        # Soft update targets
        with torch.no_grad():
            for p, pt in zip(self.q_pos.parameters(),
                             self.q_pos_target.parameters()):
                pt.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)
            for p, pt in zip(self.q_bb.parameters(),
                             self.q_bb_target.parameters()):
                pt.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)
            for p, pt in zip(self.route_encoder.parameters(),
                             self.route_encoder_target.parameters()):
                pt.data.mul_(self.polyak).add_((1 - self.polyak) * p.data)

        self._train_steps += 1

        # Refresh block embeddings periodically
        if self._train_steps % self.block_emb_refresh_freq == 0:
            self.q_bb.precompute_block_embeddings(self.block_fps)
            self.q_bb_target.precompute_block_embeddings(self.block_fps)

        return {
            'loss_pos': loss_pos.item(),
            'loss_bb': loss_bb.item(),
            'q_pos_mean': q_pos_current.mean().item(),
            'q_bb_mean': q_bb_current.mean().item(),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save all model weights and optimizer state."""
        torch.save({
            'template_embs': self.template_embs.state_dict(),
            'block_fp_proj': self._block_fp_proj.state_dict(),
            'route_encoder': self.route_encoder.state_dict(),
            'route_encoder_target': self.route_encoder_target.state_dict(),
            'q_pos': self.q_pos.state_dict(),
            'q_pos_target': self.q_pos_target.state_dict(),
            'q_bb': self.q_bb.state_dict(),
            'q_bb_target': self.q_bb_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self._train_steps,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model weights from checkpoint.

        Handles backward compatibility: old checkpoints with Q_pos output dim
        = max_route_len are zero-padded to n_positions (max_route_len + 2).
        """
        ckpt = torch.load(path, map_location=self.device)

        # --- Q_pos backward compat: expand output layer if needed ---
        for key_prefix in ('q_pos', 'q_pos_target'):
            w_key = f'scorer.2.weight'
            b_key = f'scorer.2.bias'
            sd = ckpt[key_prefix]
            if w_key in sd and sd[w_key].shape[0] != self.n_positions:
                old_n = sd[w_key].shape[0]
                print(f"  Expanding {key_prefix} Q_pos output: "
                      f"{old_n} -> {self.n_positions}")
                new_w = torch.zeros(
                    self.n_positions, sd[w_key].shape[1],
                    dtype=sd[w_key].dtype)
                new_w[:old_n] = sd[w_key]
                sd[w_key] = new_w
                if b_key in sd:
                    new_b = torch.zeros(
                        self.n_positions, dtype=sd[b_key].dtype)
                    new_b[:old_n] = sd[b_key]
                    sd[b_key] = new_b

        self.template_embs.load_state_dict(ckpt['template_embs'])
        self._block_fp_proj.load_state_dict(ckpt['block_fp_proj'])
        self.route_encoder.load_state_dict(ckpt['route_encoder'])
        self.route_encoder_target.load_state_dict(
            ckpt['route_encoder_target'])
        self.q_pos.load_state_dict(ckpt['q_pos'])
        self.q_pos_target.load_state_dict(ckpt['q_pos_target'])
        self.q_bb.load_state_dict(ckpt['q_bb'])
        self.q_bb_target.load_state_dict(ckpt['q_bb_target'])
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except ValueError:
                print("  WARNING: Optimizer state incompatible (dim change), "
                      "reinitializing optimizer")
        self._train_steps = ckpt.get('train_steps', 0)
        # Refresh block embeddings
        self.q_bb.precompute_block_embeddings(self.block_fps)
        self.q_bb_target.precompute_block_embeddings(self.block_fps)
        print(f"  Loaded checkpoint ({self._train_steps} train steps)")

    def close(self):
        """Shut down thread pool."""
        if self._cascade_executor is not None:
            self._cascade_executor.shutdown(wait=False)
            self._cascade_executor = None
