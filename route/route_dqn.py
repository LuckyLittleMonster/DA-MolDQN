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

from route.route import (
    SynthesisRoute, cascade_validate, update_route,
    _validate_one_candidate_fast, _mp_validate_route,
)
from route.policy_network import (
    RouteStateEncoder,
    PositionPolicyNetwork,
    RouteBBScoringNetwork,
)


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

        # Replay buffer
        self.replay_buffer: list[dict] = []
        self.replay_pos = 0

        # Cascade validation: prefer mp_pool (true parallelism) over threads
        if mp_pool is not None:
            self._cascade_executor = None
        else:
            self._cascade_executor = (
                ThreadPoolExecutor(max_workers=cascade_workers)
                if cascade_workers > 0 else None)

        self._train_steps = 0

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
            self.max_route_len, dtype=torch.bool, device=self.device)
        for p in valid_positions:
            if p < self.max_route_len:
                position_mask[p] = True

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

        Old: encode → Q_pos → cascade ALL L2 (~23K) → Q_bb
        New: encode → Q_pos → Q_bb (L2 mask) → top-k → cascade top-k → select

        Returns dict with:
            positions: list[int]
            block_indices: list[int]
            valid_blocks_list: list[list[int]]
            route_states: np.ndarray (n, route_emb_dim)
            position_masks: np.ndarray (n, max_route_len) bool
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
        pos_masks_np = np.zeros((n, self.max_route_len), dtype=bool)
        valid_pos_lists = []
        for i, route in enumerate(routes):
            vp = [j for j, m in enumerate(route.modifiable_mask)
                  if m and j < self.max_route_len]
            valid_pos_lists.append(vp)
            for p in vp:
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

        # Ensure positions are not uni
        for i, route in enumerate(routes):
            if route.steps[positions[i]].is_uni:
                vp = valid_pos_lists[i]
                if vp:
                    positions[i] = random.choice(vp)
        timings['q_pos'] = time.perf_counter() - t0

        # --- Phase 3: Q_bb scoring with L2 mask only (cheap matmul) ---
        t0 = time.perf_counter()
        pos_fps_list = []
        template_idx_list = []
        l2_masks = torch.zeros(
            n, self.n_blocks, dtype=torch.bool, device=self.device)

        for i, route in enumerate(routes):
            pos = positions[i]
            step = route.steps[pos]

            # Position FP
            if pos == 0:
                pos_fps_list.append(self._compute_fp(route.init_mol_smi))
            else:
                pos_fps_list.append(self._compute_fp(
                    route.steps[pos - 1].intermediate_smi))
            template_idx_list.append(step.template_idx)

            # L2 template compatibility mask
            if not step.is_uni:
                l2 = self.tp.bi_compat.get(step.bi_rxn_idx)
                if l2 is not None and len(l2) > 0:
                    l2_masks[i, l2] = True

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

        # --- Phase 4: Top-k selection + cascade validation ---
        t0 = time.perf_counter()
        topk_per_route = []
        for i in range(n):
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

        # --- Phase 5: Select from cascade-validated candidates ---
        q_bb_cpu = q_bb_all.cpu()
        block_indices = []
        for i in range(n):
            vb = valid_blocks_list[i]
            if not vb:
                block_indices.append(
                    routes[i].steps[positions[i]].block_idx)
            elif explore[i]:
                block_indices.append(random.choice(vb))
            else:
                # Pick the cascade-valid BB with highest Q_bb score
                best_idx = max(vb, key=lambda idx: q_bb_cpu[i, idx].item())
                block_indices.append(best_idx)

        return {
            'positions': positions,
            'block_indices': block_indices,
            'valid_blocks_list': valid_blocks_list,
            'route_states': route_states.cpu().numpy(),
            'position_masks': pos_masks_np,
            'pos_fps': pos_fps_list,
            'template_indices': template_idx_list,
            'timings': timings,
        }

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
        if len(self.replay_buffer) < self.replay_size:
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[self.replay_pos] = transition
        self.replay_pos = (self.replay_pos + 1) % self.replay_size

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch_size: int = 32) -> dict:
        """One DQN gradient step for both Q_pos and Q_bb.

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
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = random.sample(self.replay_buffer, batch_size)

        # Collate
        route_states = torch.tensor(
            np.array([t['route_state'] for t in batch]),
            dtype=torch.float32, device=self.device)
        positions = torch.tensor(
            [t['position'] for t in batch],
            dtype=torch.long, device=self.device)
        position_masks = torch.tensor(
            np.array([t['position_mask'] for t in batch]),
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
            np.array([t['next_position_mask'] for t in batch]),
            dtype=torch.bool, device=self.device)
        dones = torch.tensor(
            [float(t['done']) for t in batch],
            dtype=torch.float32, device=self.device)

        # --- Q_pos loss ---
        q_pos_all = self.q_pos(route_states, position_masks)
        q_pos_current = q_pos_all.gather(
            1, positions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_pos_next = self.q_pos_target(
                next_route_states, next_position_masks)
            has_valid_pos = next_position_masks.any(dim=1)
            q_pos_next_max = q_pos_next.max(dim=1).values
            q_pos_next_max = q_pos_next_max.masked_fill(~has_valid_pos, 0.0)
            q_pos_target_vals = (
                rewards + (1 - dones) * self.gamma * q_pos_next_max)

        loss_pos = F.smooth_l1_loss(q_pos_current, q_pos_target_vals)

        # --- Q_bb loss ---
        # Q_bb uses current state's context (position_fp, template_emb)
        safe_tmpl = template_ids.clamp(min=0)
        template_embs = self.template_embs(safe_tmpl)
        template_embs = template_embs.masked_fill(
            (template_ids < 0).unsqueeze(-1), 0.0)
        block_fps_selected = self.block_fps[block_ids]

        q_bb_current = self.q_bb.score_single_block(
            route_states, position_fps, template_embs.detach(),
            block_fps_selected)

        with torch.no_grad():
            # Q_bb target: use Q_pos's next-state value as bootstrap.
            # This avoids needing next-state position context (which depends
            # on a future Q_pos decision). Q_bb learns: "value of choosing
            # this BB = immediate reward + future route value from Q_pos".
            q_bb_target_vals = (
                rewards + (1 - dones) * self.gamma * q_pos_next_max)

        loss_bb = F.smooth_l1_loss(q_bb_current, q_bb_target_vals)

        # Combined loss
        loss = loss_pos + loss_bb
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._all_params, max_norm=10.0)
        self.optimizer.step()

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
        """Load model weights from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
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
            self.optimizer.load_state_dict(ckpt['optimizer'])
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
