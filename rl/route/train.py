"""Route-DQN training — RouteTrainer(RLTrainer).

Handles template-based route generation, environment setup, hierarchical
DQN training (Q_pos + Q_bb), and cascade validation with multiprocessing.
"""

import json
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import QED as QEDModule

from rl.trainer import RLTrainer
from training_utils import save_pickle, load_molecules
from reward import compute_reward, make_dock_scorer

from .route import SynthesisRoute
from .route_environment import RouteEnvironment
from .route_dqn import RouteDQN
from .retro_decompose import (
    build_random_routes, decompose_molecule, build_retro_routes,
    build_routes_from_aizynth, build_routes_from_paroutes,
)

# 3 levels up: rl/route/train.py -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


# ── Serialization helpers ─────────────────────────────────────────────

def _route_to_dict(route, episode, route_idx, sa=None, dock=None, reward=None):
    """Serialize a SynthesisRoute to a picklable dict."""
    mol = Chem.MolFromSmiles(route.final_product_smi)
    if mol is None:
        return None
    qed_val = QEDModule.qed(mol)
    return {
        'smiles': route.final_product_smi,
        'qed': qed_val,
        'sa': sa,
        'dock': dock,
        'reward': reward,
        'episode': episode,
        'route_idx': route_idx,
        'init_mol': route.init_mol_smi,
        'n_steps': len(route),
        'steps': [
            {
                'template_idx': s.template_idx,
                'bi_rxn_idx': s.bi_rxn_idx,
                'block_idx': s.block_idx,
                'block_smi': s.block_smi,
                'intermediate_smi': s.intermediate_smi,
                'is_uni': s.is_uni,
            }
            for s in route.steps
        ],
    }


def _make_step_record(route_idx, rl_step, position, action_type,
                      old_block_smi, new_block_smi,
                      product_smi, qed, sa, dock, reward, success):
    """Record one RL step for trajectory tracking."""
    return {
        'route_idx': route_idx,
        'rl_step': rl_step,
        'position': position,
        'action_type': action_type,
        'old_block_smi': old_block_smi,
        'new_block_smi': new_block_smi,
        'product_smi': product_smi,
        'qed': qed, 'sa': sa, 'dock': dock,
        'reward': reward, 'success': success,
    }


# ── RouteTrainer ──────────────────────────────────────────────────────

class RouteTrainer(RLTrainer):
    """Route-DQN trainer using template-based synthesis routes."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tp = None
        self.mp_pool = None
        self.env = None
        self.dqn = None
        self.dock_scorer = None
        # Tracking lists (populated during training)
        self.all_rewards = []
        self.all_qeds = []
        self.all_sas = []
        self.all_docks = []
        self.best_products = []
        self.route_top5 = {}
        self.last_episodes_buf = []
        self.phase_times = {
            'encode': 0.0, 'q_pos': 0.0, 'cascade': 0.0, 'q_bb': 0.0,
            'extend': 0.0,
            'env_step': 0.0, 'next_encode': 0.0, 'store': 0.0, 'train': 0.0,
        }
        self.timing_steps = 0
        self.oracle_counts = []  # Track oracle consumption per episode

    # ── Setup ─────────────────────────────────────────────────────

    def setup(self):
        cfg = self.cfg

        print(f"\n{'='*60}")
        print(f"Route-DQN Training")
        print(f"  reward={cfg.reward.name}, episodes={cfg.episodes}, "
              f"max_steps={cfg.max_steps}")
        print(f"  decompose={cfg.method.decompose_method}, "
              f"batch_size={cfg.batch_size}")
        print(f"{'='*60}\n")

        # 1. Load template predictor (BEFORE CUDA / mp pool)
        self._load_template_predictor()

        # 2. Create multiprocessing pool (BEFORE CUDA init)
        self._create_mp_pool()

        # 3. Device (already resolved in base __init__)
        print(f"  Device: {self.device}")

        # 4. Load initial molecules + generate routes
        init_mols = self._load_init_molecules()
        all_routes = self._generate_routes(init_mols)

        if not all_routes:
            raise RuntimeError("No valid routes generated! "
                               "Try different molecules.")

        # 5. Create environment and DQN
        self._create_env_and_dqn(all_routes)

        print(f"\n{'='*60}")
        print(f"Training: {cfg.episodes} episodes, {self.env.n_routes} routes, "
              f"{cfg.max_steps} steps/ep")
        print(f"{'='*60}\n")

    def _load_template_predictor(self):
        cfg = self.cfg
        print("--- Loading template predictor ---")
        from rl.template.template_predictor import TemplateReactionPredictor

        _data_dir = PROJECT_ROOT / 'rl' / 'template' / 'data'

        if cfg.method.decompose_method == 'aizynth':
            bb_path = (cfg.method.bb_library
                       or str(_data_dir / 'building_blocks_merged.smi'))
            self.tp = TemplateReactionPredictor(
                template_path=str(_data_dir / 'templates_aizynth.txt'),
                building_block_path=bb_path,
                num_workers=0,
            )
        elif cfg.method.decompose_method == 'paroutes':
            bb_path = (cfg.method.bb_library
                       or str(_data_dir / 'building_blocks_paroutes.smi'))
            self.tp = TemplateReactionPredictor(
                template_path=str(_data_dir / 'templates_paroutes.txt'),
                building_block_path=bb_path,
                num_workers=0,
            )
        else:
            if cfg.method.bb_library:
                self.tp = TemplateReactionPredictor(
                    building_block_path=cfg.method.bb_library, num_workers=0)
            else:
                self.tp = TemplateReactionPredictor(num_workers=0)

        self.tp.load()
        self.tp._uni_by_template_idx = {r.index: r for r in self.tp.uni_reactions}

    def _create_mp_pool(self):
        cascade_workers = self.cfg.method.cascade_workers
        if cascade_workers > 0:
            from . import route as route_module
            route_module._mp_tp = self.tp
            ctx = mp.get_context('fork')
            self.mp_pool = ctx.Pool(cascade_workers)
            print(f"  MP pool: {cascade_workers} workers (fork)")

    def _load_init_molecules(self):
        cfg = self.cfg
        if cfg.method.decompose_method not in ('paroutes', 'aizynth'):
            init_mols = load_molecules(cfg, PROJECT_ROOT)
        else:
            if cfg.init_mol_path:
                with open(cfg.init_mol_path) as f:
                    init_mols = [line.strip() for line in f if line.strip()]
            else:
                init_mols = list(cfg.init_mol)

        print(f"  Init molecules ({len(init_mols)}): "
              f"{init_mols[:8]}{'...' if len(init_mols) > 8 else ''}")
        return init_mols

    def _generate_routes(self, init_mols):
        cfg = self.cfg
        print("\n--- Generating initial routes ---")
        t0 = time.perf_counter()
        all_routes = []

        if cfg.method.decompose_method == 'aizynth':
            aizynth_path = cfg.method.aizynth_path
            if aizynth_path is None:
                aizynth_path = str(PROJECT_ROOT / 'Experiments' /
                                   'aizynth_zinc128_results.json')
            with open(aizynth_path) as f:
                aizynth_data = json.load(f)
            solved = [d for d in aizynth_data if d.get('is_solved', False)]
            az_offset = cfg.get('init_mol_offset', 0)
            n_mol = cfg.num_molecules or len(solved)
            solved = solved[az_offset:az_offset + n_mol]
            print(f"  AiZynth: {len(solved)} solved molecules "
                  f"(offset={az_offset})")
            all_routes = build_routes_from_aizynth(solved, self.tp)

        elif cfg.method.decompose_method == 'paroutes':
            paroutes_path = cfg.method.paroutes_path
            if paroutes_path is None:
                paroutes_path = str(PROJECT_ROOT / 'Data' / 'paroutes' /
                                    'n1_routes_matched.json')
            with open(paroutes_path) as f:
                paroutes_data = json.load(f)

            offset = cfg.get('init_mol_offset', 0)
            n_mol = cfg.num_molecules or 64
            if cfg.mol_json:
                mol_json_path = pathlib.Path(cfg.mol_json)
                if not mol_json_path.is_absolute():
                    mol_json_path = PROJECT_ROOT / mol_json_path
                with open(mol_json_path) as f:
                    target_mols = json.load(f)
                target_smiles = [d['smiles']
                                 for d in target_mols[offset:offset + n_mol]]
                route_lookup = {r['smiles']: r for r in paroutes_data}
                matched = [route_lookup[smi] for smi in target_smiles
                           if smi in route_lookup]
                skipped = n_mol - len(matched)
                paroutes_data = matched
                print(f"  PaRoutes: {len(paroutes_data)} routes matched "
                      f"(offset={offset}, skipped={skipped} without routes)")
            else:
                paroutes_data = paroutes_data[offset:offset + n_mol]
                print(f"  PaRoutes: {len(paroutes_data)} routes "
                      f"(offset={offset})")

            all_routes = build_routes_from_paroutes(paroutes_data, self.tp)

        else:
            for mol_smi in init_mols:
                if cfg.method.decompose_method == 'random':
                    routes = build_random_routes(
                        mol_smi, self.tp,
                        n_routes=cfg.method.routes_per_mol,
                        n_steps=cfg.method.route_steps,
                        seed=cfg.seed,
                    )
                elif cfg.method.decompose_method == 'retro':
                    routes = build_retro_routes(
                        mol_smi, self.tp,
                        max_depth=cfg.method.route_steps,
                        max_routes=cfg.method.routes_per_mol,
                        min_steps=2,
                        seed=cfg.seed,
                        verify=True,
                    )
                else:
                    routes = decompose_molecule(
                        mol_smi, self.tp,
                        max_depth=cfg.method.route_steps,
                        max_routes=cfg.method.routes_per_mol,
                    )
                all_routes.extend(routes)

        elapsed = time.perf_counter() - t0
        print(f"Generated {len(all_routes)} routes in {elapsed:.1f}s")
        for i, route in enumerate(all_routes[:5]):
            print(f"  Route {i}: {len(route)} steps, "
                  f"{route.n_modifiable} modifiable, "
                  f"init={route.init_mol_smi}, "
                  f"final={route.final_product_smi}")
        return all_routes

    def _create_env_and_dqn(self, all_routes):
        cfg = self.cfg
        print("\n--- Creating environment and DQN ---")

        self.dock_scorer = (make_dock_scorer(cfg.reward)
                            if cfg.reward.name != 'qed' else None)

        self.env = RouteEnvironment(
            routes=all_routes,
            tp=self.tp,
            max_steps=cfg.max_steps,
            discount=cfg.gamma,
            cfg_reward=cfg.reward,
            dock_scorer=self.dock_scorer,
        )

        self.dqn = RouteDQN(
            tp=self.tp,
            device=self.device,
            fp_dim=cfg.method.fp_dim,
            template_emb_dim=cfg.method.template_emb_dim,
            block_emb_dim=cfg.method.block_emb_dim,
            route_emb_dim=cfg.method.route_emb_dim,
            max_route_len=cfg.method.max_route_len,
            hidden_dim=cfg.method.hidden_dim,
            lr=cfg.lr,
            gamma=cfg.gamma,
            polyak=cfg.polyak,
            replay_size=cfg.replay_size,
            cascade_workers=cfg.method.cascade_workers,
            mp_pool=self.mp_pool,
            enable_variable_length=cfg.method.get(
                'enable_variable_length', True),
            min_route_len=cfg.method.get('min_route_len', 2),
            enable_exp=cfg.method.get('enable_exp', False),
            subsample_k=cfg.method.get('subsample_k', 512),
        )

    # ── Checkpoint ────────────────────────────────────────────────

    def try_load_checkpoint(self):
        """Route uses RouteDQN's built-in checkpoint loading."""
        if self.cfg.load_checkpoint and os.path.exists(self.cfg.load_checkpoint):
            self.dqn.load_checkpoint(self.cfg.load_checkpoint)

    def save(self, episode, history):
        if not self.cfg.get('no_save_checkpoint', False):
            self.dqn.save(str(self.model_save_dir
                              / f'{self.prefix}_checkpoint.pth'))

        best_products_sorted = sorted(
            self.best_products,
            key=lambda x: x.get('reward', x['qed']),
            reverse=True)[:20]

        save_pickle(
            self.exp_dir / f'{self.prefix}_history.pickle',
            {
                'rewards': self.all_rewards,
                'qeds': self.all_qeds,
                'sas': self.all_sas,
                'docks': self.all_docks,
                'oracle_counts': self.oracle_counts,
                'best_products': best_products_sorted,
                'route_top5': dict(self.route_top5),
                'last_episodes': list(self.last_episodes_buf),
                'config': OmegaConf.to_container(self.cfg, resolve=True),
            })

    # ── Episode ───────────────────────────────────────────────────

    def run_episode(self, episode):
        cfg = self.cfg
        routes = self.env.reset()
        n = self.env.n_routes
        episode_rewards = [0.0] * n
        episode_qeds_final = [0.0] * n
        episode_trajectories = [[] for _ in range(n)]
        enable_exp = self.dqn.enable_exp

        # Reward shaping: track QED before each step
        if enable_exp:
            prev_qeds = []
            for route in routes:
                mol = Chem.MolFromSmiles(route.final_product_smi)
                prev_qeds.append(QEDModule.qed(mol) if mol else 0.0)

        for step in range(cfg.max_steps):
            batch_result = self.dqn.act_batch(
                routes, self.eps,
                cascade_top_k=cfg.method.get('cascade_topk', 64))

            positions = batch_result['positions']
            block_indices = batch_result['block_indices']
            route_states_np = batch_result['route_states']
            pos_masks_np = batch_result['position_masks']
            pos_fps = batch_result['pos_fps']
            template_indices = batch_result['template_indices']
            action_types = batch_result.get('action_types')

            for k, v in batch_result['timings'].items():
                self.phase_times[k] += v

            # Save old block info before env.step modifies routes
            old_block_info = []
            for i, route in enumerate(routes):
                pos = positions[i]
                atype = action_types[i] if action_types else 'swap'
                if atype == 'swap' and pos < len(route.steps):
                    s = route.steps[pos]
                    old_block_info.append(
                        s.block_smi if not s.is_uni else '')
                else:
                    old_block_info.append('')

            t0 = time.perf_counter()
            result = self.env.step(
                positions, block_indices,
                action_types=action_types,
                extend_bi_rxn_indices=batch_result.get(
                    'extend_bi_rxn_indices'))
            self.phase_times['env_step'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            next_route_states = (self.dqn.encode_routes(routes)
                                 .cpu().numpy())
            self.phase_times['next_encode'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            n_positions = getattr(self.dqn, 'n_positions',
                                  cfg.method.max_route_len)
            for i in range(len(routes)):
                next_pos_mask = np.zeros(n_positions, dtype=bool)
                for p_idx, m in enumerate(routes[i].modifiable_mask):
                    if p_idx < cfg.method.max_route_len and m:
                        next_pos_mask[p_idx] = True
                if (hasattr(self.dqn, 'enable_variable_length')
                        and self.dqn.enable_variable_length):
                    if len(routes[i]) < self.dqn.max_route_len:
                        next_pos_mask[self.dqn.EXTEND_POS] = True
                    if len(routes[i]) > self.dqn.min_route_len:
                        next_pos_mask[self.dqn.TRUNCATE_POS] = True

                # Reward shaping: F = gamma * QED(s') - QED(s)
                shaped_reward = result['rewards'][i]
                if enable_exp:
                    new_qed = result['QED'][i]
                    shaped_reward += (cfg.gamma * new_qed - prev_qeds[i])

                transition = {
                    'route_state': route_states_np[i],
                    'position': positions[i],
                    'position_mask': pos_masks_np[i],
                    'block_idx': block_indices[i],
                    'block_mask': np.zeros(1, dtype=bool),
                    'position_fp': pos_fps[i],
                    'template_idx': template_indices[i],
                    'reward': shaped_reward,
                    'next_route_state': next_route_states[i],
                    'next_position_mask': next_pos_mask,
                    'done': result['done'],
                }
                self.dqn.store_transition(transition)

                episode_rewards[i] += result['rewards'][i]
                episode_qeds_final[i] = result['QED'][i]

                # Record trajectory step
                atype = action_types[i] if action_types else 'swap'
                pos = positions[i]
                new_blk_smi = ''
                if atype == 'swap' and pos < len(routes[i].steps):
                    new_blk_smi = routes[i].steps[pos].block_smi
                elif atype == 'extend' and len(routes[i].steps) > 0:
                    new_blk_smi = routes[i].steps[-1].block_smi
                episode_trajectories[i].append(_make_step_record(
                    route_idx=i, rl_step=step,
                    position=pos, action_type=atype,
                    old_block_smi=old_block_info[i],
                    new_block_smi=new_blk_smi,
                    product_smi=routes[i].final_product_smi,
                    qed=result['QED'][i], sa=result['SA_score'][i],
                    dock=result['dock_score'][i],
                    reward=result['rewards'][i],
                    success=result['success'][i]))
            # Update prev_qeds for reward shaping
            if enable_exp:
                for i in range(len(routes)):
                    prev_qeds[i] = result['QED'][i]

            self.phase_times['store'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            n_updates = max(1, n // cfg.batch_size)
            for _ in range(n_updates):
                self.dqn.train_step(cfg.batch_size)
            self.phase_times['train'] += time.perf_counter() - t0

            self.timing_steps += 1

        # PER beta annealing (progress = episode / total_episodes)
        if enable_exp:
            progress = (episode + 1) / cfg.episodes
            self.dqn.replay_buffer.anneal_beta(progress)

        # ── Episode stats ─────────────────────────────────────────
        mean_reward = np.mean(episode_rewards)
        mean_qed = np.mean(episode_qeds_final)
        max_qed = max(episode_qeds_final)
        mean_sa = np.mean(result['SA_score'])
        ep_dock_scores = result.get('dock_score', [0.0] * n)
        mean_dock = np.mean(ep_dock_scores)
        best_dock = min(ep_dock_scores) if self.dock_scorer else 0.0

        self.all_rewards.append(mean_reward)
        self.all_qeds.append(mean_qed)
        self.all_sas.append(mean_sa)
        self.all_docks.append(mean_dock)

        # Track oracle consumption (unique molecules scored by proxy)
        oracle_count = self.dock_scorer.cache_size if self.dock_scorer else 0
        self.oracle_counts.append(oracle_count)

        # Track best products and route snapshots
        ep_route_snapshots = []
        for i, route in enumerate(routes):
            rd = _route_to_dict(route, episode, i,
                                sa=result['SA_score'][i],
                                dock=ep_dock_scores[i],
                                reward=episode_rewards[i])
            if rd is None:
                continue
            rd['trajectory'] = episode_trajectories[i]
            qed_val = rd['qed']
            self.best_products.append({
                'smiles': route.final_product_smi,
                'qed': qed_val,
                'sa': result['SA_score'][i],
                'dock': ep_dock_scores[i],
                'reward': episode_rewards[i],
                'episode': episode,
                'route_idx': i,
                'init_mol': route.init_mol_smi,
                'n_steps': len(route),
            })
            ep_route_snapshots.append(rd)

            if i not in self.route_top5:
                self.route_top5[i] = []
            top_list = self.route_top5[i]
            cmp_val = rd.get('reward', qed_val)
            if (len(top_list) < 5
                    or cmp_val > top_list[-1].get(
                        'reward', top_list[-1]['qed'])):
                existing = {d['smiles'] for d in top_list}
                if rd['smiles'] not in existing:
                    top_list.append(rd)
                    top_list.sort(
                        key=lambda x: x.get('reward', x['qed']),
                        reverse=True)
                    self.route_top5[i] = top_list[:5]

        self.last_episodes_buf.append({
            'episode': episode,
            'routes': ep_route_snapshots,
        })
        if len(self.last_episodes_buf) > 5:
            self.last_episodes_buf.pop(0)

        # Keep reference to last routes for finalize()
        self._last_routes = routes

        return {
            'reward': mean_reward,
            'qed': mean_qed,
            'max_qed': max_qed,
            'sa': mean_sa,
            'dock': mean_dock,
            'best_dock': best_dock,
            'n_success': sum(result['success']),
            'replay_size': self.dqn.replay_len,
            'routes': routes,
            'ep_dock_scores': ep_dock_scores,
            'episode_qeds_final': episode_qeds_final,
            'oracle_count': oracle_count,
        }

    # ── Logging ───────────────────────────────────────────────────

    def log_episode(self, episode, metrics):
        cfg = self.cfg
        dock_str = ""
        if self.dock_scorer:
            dock_str = (f" | Dock: {metrics['dock']:.1f} "
                        f"(best {metrics['best_dock']:.1f})")

        oracle_str = ""
        if 'oracle_count' in metrics:
            oracle_str = f" | Oracle: {metrics['oracle_count']}"

        print(f"Ep {episode+1:4d}/{cfg.episodes} | "
              f"R: {metrics['reward']:.4f} | "
              f"QED: {metrics['qed']:.3f} "
              f"(max {metrics['max_qed']:.3f}) | "
              f"SA: {metrics['sa']:.2f}{dock_str} | "
              f"Eps: {metrics['eps']:.3f} | "
              f"Success: {metrics['n_success']}/{self.env.n_routes} | "
              f"Buf: {metrics['replay_size']}{oracle_str} | "
              f"Time: {metrics['elapsed']:.1f}s")

        routes = metrics['routes']
        ep_dock_scores = metrics['ep_dock_scores']
        episode_qeds_final = metrics['episode_qeds_final']
        for i in range(min(2, self.env.n_routes)):
            route = routes[i]
            dock_info = ""
            if self.dock_scorer:
                dock_info = f", Dock={ep_dock_scores[i]:.1f}"
            print(f"  Route {i}: {route.init_mol_smi} -> "
                  f"{route.final_product_smi} "
                  f"(QED={episode_qeds_final[i]:.3f}{dock_info})")

        if self.timing_steps > 0:
            avg_ms = {k: v / self.timing_steps * 1000
                      for k, v in self.phase_times.items()}
            total_ms = sum(avg_ms.values())
            print(f"  Timing/step ({self.timing_steps} steps): "
                  f"encode={avg_ms['encode']:.1f}ms "
                  f"Q_pos={avg_ms['q_pos']:.1f}ms "
                  f"cascade={avg_ms['cascade']:.1f}ms "
                  f"Q_bb={avg_ms['q_bb']:.1f}ms "
                  f"extend={avg_ms.get('extend', 0):.1f}ms "
                  f"env={avg_ms['env_step']:.1f}ms "
                  f"next_enc={avg_ms['next_encode']:.1f}ms "
                  f"store={avg_ms['store']:.1f}ms "
                  f"train={avg_ms['train']:.1f}ms "
                  f"total={total_ms:.1f}ms")
            self.phase_times = {k: 0.0 for k in self.phase_times}
            self.timing_steps = 0

    # ── Finalize ──────────────────────────────────────────────────

    def finalize(self):
        best_reward = self.best_metric or 0.0
        print(f"\n{'='*60}")
        print(f"Training complete! Best mean reward: {best_reward:.4f}")
        print(f"{'='*60}")

        best_by_qed = sorted(
            self.best_products, key=lambda x: x['qed'], reverse=True)[:10]
        print("\nTop 10 products by QED:")
        for i, p in enumerate(best_by_qed):
            print(f"  {i+1}. QED={p['qed']:.3f} | {p['smiles']} "
                  f"(from {p['init_mol']}, ep {p['episode']})")

        routes = getattr(self, '_last_routes', None)
        if routes:
            print("\nFinal routes:")
            for i, route in enumerate(routes[:5]):
                mol = Chem.MolFromSmiles(route.final_product_smi)
                qed_val = QEDModule.qed(mol) if mol else 0
                print(f"  Route {i}: {route.init_mol_smi} -> "
                      f"{route.final_product_smi} (QED={qed_val:.3f})")
                for j, step_obj in enumerate(route.steps):
                    rxn_type = "uni" if step_obj.is_uni else "bi"
                    bb_str = (f" + {step_obj.block_smi}"
                              if step_obj.block_smi else "")
                    print(f"    Step {j}: [{rxn_type}] T{step_obj.template_idx}"
                          f"{bb_str} -> {step_obj.intermediate_smi}")

        print(f"\nPer-route top-5 products ({len(self.route_top5)} routes):")
        for ri in sorted(self.route_top5.keys())[:10]:
            tops = self.route_top5[ri]
            if tops:
                print(f"  Route {ri} ({tops[0]['init_mol']}):")
                for j, t in enumerate(tops):
                    path_str = " -> ".join(
                        s['intermediate_smi'][:40] for s in t['steps'])
                    print(f"    {j+1}. QED={t['qed']:.3f} "
                          f"ep{t['episode']:4d} | {t['smiles'][:60]}")
                    print(f"       Path: {path_str[:120]}")

        # ── Oracle consumption summary ───────────────────────────
        if self.oracle_counts:
            total_oracle = self.oracle_counts[-1]
            print(f"\n{'='*60}")
            print(f"Oracle Consumption")
            print(f"{'='*60}")
            print(f"  Total unique proxy calls: {total_oracle}")
            # Show oracle count at key episodes
            for ep_idx in [9, 24, 49, 74, 99]:
                if ep_idx < len(self.oracle_counts):
                    print(f"  After ep {ep_idx+1:3d}: {self.oracle_counts[ep_idx]} oracles")

        # ── Hypervolume computation ──────────────────────────────
        self._compute_hypervolume()

    # ── Hypervolume ─────────────────────────────────────────────

    def _compute_hypervolume(self):
        """Compute HV from all scored molecules using per-target proxy scores.

        4-obj: (GSK3B, JNK3, QED, SA_norm), ref=(0,0,0,0)
        2-obj: (GSK3B, JNK3), ref=(0,0)
        """
        if not self.dock_scorer or not hasattr(self.dock_scorer, 'get_all_scored'):
            print("\n  [HV] No multi-target scorer available, skipping HV.")
            return

        try:
            from botorch.utils.multi_objective.hypervolume import Hypervolume
            from botorch.utils.multi_objective.pareto import is_non_dominated
        except ImportError:
            print("\n  [HV] botorch not installed, skipping HV computation.")
            return

        per_target = self.dock_scorer.get_all_scored()
        if not per_target:
            print("\n  [HV] No scored molecules found.")
            return

        from reward.core import _load_sascorer
        sascorer = _load_sascorer()

        # Build 4-obj matrix: (GSK3B, JNK3, QED, SA_norm) for each scored mol
        points_4d = []
        points_2d = []
        for smi, scores in per_target.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            gsk3b = scores.get('gsk3b', 0.0)
            jnk3 = scores.get('jnk3', 0.0)
            qed = QEDModule.qed(mol)
            sa_raw = sascorer.calculateScore(mol)
            sa_norm = max(0.0, min(1.0, (10.0 - sa_raw) / 9.0))
            points_4d.append([gsk3b, jnk3, qed, sa_norm])
            points_2d.append([gsk3b, jnk3])

        if not points_4d:
            print("\n  [HV] No valid molecules for HV computation.")
            return

        pts_4d = torch.tensor(points_4d)
        pts_2d = torch.tensor(points_2d)

        # 4-obj HV
        ref_4d = torch.zeros(4)
        hv4 = Hypervolume(ref_4d)
        pareto_4d = pts_4d[is_non_dominated(pts_4d)]
        hv_4d = hv4.compute(pareto_4d) if len(pareto_4d) > 0 else 0.0

        # 2-obj HV
        ref_2d = torch.zeros(2)
        hv2 = Hypervolume(ref_2d)
        pareto_2d = pts_2d[is_non_dominated(pts_2d)]
        hv_2d = hv2.compute(pareto_2d) if len(pareto_2d) > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Hypervolume (ref=(0,...,0))")
        print(f"{'='*60}")
        print(f"  Total scored molecules: {len(points_4d)}")
        print(f"  4-obj Pareto front: {len(pareto_4d)} molecules")
        print(f"  4-obj HV: {hv_4d:.6f}")
        print(f"  2-obj Pareto front: {len(pareto_2d)} molecules")
        print(f"  2-obj HV: {hv_2d:.6f}")
        print(f"  (Baseline 4-obj: Genetic-GFN=0.642, HN-GFN=0.416)")
        print(f"  (Baseline 2-obj: Genetic-GFN=0.718, HN-GFN=0.669)")

        # Show Pareto front extremes
        if len(pareto_4d) > 0:
            print(f"\n  4-obj Pareto extremes:")
            for dim, name in enumerate(['GSK3B', 'JNK3', 'QED', 'SA']):
                best_idx = pareto_4d[:, dim].argmax()
                vals = pareto_4d[best_idx]
                print(f"    Best {name}: [{vals[0]:.3f}, {vals[1]:.3f}, "
                      f"{vals[2]:.3f}, {vals[3]:.3f}]")

    # ── Cleanup ───────────────────────────────────────────────────

    def cleanup(self):
        if self.dqn is not None:
            self.dqn.close()
            self.dqn = None
        if self.mp_pool is not None:
            self.mp_pool.close()
            self.mp_pool.join()
            self.mp_pool = None


# ── Backward-compatible entry point ──────────────────────────────────

def train_route(cfg):
    """Route-DQN training entry point."""
    RouteTrainer(cfg).train()
