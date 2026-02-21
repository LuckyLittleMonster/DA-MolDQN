"""Profile FastSampler with KV cache — timing breakdown per phase.

Usage:
    cd /shared/data1/Users/l1062811/git/DA-MolDQN/refs/ReaSyn
    conda run -n reasyn --live-stream python ../../scripts/profile_fast_sampler.py
"""

import sys
import pathlib
import pickle
import time

import torch
from omegaconf import OmegaConf

REASYN_ROOT = pathlib.Path(__file__).resolve().parent.parent / "refs" / "ReaSyn"
sys.path.insert(0, str(REASYN_ROOT))

from rl.reasyn.chem.fpindex import FingerprintIndex
from rl.reasyn.chem.matrix import ReactantReactionMatrix
from rl.reasyn.chem.mol import Molecule
from rl.reasyn.models.reasyn import ReaSyn
from rl.reasyn.sampler.sampler_fast import FastSampler
from rl.reasyn.utils.sample_utils import TimeLimit


def load_models(model_dir, device="cuda"):
    ar_path = model_dir / "nv-reasyn-ar-166m-v2.ckpt"
    eb_path = model_dir / "nv-reasyn-eb-174m-v2.ckpt"
    models = []
    config = None
    for ckpt_path in [ar_path, eb_path]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = ReaSyn(config.model).to(device)
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        model.eval()
        models.append(model)
    fpindex = pickle.load(open(config.chem.fpindex, "rb"))
    rxn_matrix = pickle.load(open(config.chem.rxn_matrix, "rb"))
    return models, fpindex, rxn_matrix


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exhaustiveness", type=int, default=64)
    parser.add_argument("--search_width", type=int, default=8)
    parser.add_argument("--mol_idx", type=int, default=0)
    parser.add_argument("--max_evolve_steps", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_branch", type=int, default=0)
    parser.add_argument("--skip_ef", action="store_true")
    parser.add_argument("--rxn_plim", type=int, default=0, help="P3: rxn_product_limit")
    parser.add_argument("--compile", action="store_true", help="P3: torch.compile decoder")
    args = parser.parse_args()

    model_dir = REASYN_ROOT / "data" / "trained_model"
    zinc_path = REASYN_ROOT / "data" / "zinc_first64.txt"

    print("Loading models...")
    models, fpindex, rxn_matrix = load_models(model_dir)

    with open(zinc_path) as f:
        lines = f.read().strip().split("\n")
    if lines[0].upper() == "SMILES":
        lines = lines[1:]
    mol = Molecule(lines[args.mol_idx].strip())
    print(f"Molecule: {mol.csmiles}")
    print(f"exhaustiveness={args.exhaustiveness}, search_width={args.search_width}")

    # Monkey-patch _evolve_ar_singlestep to add timing
    original_evolve = FastSampler._evolve_ar_singlestep

    phase_times = {
        "featurize": [],
        "first_token": [],
        "bb_gen": [],
        "fpindex": [],
        "rxn": [],
        "branching": [],
        "total": [],
    }

    def timed_evolve(self, gpu_lock=None, time_limit=None, sampling_direction='bu'):
        from rl.reasyn.chem.featurize import TokenType, decode_smiles
        from rl.reasyn.data.common import featurize_stack
        from rl.reasyn.utils.sample_utils import get_reactions

        if len(self._active) == 0:
            return

        N = len(self._active)
        t0 = time.perf_counter()

        # Phase 1: Featurize
        feat_list = [
            featurize_stack(s.stack, end_token=False, sampling_direction=sampling_direction)
            for s in self._active
        ]
        token_list = []
        for feat in feat_list:
            tokens = feat['tokens'].to(self.device)
            if sampling_direction == 'bu' and len(tokens) == 1:
                tokens = torch.cat([tokens, torch.tensor([TokenType.MOL_START], device=self.device)])
            token_list.append(tokens)
        t1 = time.perf_counter()
        phase_times["featurize"].append(t1 - t0)

        # Phase 2a: First token
        code, code_padding_mask = self.code
        first_tokens, all_logits, lengths = self._predict_first_token_batch(
            code, code_padding_mask, token_list, sampling_direction=sampling_direction
        )
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        phase_times["first_token"].append(t2 - t1)

        # Classify
        first_tokens_cpu = first_tokens.cpu().tolist()
        bb_indices, rxn_indices, end_indices, abort_indices = [], [], [], []
        for i, ft in enumerate(first_tokens_cpu):
            if lengths[i] > self.model.max_len:
                abort_indices.append(i)
            elif ft == TokenType.MOL_START or token_list[i][-1].item() == TokenType.MOL_START:
                bb_indices.append(i)
            elif ft >= TokenType.RXN_MIN:
                rxn_indices.append(i)
            elif ft == TokenType.END or ft == TokenType.MOL_END:
                end_indices.append(i)
            else:
                end_indices.append(i)

        # Phase 2b: BB generation
        max_tok_len = max(lengths)
        padded_tokens = torch.zeros(len(token_list), max_tok_len, dtype=torch.long, device=self.device)
        for i, t in enumerate(token_list):
            padded_tokens[i, :lengths[i]] = t

        bb_token_lists = self._generate_bb_batch(
            code, code_padding_mask, bb_indices, padded_tokens, lengths, first_tokens
        )
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        phase_times["bb_gen"].append(t3 - t2)

        # Phase 2c: fpindex
        bb_results = {}
        if bb_indices:
            smiles_list = [decode_smiles(bb_token_lists[i]) for i in range(len(bb_indices))]
            all_reactants = self._batch_get_reactants(smiles_list)
            for local_i, state_i in enumerate(bb_indices):
                reactants = all_reactants[local_i]
                if reactants is None:
                    abort_indices.append(state_i)
                else:
                    bb_results[state_i] = reactants
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        phase_times["fpindex"].append(t4 - t3)

        # Phase 2d: RXN
        rxn_results = {}
        for state_i in rxn_indices:
            reaction_logits = all_logits[state_i, TokenType.RXN_MIN:TokenType.RXN_MAX + 1]
            rxn_results[state_i] = get_reactions(reaction_logits, rxn_matrix=self._rxn_matrix)
        t5 = time.perf_counter()
        phase_times["rxn"].append(t5 - t4)

        if gpu_lock is not None:
            gpu_lock.release()

        # Phase 3: Branching — use the actual FastSampler branching code path
        # (supports P3 product_limit and parallel_branching)
        import copy
        from concurrent.futures import ThreadPoolExecutor

        state_results = {}
        for i in end_indices:
            state_results[i] = ('END', None)
        for i in abort_indices:
            state_results[i] = ('ABORTED', None)
        for i, items in bb_results.items():
            state_results[i] = ('BB', items)
        for i, items in rxn_results.items():
            state_results[i] = ('RXN', items)

        finished = []
        next_states = []
        num_branches = len(self._active) * self._factor
        product_limit = self._rxn_product_limit
        use_threads = num_branches > 32

        if use_threads:
            with ThreadPoolExecutor(max_workers=min(32, num_branches)) as pool:
                futures = []
                for state_i, base_state in enumerate(self._active):
                    if state_i not in state_results:
                        continue
                    sampled_type, sampled_item = state_results[state_i]
                    if sampled_type == 'RXN' and sampling_direction != 'td':
                        self._process_rxn_branches_bu(
                            base_state, sampled_item,
                            next_states, finished, self._aborted,
                        )
                        continue
                    for branch_i in range(self._factor):
                        if sampling_direction == 'td':
                            futures.append(pool.submit(
                                self._process_branch_td,
                                base_state, sampled_type, sampled_item, branch_i,
                            ))
                        else:
                            futures.append(pool.submit(
                                self._process_branch_bu,
                                base_state, sampled_type, sampled_item, branch_i,
                                self._mols_to_filter, self._filter_sim,
                                product_limit,
                            ))
                for f in futures:
                    action, ns, fs = f.result()
                    if action == 'next' and ns is not None:
                        next_states.append(ns)
                    elif action == 'finished' and fs is not None:
                        finished.append(fs)
                    elif action == 'both' and ns is not None:
                        next_states.append(ns)
                        finished.append(ns)
        else:
            for state_i, base_state in enumerate(self._active):
                if state_i not in state_results:
                    continue
                sampled_type, sampled_item = state_results[state_i]
                if sampled_type == 'RXN' and sampling_direction != 'td':
                    self._process_rxn_branches_bu(
                        base_state, sampled_item,
                        next_states, finished, self._aborted,
                    )
                    continue
                for branch_i in range(self._factor):
                    if sampling_direction == 'td':
                        action, ns, fs = self._process_branch_td(
                            base_state, sampled_type, sampled_item, branch_i,
                        )
                    else:
                        action, ns, fs = self._process_branch_bu(
                            base_state, sampled_type, sampled_item, branch_i,
                            self._mols_to_filter, self._filter_sim,
                            product_limit,
                        )
                    if action == 'next' and ns is not None:
                        next_states.append(ns)
                    elif action == 'finished' and fs is not None:
                        finished.append(fs)
                    elif action == 'both' and ns is not None:
                        next_states.append(ns)
                        finished.append(ns)

        del self._active
        self._active = next_states
        self._sort_states()
        if sampling_direction == 'td':
            for state in finished:
                state.stack.final_seq_topdown()
        self._add_finished_states(finished)

        t6 = time.perf_counter()
        phase_times["branching"].append(t6 - t5)
        phase_times["total"].append(t6 - t0)

        print(f"  Step N={N:3d} | feat {t1-t0:.3f}s | first_tok {t2-t1:.3f}s | "
              f"bb_gen {t3-t2:.3f}s ({len(bb_indices)} bb) | "
              f"fpidx {t4-t3:.3f}s | rxn {t5-t4:.3f}s ({len(rxn_indices)} rxn) | "
              f"branch {t6-t5:.3f}s | total {t6-t0:.3f}s")

    # Patch
    FastSampler._evolve_ar_singlestep = timed_evolve

    # Parse P2/P3 flags
    p2_kwargs = {}
    if getattr(args, 'fp16', False):
        p2_kwargs['use_fp16'] = True
    if getattr(args, 'max_branch', 0) > 0:
        p2_kwargs['max_branch_states'] = args.max_branch
    if getattr(args, 'skip_ef', False):
        p2_kwargs['skip_editflow'] = True
    if getattr(args, 'rxn_plim', 0) > 0:
        p2_kwargs['rxn_product_limit'] = args.rxn_plim
    if getattr(args, 'compile', False):
        p2_kwargs['use_compile'] = True

    # Run
    torch.cuda.reset_peak_memory_stats()
    sampler = FastSampler(
        fpindex=fpindex, rxn_matrix=rxn_matrix, mol=mol, model=models,
        factor=args.search_width, max_active_states=args.exhaustiveness,
        **p2_kwargs,
    )
    tl = TimeLimit(120)
    t_start = time.perf_counter()
    sampler.evolve(
        gpu_lock=None, time_limit=tl,
        max_evolve_steps=args.max_evolve_steps,
        num_cycles=1, num_editflow_samples=10, num_editflow_steps=50,
    )
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    df = sampler.get_dataframe()
    print(f"\n{'='*60}")
    print(f"Total: {t_total:.1f}s | {peak_mem:.0f} MB | {len(df)} results")
    if len(df) > 0:
        print(f"Max score: {df['score'].max():.3f} | Mean score: {df['score'].mean():.3f}")

    # Phase breakdown
    print(f"\nPhase breakdown (sum across steps):")
    for phase, times in phase_times.items():
        total = sum(times)
        pct = 100 * total / sum(phase_times["total"]) if sum(phase_times["total"]) > 0 else 0
        print(f"  {phase:>12s}: {total:6.2f}s ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
