"""FastSampler: Batched GPU inference + parallel chemistry for ReaSyn.

Inherits from Sampler, only overrides _evolve_ar_singlestep() to batch
all GPU calls and parallelize CPU-bound chemistry/state operations.

Optimizations over base Sampler:
1. Batched first-token prediction (all states in one forward pass)
2. KV-cached BB SMILES generation (O(1) per step instead of O(L))
3. Batched fpindex queries (one cdist for all BB states)
4. ThreadPool for CPU-bound state branching
"""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing.synchronize import Lock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chem.featurize import TokenType, decode_smiles
from ..chem.mol import Molecule
from ..chem.stack import Stack
from ..data.common import featurize_stack
from .sampler import Sampler
from ..utils.sample_utils import (
    PredictResult, State, TimeLimit, get_reactants, get_reactions,
    _ReactantItem,
)

# Maximum batch size for a single GPU forward pass (memory safety)
_MAX_BATCH = 64


def _fast_copy_state(state: State) -> State:
    """Fast structural copy of State, sharing immutable Molecule/Reaction objects.

    copy.deepcopy is expensive because it recursively visits every object even
    though Molecule (defined by SMILES) and Reaction (defined by SMARTS) are
    immutable. This function copies only the mutable container structure
    (lists, sets) while sharing all element references.

    ~50x faster than copy.deepcopy for typical States.
    """
    new = State.__new__(State)
    s = state.stack
    ns = Stack.__new__(Stack)
    ns._mols = list(s._mols)
    ns._rxns = list(s._rxns)
    ns._tokens = list(s._tokens)
    ns._stack = [set(x) for x in s._stack]
    ns._seq_topdown = list(s._seq_topdown)
    new.stack = ns
    new.scores = list(state.scores)
    return new


class _nullcontext:
    """Minimal no-op context manager (avoids importing contextlib)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


class _KVCacheDecoder:
    """KV cache wrapper for incremental autoregressive decoding.

    Wraps a Decoder module, providing prefill() and decode_step() methods
    that cache self-attention and cross-attention K,V tensors to avoid
    reprocessing the full sequence at each generation step.

    Math equivalence: produces the same logits as full-sequence forward
    passes because causal mask + absolute PE guarantees position-T output
    depends only on positions 0..T regardless of padding beyond T.
    """

    def __init__(self, decoder_module, nhead: int, d_model: int, device: torch.device):
        self.decoder = decoder_module
        self.layers = list(decoder_module.dec.layers)
        self.final_norm = decoder_module.dec.norm  # may be None
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.device = device

        # Populated by prefill()
        self._sa_k: list[torch.Tensor] = []
        self._sa_v: list[torch.Tensor] = []
        self._xa_k: list[torch.Tensor] = []
        self._xa_v: list[torch.Tensor] = []
        self._xa_mask: torch.Tensor | None = None
        self._valid_len: list[int] = []
        self._max_cache: int = 0

    def _proj_qkv(self, mha: nn.MultiheadAttention, x: torch.Tensor):
        """Project x -> Q, K, V via packed in_proj_weight."""
        B, L, _ = x.shape
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        qkv = qkv.view(B, L, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nhead, L, head_dim)
        return qkv[0], qkv[1], qkv[2]

    def _proj_q(self, mha: nn.MultiheadAttention, x: torch.Tensor):
        """Project x -> Q only (cross-attention query)."""
        D = self.d_model
        q = F.linear(x, mha.in_proj_weight[:D], mha.in_proj_bias[:D])
        B, L, _ = q.shape
        return q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def _proj_kv(self, mha: nn.MultiheadAttention, x: torch.Tensor):
        """Project x -> K, V (cross-attention key/value from encoder)."""
        D = self.d_model
        kv = F.linear(x, mha.in_proj_weight[D:], mha.in_proj_bias[D:])
        B, L, _ = kv.shape
        kv = kv.view(B, L, 2, self.nhead, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, nhead, L, head_dim)
        return kv[0], kv[1]

    def _out_proj(self, mha: nn.MultiheadAttention, attn_out: torch.Tensor):
        """(B, nhead, L, head_dim) -> (B, L, d_model) -> out_proj."""
        B, _, L, _ = attn_out.shape
        out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)

    @torch.no_grad()
    def prefill(
        self,
        code: torch.Tensor,
        code_padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
        token_padding_mask: torch.Tensor | None,
        max_gen_steps: int = 40,
    ) -> torch.Tensor:
        """Full forward pass with KV caching.

        Returns: (M, L, d_model) hidden states at all positions.
        """
        M, L = tokens.shape
        max_cache = L + max_gen_steps
        self._max_cache = max_cache

        # Track per-sequence valid length (non-padding tokens)
        if token_padding_mask is not None:
            self._valid_len = [(~token_padding_mask[i]).sum().item() for i in range(M)]
        else:
            self._valid_len = [L] * M

        # Embed tokens (includes absolute positional encoding)
        x = self.decoder.embed(tokens)
        dtype = x.dtype

        # Self-attention mask: causal + padding
        causal = nn.Transformer.generate_square_subsequent_mask(
            L, dtype=dtype, device=self.device,
        )
        if token_padding_mask is not None:
            pad_float = torch.zeros(M, L, dtype=dtype, device=self.device)
            pad_float.masked_fill_(token_padding_mask, -torch.finfo(dtype).max)
            sa_mask = causal[None, None, :, :] + pad_float[:, None, None, :]
        else:
            sa_mask = causal[None, None, :, :]

        # Cross-attention mask
        if code_padding_mask is not None:
            xa_float = torch.zeros(M, code.shape[1], dtype=dtype, device=self.device)
            xa_float.masked_fill_(code_padding_mask, -torch.finfo(dtype).max)
            self._xa_mask = xa_float[:, None, None, :]
        else:
            self._xa_mask = None

        self._sa_k = []
        self._sa_v = []
        self._xa_k = []
        self._xa_v = []

        for layer in self.layers:
            # === Self-attention ===
            x_n = layer.norm1(x)
            q, k, v = self._proj_qkv(layer.self_attn, x_n)

            # Pre-allocate cache and fill prefill entries
            # Use k's dtype (may differ from x's dtype under autocast)
            k_cache = torch.zeros(
                M, self.nhead, max_cache, self.head_dim, dtype=k.dtype, device=self.device,
            )
            v_cache = torch.zeros(
                M, self.nhead, max_cache, self.head_dim, dtype=v.dtype, device=self.device,
            )
            k_cache[:, :, :L, :] = k
            v_cache[:, :, :L, :] = v
            self._sa_k.append(k_cache)
            self._sa_v.append(v_cache)

            sa_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sa_mask)
            sa_out = self._out_proj(layer.self_attn, sa_out)
            x = x + sa_out

            # === Cross-attention ===
            x_n2 = layer.norm2(x)
            q_xa = self._proj_q(layer.multihead_attn, x_n2)
            k_xa, v_xa = self._proj_kv(layer.multihead_attn, code)
            self._xa_k.append(k_xa)
            self._xa_v.append(v_xa)

            xa_out = F.scaled_dot_product_attention(
                q_xa, k_xa, v_xa, attn_mask=self._xa_mask,
            )
            xa_out = self._out_proj(layer.multihead_attn, xa_out)
            x = x + xa_out

            # === FFN ===
            x_n3 = layer.norm3(x)
            ff = layer.linear2(layer.activation(layer.linear1(x_n3)))
            x = x + ff

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x

    @torch.no_grad()
    def decode_step(
        self,
        new_token_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one new token per sequence using cached K,V.

        Args:
            new_token_ids: (M,) token IDs
            active_mask: (M,) bool, True = active sequence

        Returns: (M, d_model) hidden states for the new tokens
        """
        M = new_token_ids.shape[0]
        positions = torch.tensor(self._valid_len, device=self.device, dtype=torch.long)

        # Embed with PE at correct position
        tokens_2d = new_token_ids.unsqueeze(1)  # (M, 1)
        emb = self.decoder.in_token(tokens_2d)  # (M, 1, d_model)
        pe = self.decoder.pe_dec.pe  # (1, max_pe_len, d_model)
        pos_emb = pe[0, positions].unsqueeze(1)  # (M, 1, d_model)
        x = emb + pos_emb
        dtype = x.dtype

        # Pre-compute attention mask: valid range is 0..valid_len[i] inclusive
        # (because we write new K,V at valid_len[i] before attending)
        # Use cache dtype for mask (may be fp16 under autocast)
        cache_dtype = self._sa_k[0].dtype
        new_valid = positions.clone()
        new_valid[active_mask] += 1
        cache_pos = torch.arange(self._max_cache, device=self.device)
        invalid = cache_pos[None, :] >= new_valid[:, None]  # (M, max_cache)
        sa_mask = torch.zeros(M, self._max_cache, dtype=cache_dtype, device=self.device)
        sa_mask.masked_fill_(invalid, -torch.finfo(cache_dtype).max)
        sa_mask = sa_mask[:, None, None, :]  # (M, 1, 1, max_cache)

        # Active indices for vectorized cache writes
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        active_pos = positions[active_idx]

        for layer_idx, layer in enumerate(self.layers):
            # === Self-attention ===
            x_n = layer.norm1(x)
            q, k_new, v_new = self._proj_qkv(layer.self_attn, x_n)

            # Write new K,V to cache (vectorized)
            self._sa_k[layer_idx][active_idx, :, active_pos, :] = k_new[active_idx, :, 0, :]
            self._sa_v[layer_idx][active_idx, :, active_pos, :] = v_new[active_idx, :, 0, :]

            sa_out = F.scaled_dot_product_attention(
                q, self._sa_k[layer_idx], self._sa_v[layer_idx], attn_mask=sa_mask,
            )
            sa_out = self._out_proj(layer.self_attn, sa_out)
            x = x + sa_out

            # === Cross-attention ===
            x_n2 = layer.norm2(x)
            q_xa = self._proj_q(layer.multihead_attn, x_n2)
            xa_out = F.scaled_dot_product_attention(
                q_xa, self._xa_k[layer_idx], self._xa_v[layer_idx],
                attn_mask=self._xa_mask,
            )
            xa_out = self._out_proj(layer.multihead_attn, xa_out)
            x = x + xa_out

            # === FFN ===
            x_n3 = layer.norm3(x)
            ff = layer.linear2(layer.activation(layer.linear1(x_n3)))
            x = x + ff

        if self.final_norm is not None:
            x = self.final_norm(x)

        # Update valid lengths for active sequences
        for i in active_idx.tolist():
            self._valid_len[i] += 1

        return x.squeeze(1)  # (M, d_model)


class FastSampler(Sampler):
    """Sampler with batched AR inference and parallel chemistry.

    Phase 2 parameters (result-changing optimizations):
        use_fp16: Use half-precision inference via torch.autocast.
        max_branch_states: Only branch the top-N scoring active states per step
            (0 = branch all, default).
        skip_editflow: Skip the EditFlow refinement phase entirely.

    Phase 3 parameters (CPU-bound optimizations):
        rxn_product_limit: Limit _run_rxn to N products (0 = no limit, default).
            push_rxn does random.choice(prods) anyway, so product_limit=1 avoids
            wasted RunReactants calls in combinatorial product enumeration.

    Internal optimizations (always active):
        _fast_copy_state: Replaces copy.deepcopy with shallow-structure copy.
            Shares immutable Molecule/Reaction objects, only copies container
            structure (lists, sets). ~50x faster than deepcopy.
    """

    def __init__(
        self,
        *args,
        max_gpu_batch: int = _MAX_BATCH,
        use_fp16: bool = False,
        max_branch_states: int = 0,
        skip_editflow: bool = False,
        rxn_product_limit: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_gpu_batch = max_gpu_batch
        self._use_fp16 = use_fp16
        self._max_branch_states = max_branch_states
        self._skip_editflow = skip_editflow
        self._rxn_product_limit = rxn_product_limit or None  # 0 → None (no limit)

        if use_fp16:
            # Use autocast context managers instead of model.half()
            # to avoid dtype mismatch with float inputs (e.g. time embedder).
            # Autocast automatically handles mixed-precision for both forward
            # and attention operations.
            pass  # autocast applied in forward methods

    # ------------------------------------------------------------------
    # Override evolve() for skip_editflow support
    # ------------------------------------------------------------------

    def evolve(
        self,
        gpu_lock: Lock | None = None,
        time_limit: TimeLimit | None = None,
        max_evolve_steps: float = 8,
        num_cycles: int = 1,
        num_editflow_samples: int = 10,
        num_editflow_steps: int = 100,
    ) -> None:
        if self._skip_editflow:
            # Run only AR phases (BU + TD), skip EditFlow (i%3==2)
            for i in range(num_cycles * 3):
                if time_limit is not None and time_limit.exceeded():
                    break
                if self.exact_break:
                    max_sim = max([s.score for s in self._finished] or [-1])
                    if max_sim == 1.0:
                        break
                self._finished.sort(key=lambda s: s.score, reverse=True)

                if i % 3 == 2:
                    continue  # skip EditFlow

                sampling_direction = 'bu' if i % 3 == 0 else 'td'
                if self._finished:
                    finished = [s for s in self._finished if s.stack.count_reactions()]
                    if finished:
                        scores = np.array([s.score for s in finished])
                        if scores.sum() == 0:
                            scores = np.ones_like(scores)
                        scores /= scores.sum()
                        self._active = []
                        for _ in range(self._max_active_states):
                            from ..utils.sample_utils import get_sub_stacks, State
                            stack_to_repredict = np.random.choice(finished, p=scores).stack
                            sub_stack = get_sub_stacks(
                                stack=stack_to_repredict, num_samples=1,
                                topdown=sampling_direction == 'td',
                            )[0]
                            self._active.append(State(sub_stack))

                for _ in range(int(max_evolve_steps)):
                    self._evolve_ar_singlestep(
                        gpu_lock=gpu_lock, time_limit=time_limit,
                        sampling_direction=sampling_direction,
                    )
                    if self.exact_break:
                        max_sim = max([s.score for s in self._finished] or [-1])
                        if max_sim == 1.0:
                            break
                self._finished.sort(key=lambda s: s.score, reverse=True)
        else:
            super().evolve(
                gpu_lock=gpu_lock, time_limit=time_limit,
                max_evolve_steps=max_evolve_steps, num_cycles=num_cycles,
                num_editflow_samples=num_editflow_samples,
                num_editflow_steps=num_editflow_steps,
            )

    def _evolve_editflow(self, gpu_lock=None, num_samples=1):
        """Override to wrap EditFlow in autocast when use_fp16=True."""
        if self._use_fp16:
            if gpu_lock is not None:
                gpu_lock.acquire()
            with torch.autocast('cuda', dtype=torch.float16):
                # Replicate parent logic inside autocast
                scores = np.array([s.score for s in self._finished])
                if scores.sum() == 0:
                    scores = np.ones_like(scores)
                scores /= scores.sum()
                states = [np.random.choice(self._finished, p=scores) for _ in range(num_samples)]
                from ..data.common import featurize_stack
                feat_list = [featurize_stack(s.stack, end_token=True) for s in states]
                inputs = self._collate_editflow(feat_list)['tokens'].to(self.device)
                code, code_padding_mask = self.code_editflow
                code = code.expand(len(inputs), -1, -1)
                code_padding_mask = code_padding_mask.expand(len(inputs), -1)
                finished = self._predict_editflow(
                    code=code, code_padding_mask=code_padding_mask, tokens=inputs,
                )
            self._add_finished_states(finished)
            if gpu_lock is not None:
                gpu_lock.release()
        else:
            super()._evolve_editflow(gpu_lock=gpu_lock, num_samples=num_samples)

    # ------------------------------------------------------------------
    # Batched first-token prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_first_token_batch(
        self,
        code: torch.Tensor,
        code_padding_mask: torch.Tensor,
        token_list: list[torch.Tensor],
        temperature: float = 0.1,
        sampling_direction: str = 'bu',
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Predict the first token for every active state in one GPU call.

        Args:
            token_list: list of 1-D token tensors (variable length, on device)

        Returns:
            first_tokens: (N,) sampled token ids
            all_logits: (N, vocab) raw logits
            lengths: list of original token lengths
        """
        N = len(token_list)
        lengths = [t.size(0) for t in token_list]
        max_len = max(lengths)

        # Pad tokens and build masks
        padded = torch.zeros(N, max_len, dtype=torch.long, device=self.device)
        pad_mask = torch.ones(N, max_len, dtype=torch.bool, device=self.device)
        for i, t in enumerate(token_list):
            padded[i, :lengths[i]] = t
            pad_mask[i, :lengths[i]] = False

        # Expand encoder output (zero-copy via broadcast)
        code_batch = code.expand(N, -1, -1)
        code_mask_batch = code_padding_mask.expand(N, -1)

        # --- sub-batch forward if N > max_gpu_batch ---
        all_logits = torch.empty(N, self.model.vocab_size, device=self.device)
        for start in range(0, N, self._max_gpu_batch):
            end = min(start + self._max_gpu_batch, N)
            h = self.model.decoder(
                code=code_batch[start:end],
                code_padding_mask=code_mask_batch[start:end],
                tokens=padded[start:end],
                token_padding_mask=pad_mask[start:end],
            )  # (B, seq_len, d_model)
            # Gather last real position per state
            batch_lengths = torch.tensor(
                lengths[start:end], device=self.device, dtype=torch.long
            )
            last_idx = batch_lengths - 1  # (B,)
            h_last = h[torch.arange(end - start, device=self.device), last_idx]
            all_logits[start:end] = self.model.token_head(h_last)

        # TD enforcement: only RXN tokens when tokens == [START]
        if sampling_direction == 'td':
            td_mask = torch.tensor(
                [l == 1 for l in lengths], dtype=torch.bool, device=self.device
            )
            if td_mask.any():
                all_logits[td_mask, :TokenType.RXN_MIN] = -torch.inf

        # Abort states that exceed max_len
        for i, l in enumerate(lengths):
            if l > self.model.max_len:
                all_logits[i] = -torch.inf
                all_logits[i, TokenType.END] = 0.0  # force END

        # Sample (cast to float32 for numerical stability in softmax)
        probs = F.softmax(all_logits.float() / temperature, dim=-1)
        first_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (N,)

        return first_tokens, all_logits, lengths

    # ------------------------------------------------------------------
    # Batched BB SMILES generation with KV cache
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_bb_batch(
        self,
        code: torch.Tensor,
        code_padding_mask: torch.Tensor,
        bb_indices: list[int],
        all_padded_tokens: torch.Tensor,
        all_lengths: list[int],
        first_tokens: torch.Tensor,
        temperature: float = 0.1,
    ) -> list[torch.Tensor]:
        """BB SMILES generation with KV cache — O(1) per step instead of O(L).

        Uses _KVCacheDecoder to cache self-attention and cross-attention K,V,
        avoiding full-sequence reprocessing at each autoregressive step.
        """
        if not bb_indices:
            return []

        M = len(bb_indices)
        max_extra = 40  # BB SMILES rarely exceeds 30 tokens

        # Build initial sequences: original tokens + first_token
        init_seqs = []
        for idx in bb_indices:
            orig = all_padded_tokens[idx, :all_lengths[idx]]
            ft = first_tokens[idx].unsqueeze(0)
            init_seqs.append(torch.cat([orig, ft]))

        cur_lengths = [s.size(0) for s in init_seqs]
        max_init = max(cur_lengths)

        # Pad for prefill
        prefill_tokens = torch.zeros(M, max_init, dtype=torch.long, device=self.device)
        prefill_mask = torch.ones(M, max_init, dtype=torch.bool, device=self.device)
        for i, s in enumerate(init_seqs):
            prefill_tokens[i, :s.size(0)] = s
            prefill_mask[i, :s.size(0)] = False

        active = torch.ones(M, dtype=torch.bool, device=self.device)
        for i in range(M):
            last_tok = prefill_tokens[i, cur_lengths[i] - 1].item()
            if last_tok == TokenType.MOL_END or cur_lengths[i] >= self.model.max_len - 2:
                active[i] = False

        # Collect generated tokens per BB state (first_token is always included)
        generated = [[] for _ in range(M)]

        if not active.any():
            return self._build_bb_token_lists(M, bb_indices, first_tokens, generated)

        # Initialize KV cache decoder
        layer0 = self.model.decoder.dec.layers[0]
        kv = _KVCacheDecoder(
            self.model.decoder,
            nhead=layer0.self_attn.num_heads,
            d_model=self.model.d_model,
            device=self.device,
        )
        code_bb = code.expand(M, -1, -1)
        code_mask_bb = code_padding_mask.expand(M, -1)

        # Prefill: cache K,V, get logits at last real position
        h = kv.prefill(code_bb, code_mask_bb, prefill_tokens, prefill_mask,
                       max_gen_steps=max_extra)

        last_pos = torch.tensor(cur_lengths, device=self.device, dtype=torch.long) - 1
        h_last = h[torch.arange(M, device=self.device), last_pos]
        logits = self.model.token_head(h_last)
        probs = F.softmax(logits.float() / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)  # (M,)

        # Process first sampled token
        for i in range(M):
            if active[i]:
                tok = sampled[i].item()
                generated[i].append(tok)
                if tok == TokenType.MOL_END or cur_lengths[i] >= self.model.max_len - 2:
                    active[i] = False
                else:
                    cur_lengths[i] += 1

        # Decode loop with KV cache — single token per step
        for step in range(1, max_extra):
            if not active.any():
                break

            h_step = kv.decode_step(sampled, active)  # (M, d_model)
            logits = self.model.token_head(h_step)
            probs = F.softmax(logits.float() / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            for i in range(M):
                if active[i]:
                    tok = sampled[i].item()
                    generated[i].append(tok)
                    if tok == TokenType.MOL_END or cur_lengths[i] >= self.model.max_len - 2:
                        active[i] = False
                    else:
                        cur_lengths[i] += 1

        return self._build_bb_token_lists(M, bb_indices, first_tokens, generated)

    def _build_bb_token_lists(
        self,
        M: int,
        bb_indices: list[int],
        first_tokens: torch.Tensor,
        generated: list[list[int]],
    ) -> list[torch.Tensor]:
        """Combine first_token + generated tokens into decode_smiles-compatible lists."""
        bb_token_lists = []
        for i in range(M):
            toks = [first_tokens[bb_indices[i]].item()] + generated[i]
            # Remove trailing MOL_END (decode_smiles would skip it anyway)
            if toks and toks[-1] == TokenType.MOL_END:
                toks = toks[:-1]
            bb_token_lists.append(
                torch.tensor(toks, dtype=torch.long, device=self.device)
            )
        return bb_token_lists

    # ------------------------------------------------------------------
    # Batched fpindex queries
    # ------------------------------------------------------------------

    def _batch_get_reactants(
        self, smiles_list: list[str],
    ) -> list[list[_ReactantItem] | None]:
        """Batch version of get_reactants: one cdist call for all valid molecules."""
        M = len(smiles_list)
        fps = []
        fp_indices = []  # map into smiles_list for valid mols
        edit_indices = []  # indices needing edit-distance fallback

        for i, smi in enumerate(smiles_list):
            mol = Molecule(smi)
            if mol._rdmol is not None:
                fp = torch.tensor(
                    mol.get_fingerprint(option=self._fpindex._fp_option),
                    dtype=torch.float,
                )
                fps.append(fp)
                fp_indices.append(i)
            else:
                edit_indices.append(i)

        results: list[list[_ReactantItem] | None] = [None] * M

        # Batch GPU query for valid molecules
        if fps:
            fps_batch = torch.stack(fps).to(self.device)  # (V, fp_dim)
            query_results = self._fpindex.query_cuda(fps_batch, k=100)
            for local_i, global_i in enumerate(fp_indices):
                qr = query_results[local_i]
                mols_arr = np.array([q.molecule for q in qr])
                mol_idxs = np.array([q.index for q in qr])
                distances = np.array([q.distance for q in qr])
                scores = 1.0 / (distances + 0.1)
                sorted_idx = (-scores).argsort()
                results[global_i] = [
                    _ReactantItem(
                        reactant=mols_arr[j], index=int(mol_idxs[j]), score=float(scores[j]),
                    )
                    for j in sorted_idx
                ]

        # Edit-distance fallback for invalid molecules
        for i in edit_indices:
            results[i] = get_reactants(
                smiles_list[i], fpindex=self._fpindex, topk=100,
                use_edit_distance=True,
            )

        return results

    # ------------------------------------------------------------------
    # Process a single branch (for ThreadPoolExecutor)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_branch_bu(
        base_state: State,
        sampled_type: str,
        sampled_item,
        branch_i: int,
        mols_to_filter,
        filter_sim: float,
        product_limit: int | None = None,
    ) -> tuple[str, State | None, State | None]:
        """Process one branch for BU direction. Returns (action, next_state, finished_state)."""
        if sampled_type == 'END':
            return 'finished', None, base_state

        if sampled_type == 'BB':
            if branch_i >= len(sampled_item):
                return 'aborted', None, None
            reactant, mol_idx, score = sampled_item[branch_i]
            new_state = _fast_copy_state(base_state)
            new_state.stack.push_mol(reactant, mol_idx)
            new_state.scores.append(score)
            return 'next', new_state, None

        if sampled_type == 'RXN':
            if branch_i >= len(sampled_item):
                return 'aborted', None, None
            new_state = _fast_copy_state(base_state)
            # Try reactions starting from branch_i
            j = branch_i
            for j in range(branch_i, len(sampled_item)):
                reaction, rxn_idx, score = sampled_item[j]
                success = new_state.stack.push_rxn(reaction, rxn_idx, product_limit=product_limit)
                if success:
                    # Intermediate filtering
                    filtered = False
                    if mols_to_filter is not None:
                        for m in new_state.stack.get_top():
                            if max([m.sim(f) for f in mols_to_filter]) > filter_sim:
                                filtered = True
                                break
                    if not filtered:
                        new_state.scores.append(score)
                        return 'both', new_state, new_state  # goes to both next and finished
                    # If filtered, continue trying
            return 'aborted', None, None

        # ABORTED
        return 'aborted', None, None

    @staticmethod
    def _process_branch_td(
        base_state: State,
        sampled_type: str,
        sampled_item,
        branch_i: int,
    ) -> tuple[str, State | None, State | None]:
        """Process one branch for TD direction."""
        if sampled_type == 'END':
            return 'finished', None, base_state

        if sampled_type in ('BB', 'RXN'):
            if branch_i >= len(sampled_item):
                return 'aborted', None, None
            mol_or_rxn, idx, score = sampled_item[branch_i]
            new_state = _fast_copy_state(base_state)
            new_state.stack.push_topdown(mol_or_rxn, idx)
            new_state.scores.append(score)
            return 'next', new_state, None

        return 'aborted', None, None

    # ------------------------------------------------------------------
    # Overridden: batched _evolve_ar_singlestep
    # ------------------------------------------------------------------

    def _evolve_ar_singlestep(
        self,
        gpu_lock: Lock | None = None,
        time_limit: TimeLimit | None = None,
        sampling_direction: str = 'bu',
    ) -> None:
        if len(self._active) == 0:
            return

        # ---- Phase 1: Featurize all states (CPU) ----
        feat_list = [
            featurize_stack(
                state.stack,
                end_token=False,
                sampling_direction=sampling_direction,
            )
            for state in self._active
        ]

        # Prepare token tensors
        token_list: list[torch.Tensor] = []
        for feat in feat_list:
            tokens = feat['tokens'].to(self.device)
            if sampling_direction == 'bu' and len(tokens) == 1:
                tokens = torch.cat([
                    tokens,
                    torch.tensor([TokenType.MOL_START], device=self.device),
                ])
            token_list.append(tokens)

        # ---- Phase 2: GPU inference (under lock) ----
        if gpu_lock is not None:
            gpu_lock.acquire()

        # P2: autocast context for FP16 inference
        _autocast = (
            torch.autocast('cuda', dtype=torch.float16)
            if self._use_fp16
            else _nullcontext()
        )

        with _autocast:
            code, code_padding_mask = self.code

            # 2a: Batch first-token prediction
            first_tokens, all_logits, lengths = self._predict_first_token_batch(
                code, code_padding_mask, token_list,
                sampling_direction=sampling_direction,
            )

            # Classify each state by first token type
            first_tokens_cpu = first_tokens.cpu().tolist()

            bb_indices = []   # states needing BB SMILES generation
            rxn_indices = []  # states with RXN prediction
            end_indices = []  # states that finished
            abort_indices = []  # states that aborted

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
                    end_indices.append(i)  # Treat unknown tokens as END

            # 2b: Batch BB SMILES generation
            # Build padded tokens for generate_bb_batch
            max_tok_len = max(lengths)
            padded_tokens = torch.zeros(
                len(token_list), max_tok_len, dtype=torch.long, device=self.device,
            )
            for i, t in enumerate(token_list):
                padded_tokens[i, :lengths[i]] = t

            bb_token_lists = self._generate_bb_batch(
                code, code_padding_mask,
                bb_indices, padded_tokens, lengths, first_tokens,
            )

        # 2c: Process BB results — batched fpindex queries (single cdist call)
        bb_results: dict[int, list] = {}
        if bb_indices:
            smiles_list = [decode_smiles(bb_token_lists[i]) for i in range(len(bb_indices))]
            all_reactants = self._batch_get_reactants(smiles_list)
            for local_i, state_i in enumerate(bb_indices):
                reactants = all_reactants[local_i]
                if reactants is None:
                    abort_indices.append(state_i)
                else:
                    bb_results[state_i] = reactants

        # 2d: Process RXN results — extract logits, get_reactions
        rxn_results: dict[int, list] = {}
        for state_i in rxn_indices:
            reaction_logits = all_logits[state_i, TokenType.RXN_MIN:TokenType.RXN_MAX + 1].float()
            rxn_results[state_i] = get_reactions(
                reaction_logits, rxn_matrix=self._rxn_matrix,
            )

        if gpu_lock is not None:
            gpu_lock.release()

        # ---- Phase 3: State branching (parallel CPU, no GPU lock needed) ----
        # Build per-state PredictResults
        state_results: dict[int, tuple[str, list | None]] = {}
        for i in end_indices:
            state_results[i] = ('END', None)
        for i in abort_indices:
            state_results[i] = ('ABORTED', None)
        for i, items in bb_results.items():
            state_results[i] = ('BB', items)
        for i, items in rxn_results.items():
            state_results[i] = ('RXN', items)

        finished: list[State] = []
        next_states: list[State] = []

        # P2: max_branch_states — only branch top-N scoring states
        if self._max_branch_states > 0:
            branchable = [
                i for i, (t, _) in state_results.items()
                if t in ('BB', 'RXN')
            ]
            if len(branchable) > self._max_branch_states:
                scored = sorted(
                    branchable,
                    key=lambda i: self._active[i].score,
                    reverse=True,
                )
                pruned = set(scored[self._max_branch_states:])
                for i in pruned:
                    state_results[i] = ('ABORTED', None)

        # For small state counts, skip ThreadPool overhead
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

    def _process_rxn_branches_bu(
        self,
        base_state: State,
        sampled_item: list,
        next_states: list[State],
        finished: list[State],
        aborted: list[State],
    ) -> None:
        """Handle RXN branching in BU mode. Matches original Sampler logic exactly.

        The original code mutates sampled_item by removing failed reactions,
        so we keep this serial to preserve exact behavior.
        """
        product_limit = self._rxn_product_limit
        for i in range(self._factor):
            if i >= len(sampled_item):
                aborted.append(_fast_copy_state(base_state))
                continue
            new_state = _fast_copy_state(base_state)
            for j in range(i, len(sampled_item)):
                reaction, rxn_idx, score = sampled_item[j]
                success = new_state.stack.push_rxn(reaction, rxn_idx, product_limit=product_limit)
                if success:
                    filtered = False
                    if self._mols_to_filter is not None:
                        for m in new_state.stack.get_top():
                            if max([m.sim(f) for f in self._mols_to_filter]) > self._filter_sim:
                                filtered = True
                                break
                    if not filtered:
                        finished.append(new_state)
                        new_state.scores.append(score)
                        next_states.append(new_state)
                        break
            else:
                j += 1
                aborted.append(new_state)
            sampled_item = sampled_item[:i] + sampled_item[j:]

