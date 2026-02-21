"""Neural networks for hierarchical template and building block selection.

Q1 (TemplatePolicyNetwork):
    state_fp -> score for each template via bilinear scoring with learned embeddings.
    scores = MLP(fp) @ template_embs.T + bias

Q2 (BlockScoringNetwork):
    (state_fp, template_emb) -> score for each building block via bilinear scoring.
    score = MLP(cat(state_emb, template_emb)) @ block_embs.T

Both use bilinear scoring for efficient GPU inference: one matrix multiply
scores ALL candidates simultaneously (~0.1ms for 10K blocks).
"""

import torch
import torch.nn as nn


class TemplatePolicyNetwork(nn.Module):
    """Q1: Score reaction templates given molecular state.

    Uses learned template embeddings with bilinear scoring.
    Each template gets a learnable embedding vector. The molecule state
    is encoded to the same space, and scores = dot product + bias.

    Args:
        fp_dim: Input fingerprint dimension (4096 for Morgan FP).
        n_templates: Number of reaction templates.
        hidden_dim: MLP hidden layer size.
        emb_dim: Embedding dimension for bilinear scoring.
    """

    def __init__(self, fp_dim=4096, n_templates=116,
                 hidden_dim=256, emb_dim=128):
        super().__init__()
        self.n_templates = n_templates
        self.emb_dim = emb_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        # Learnable per-template embeddings
        self.template_embs = nn.Parameter(
            torch.randn(n_templates, emb_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_templates))

    def forward(self, state_fp, template_mask=None):
        """Score all templates for given state(s).

        Args:
            state_fp: (batch, fp_dim) or (fp_dim,) molecular fingerprints.
            template_mask: (batch, n_templates) or (n_templates,) bool.
                True = template is applicable. None = all valid.

        Returns:
            q_values: (batch, n_templates) or (n_templates,).
        """
        squeeze = state_fp.dim() == 1
        if squeeze:
            state_fp = state_fp.unsqueeze(0)
            if template_mask is not None:
                template_mask = template_mask.unsqueeze(0)

        state_emb = self.state_encoder(state_fp)  # (batch, emb_dim)
        q = state_emb @ self.template_embs.T + self.bias  # (batch, n_templates)

        if template_mask is not None:
            q = q.masked_fill(~template_mask, float('-inf'))

        if squeeze:
            q = q.squeeze(0)
        return q

    def get_template_embedding(self, template_ids):
        """Get template embeddings for given IDs.

        Args:
            template_ids: (batch,) long tensor or int.
        Returns:
            embs: (batch, emb_dim) or (emb_dim,).
        """
        return self.template_embs[template_ids]


class BlockScoringNetwork(nn.Module):
    """Q2: Score building blocks given state + selected template.

    Architecture:
        state_emb = state_encoder(fp)
        conditioned = condition_mlp(cat(state_emb, template_emb))
        block_emb = block_encoder(block_fp)  [precomputed for inference]
        score = conditioned @ block_embs.T

    Block embeddings are precomputed from block fingerprints at init time
    for fast inference. During training, block_encoder is called per-sample
    for proper gradient flow.

    Args:
        fp_dim: Input fingerprint dimension.
        template_emb_dim: Template embedding dim (from Q1).
        hidden_dim: MLP hidden layer size.
        block_emb_dim: Block embedding output dimension.
    """

    def __init__(self, fp_dim=4096, template_emb_dim=128,
                 hidden_dim=256, block_emb_dim=64):
        super().__init__()
        self.block_emb_dim = block_emb_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, template_emb_dim),
        )
        # Combines state + template info -> block scoring space
        self.condition_mlp = nn.Sequential(
            nn.Linear(template_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, block_emb_dim),
        )
        # Block fingerprint -> block embedding
        self.block_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, block_emb_dim),
        )
        # Precomputed block embeddings: (n_blocks, block_emb_dim)
        self.register_buffer('block_embs', torch.empty(0))

    @torch.no_grad()
    def precompute_block_embeddings(self, block_fps):
        """Precompute and cache block embeddings for inference.

        Call once at init, and periodically during training to refresh.

        Args:
            block_fps: (n_blocks, fp_dim) tensor on same device as model.
        """
        embs = []
        for i in range(0, len(block_fps), 2048):
            embs.append(self.block_encoder(block_fps[i:i + 2048]))
        self.block_embs = torch.cat(embs, dim=0)

    def forward(self, state_fp, template_emb, compat_mask=None):
        """Score all blocks using cached embeddings (inference mode).

        Args:
            state_fp: (batch, fp_dim) molecular fingerprints.
            template_emb: (batch, template_emb_dim) from Q1.
            compat_mask: (batch, n_blocks) bool. True = compatible.

        Returns:
            q_values: (batch, n_blocks).
        """
        state_emb = self.state_encoder(state_fp)
        combined = torch.cat([state_emb, template_emb], dim=-1)
        scoring_emb = self.condition_mlp(combined)  # (batch, block_emb_dim)
        q = scoring_emb @ self.block_embs.T  # (batch, n_blocks)

        if compat_mask is not None:
            q = q.masked_fill(~compat_mask, float('-inf'))
        return q

    def score_single_block(self, state_fp, template_emb, block_fp):
        """Score specific blocks with gradient flow (training mode).

        Computes block embeddings on-the-fly so gradients flow through
        the block_encoder as well.

        Args:
            state_fp: (batch, fp_dim).
            template_emb: (batch, template_emb_dim).
            block_fp: (batch, fp_dim) fingerprints of selected blocks.

        Returns:
            q_values: (batch,) scalar Q-values.
        """
        state_emb = self.state_encoder(state_fp)
        combined = torch.cat([state_emb, template_emb], dim=-1)
        scoring_emb = self.condition_mlp(combined)  # (batch, block_emb_dim)
        block_emb = self.block_encoder(block_fp)  # (batch, block_emb_dim)
        return (scoring_emb * block_emb).sum(dim=-1)  # (batch,) dot product
