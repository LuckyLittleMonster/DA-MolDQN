"""Neural networks for Route-DQN.

Three components:
  RouteStateEncoder:       route -> fixed-dim embedding (mean pool over steps)
  PositionPolicyNetwork:   route_state -> Q-values for each route position
  RouteBBScoringNetwork:   (route_state, position_context, template) -> BB scores
"""

import torch
import torch.nn as nn


class RouteStateEncoder(nn.Module):
    """Encode a synthesis route into a fixed-dimension vector.

    Each step is represented by (intermediate_fp, template_emb, block_emb).
    Steps are encoded independently then mean-pooled into a route embedding.

    Args:
        fp_dim: Molecular fingerprint dimension (4096 for Morgan FP).
        template_emb_dim: Template embedding dimension.
        block_emb_dim: Block embedding dimension.
        step_hidden: Hidden dim for per-step encoder.
        route_emb_dim: Output route embedding dimension.
    """

    def __init__(self, fp_dim=4096, template_emb_dim=128, block_emb_dim=64,
                 step_hidden=256, route_emb_dim=256):
        super().__init__()
        self.route_emb_dim = route_emb_dim
        input_dim = fp_dim + template_emb_dim + block_emb_dim

        self.step_encoder = nn.Sequential(
            nn.Linear(input_dim, step_hidden),
            nn.ReLU(),
            nn.Linear(step_hidden, route_emb_dim),
        )

    def forward(self, step_fps, step_template_embs, step_block_embs,
                route_lengths):
        """Encode routes from packed step features.

        Args:
            step_fps: (total_steps, fp_dim) fingerprints of intermediates.
            step_template_embs: (total_steps, template_emb_dim).
            step_block_embs: (total_steps, block_emb_dim).
            route_lengths: (batch,) number of steps per route.

        Returns:
            route_embs: (batch, route_emb_dim).
        """
        combined = torch.cat(
            [step_fps, step_template_embs, step_block_embs], dim=-1)
        step_embs = self.step_encoder(combined)  # (total_steps, route_emb_dim)

        # Mean pool per route using split
        route_emb_list = torch.split(step_embs, route_lengths.tolist())
        route_embs = torch.stack(
            [chunk.mean(dim=0) for chunk in route_emb_list])
        return route_embs


class PositionPolicyNetwork(nn.Module):
    """Q_pos: Select which route position to modify.

    Takes a route state embedding and outputs Q-values for each position.
    Positions masked by L1 (uni-molecular steps) get -inf.

    Output dimensions: max_route_len + 2 (EXTEND and TRUNCATE virtual positions).

    Args:
        route_emb_dim: Input route embedding dimension.
        max_route_len: Maximum number of steps in a route.
        hidden_dim: Hidden layer size.
    """

    def __init__(self, route_emb_dim=256, max_route_len=10, hidden_dim=128):
        super().__init__()
        self.max_route_len = max_route_len
        self.n_positions = max_route_len + 2  # +2 for EXTEND, TRUNCATE
        self.scorer = nn.Sequential(
            nn.Linear(route_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_positions),
        )

    def forward(self, route_state, position_mask):
        """Score positions for modification.

        Args:
            route_state: (batch, route_emb_dim) route embeddings.
            position_mask: (batch, n_positions) bool, True = modifiable.

        Returns:
            q_values: (batch, n_positions) with -inf at masked positions.
        """
        q = self.scorer(route_state)
        q = q.masked_fill(~position_mask, float('-inf'))
        return q


class RouteBBScoringNetwork(nn.Module):
    """Q_bb: Score building blocks given route context + position.

    Architecture mirrors BlockScoringNetwork but takes route_emb + position_fp
    as context instead of just molecule fp.

    Args:
        route_emb_dim: Route embedding dimension.
        fp_dim: Molecular fingerprint dimension.
        template_emb_dim: Template embedding dimension.
        hidden_dim: MLP hidden dimension.
        block_emb_dim: Block embedding dimension for bilinear scoring.
    """

    def __init__(self, route_emb_dim=256, fp_dim=4096, template_emb_dim=128,
                 hidden_dim=256, block_emb_dim=64):
        super().__init__()
        self.block_emb_dim = block_emb_dim

        # Encode (route_emb + position_intermediate_fp) -> template_emb_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(route_emb_dim + fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, template_emb_dim),
        )
        # Combine context + template -> block scoring space
        self.condition_mlp = nn.Sequential(
            nn.Linear(template_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, block_emb_dim),
        )
        # Block fingerprint -> block embedding (for training gradient flow)
        self.block_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, block_emb_dim),
        )
        # Precomputed block embeddings for fast inference
        self.register_buffer('block_embs', torch.empty(0))

    @torch.no_grad()
    def precompute_block_embeddings(self, block_fps):
        """Precompute block embeddings from fingerprints.

        Args:
            block_fps: (n_blocks, fp_dim) tensor on model device.
        """
        embs = []
        for i in range(0, len(block_fps), 2048):
            embs.append(self.block_encoder(block_fps[i:i + 2048]))
        self.block_embs = torch.cat(embs, dim=0)

    def forward(self, route_emb, position_fp, template_emb, compat_mask):
        """Score all blocks using precomputed embeddings (inference).

        Args:
            route_emb: (batch, route_emb_dim) route state.
            position_fp: (batch, fp_dim) intermediate at modified position.
            template_emb: (batch, template_emb_dim) position's template.
            compat_mask: (batch, n_blocks) bool, True = compatible (L2+L3).

        Returns:
            q_values: (batch, n_blocks).
        """
        context = self.context_encoder(
            torch.cat([route_emb, position_fp], dim=-1))
        combined = torch.cat([context, template_emb], dim=-1)
        scoring_emb = self.condition_mlp(combined)  # (batch, block_emb_dim)
        q = scoring_emb @ self.block_embs.T  # (batch, n_blocks)
        q = q.masked_fill(~compat_mask, float('-inf'))
        return q

    def score_single_block(self, route_emb, position_fp, template_emb,
                           block_fp):
        """Score specific blocks with gradient flow (training).

        Args:
            route_emb: (batch, route_emb_dim).
            position_fp: (batch, fp_dim).
            template_emb: (batch, template_emb_dim).
            block_fp: (batch, fp_dim) fingerprints of selected blocks.

        Returns:
            q_values: (batch,) scalar Q-values.
        """
        context = self.context_encoder(
            torch.cat([route_emb, position_fp], dim=-1))
        combined = torch.cat([context, template_emb], dim=-1)
        scoring_emb = self.condition_mlp(combined)
        block_emb = self.block_encoder(block_fp)
        return (scoring_emb * block_emb).sum(dim=-1)
