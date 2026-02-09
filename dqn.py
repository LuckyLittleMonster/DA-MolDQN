"""DQN models for molecular optimization.

BaseDQN — abstract base (forward: observations → Q-values).
MLPDQN  — 3-layer MLP (fingerprint + step → Q-value).
DQN     — backward-compatible alias for MLPDQN.
create_dqn() — factory that dispatches to MLPDQN or GNNDQN.
"""

from abc import abstractmethod
import torch.nn as nn


class BaseDQN(nn.Module):
    """Abstract base for all DQN variants."""

    @abstractmethod
    def forward(self, x):
        """Map observation tensor → Q-values (batch, 1)."""
        ...


class MLPDQN(BaseDQN):
    """Simple 3-layer MLP DQN for molecular optimization."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# Backward-compatible alias — state_dict keys identical to old DQN class.
DQN = MLPDQN


def create_dqn(model_type, device, *, input_dim=None,
               gnn_checkpoint=None, q_head_hidden=512):
    """Factory: create a DQN model pair (policy + target).

    Args:
        model_type: 'mlp' or 'gnn'.
        device: torch device string or object.
        input_dim: Required for 'mlp'.
        gnn_checkpoint: Required for 'gnn'.
        q_head_hidden: Hidden dim for GNN Q-head.

    Returns:
        (dqn, target_dqn, optimizer_params)
        optimizer_params: iterable of parameters to optimise.
    """
    import torch

    if model_type == 'gnn':
        from gnn_dqn import create_gnn_dqn
        dqn = create_gnn_dqn(checkpoint_path=gnn_checkpoint, device=str(device))
        target_dqn = create_gnn_dqn(checkpoint_path=gnn_checkpoint, device=str(device))
        target_dqn.load_state_dict(dqn.state_dict())
        optimizer_params = [p for p in dqn.parameters() if p.requires_grad]
        return dqn, target_dqn, optimizer_params
    else:
        assert input_dim is not None, "input_dim required for MLP DQN"
        dqn = MLPDQN(input_dim).to(device)
        target_dqn = MLPDQN(input_dim).to(device)
        target_dqn.load_state_dict(dqn.state_dict())
        optimizer_params = list(dqn.parameters())
        return dqn, target_dqn, optimizer_params
