"""PyTorch port of the BDE-db2 nfp MPNN model.

Architecture: 6 EdgeUpdate + 5 NodeUpdate rounds with residual connections.
Input: molecular graphs (atom tokens, bond tokens, connectivity, bond_indices)
Output: per-bond [BDE, BDFE] predictions in kcal/mol
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConcatDense(nn.Module):
    """Concatenate inputs then Dense(2*F, relu) -> Dense(F)."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, 2 * output_dim)
        self.dense2 = nn.Linear(2 * output_dim, output_dim)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        x = torch.cat(tensors, dim=-1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


class EdgeUpdateLayer(nn.Module):
    """For each edge, gather src/dst atoms, concat with bond, ConcatDense."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.concat_dense = ConcatDense(3 * hidden_dim, hidden_dim)

    def forward(self, atom_state, bond_state, src_idx, dst_idx, batch_idx, bond_mask):
        # Gather source and target atoms
        src_atom = atom_state[batch_idx, src_idx]
        dst_atom = atom_state[batch_idx, dst_idx]
        new_bond = self.concat_dense(bond_state, src_atom, dst_atom)
        new_bond = new_bond * bond_mask.unsqueeze(-1)
        return new_bond


class NodeUpdateLayer(nn.Module):
    """Aggregate edge messages to nodes, then MLP."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message_dense = ConcatDense(2 * hidden_dim, hidden_dim)
        self.update_dense_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.update_dense_2 = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, atom_state, bond_state, src_idx, dst_idx, batch_idx, bond_mask, num_atoms):
        B, _, H = bond_state.shape
        # source_atom = gather via connectivity[:,:,1] (neighbor endpoint)
        neighbor_atom = atom_state[batch_idx, dst_idx]
        messages = self.message_dense(neighbor_atom, bond_state)
        messages = messages * bond_mask.unsqueeze(-1)

        # Scatter-sum to target node via connectivity[:,:,0]
        agg = torch.zeros(B, num_atoms, H, device=bond_state.device, dtype=bond_state.dtype)
        target_expanded = src_idx.unsqueeze(-1).expand_as(messages)
        agg.scatter_add_(1, target_expanded, messages)

        new_atom = F.relu(self.update_dense_1(agg))
        new_atom = self.update_dense_2(new_atom)
        return new_atom


class BDEPredictor(nn.Module):
    """BDE-db2 MPNN: per-bond BDE and BDFE from molecular graph.

    6 EdgeUpdate + 5 NodeUpdate with residual. Output = mean + deviation.
    """
    def __init__(self, num_atom_types=171, num_bond_types=200, hidden_dim=128,
                 num_messages=6, output_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_messages = num_messages

        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim, padding_idx=0)
        self.bond_embedding = nn.Embedding(num_bond_types, hidden_dim, padding_idx=0)
        self.edge_updates = nn.ModuleList([EdgeUpdateLayer(hidden_dim) for _ in range(num_messages)])
        self.node_updates = nn.ModuleList([NodeUpdateLayer(hidden_dim) for _ in range(num_messages - 1)])
        self.bde_mean = nn.Embedding(num_bond_types, output_dim, padding_idx=0)
        self.bde_no_mean = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, atom, bond, connectivity, bond_indices):
        """
        Args:
            atom: [B, N] long
            bond: [B, E] long
            connectivity: [B, E, 2] long - (src, dst) per directed edge
            bond_indices: [B, E] long - RDKit bond index per directed edge
        Returns:
            [B, n_bonds, 2] float - per-bond [BDE, BDFE]
        """
        B, N = atom.shape
        E = bond.shape[1]
        H = self.hidden_dim

        atom_state = self.atom_embedding(atom)
        bond_state = self.bond_embedding(bond)

        # Masks must match embedding dtype for fp16 compatibility
        dtype = bond_state.dtype
        bond_mask = (bond != 0).to(dtype)
        atom_mask = (atom != 0).to(dtype)

        # Precompute indices for gather
        src_idx = connectivity[:, :, 0]  # [B, E]
        dst_idx = connectivity[:, :, 1]  # [B, E]
        batch_idx = torch.arange(B, device=atom.device).unsqueeze(1).expand(B, E)

        for i in range(self.num_messages):
            new_bond = self.edge_updates[i](atom_state, bond_state, src_idx, dst_idx, batch_idx, bond_mask)
            bond_state = bond_state + new_bond

            if i < self.num_messages - 1:
                new_atom = self.node_updates[i](atom_state, bond_state, src_idx, dst_idx, batch_idx, bond_mask, N)
                new_atom = new_atom * atom_mask.unsqueeze(-1)
                atom_state = atom_state + new_atom

        # Readout: scatter-mean by bond_indices, then slice first half
        # The TF Reduce(mean) scatters into a tensor of same size as input,
        # then Slice takes the first n_directed//2 entries.
        # bond_indices maps directed edges to undirected bond indices (0,0,1,1,2,2,...)

        # We scatter into a buffer of size E (same as directed edges)
        bond_agg = torch.zeros(B, E, H, device=bond_state.device, dtype=bond_state.dtype)
        bond_count = torch.zeros(B, E, 1, device=bond_state.device, dtype=bond_state.dtype)

        bi_expanded = bond_indices.unsqueeze(-1).expand(B, E, H)
        bi_count = bond_indices.unsqueeze(-1)

        masked_state = bond_state * bond_mask.unsqueeze(-1)
        masked_count = bond_mask.unsqueeze(-1)

        bond_agg.scatter_add_(1, bi_expanded, masked_state)
        bond_count.scatter_add_(1, bi_count, masked_count)
        bond_count = bond_count.clamp(min=1)
        bond_features = bond_agg / bond_count  # [B, E, H]

        # Slice: first half = undirected bonds
        n_bonds = E // 2
        bond_features = bond_features[:, :n_bonds, :]  # [B, n_bonds, H]

        # BDE deviation
        bde_deviation = self.bde_no_mean(bond_features)  # [B, n_bonds, 2]

        # BDE mean: same scatter-mean-slice pattern
        bde_mean_emb = self.bde_mean(bond)  # [B, E, 2]
        bde_mean_agg = torch.zeros(B, E, 2, device=bond_state.device, dtype=bond_state.dtype)
        bde_mean_count = torch.zeros(B, E, 1, device=bond_state.device, dtype=bond_state.dtype)

        bi_mean = bond_indices.unsqueeze(-1).expand(B, E, 2)
        bde_mean_agg.scatter_add_(1, bi_mean, bde_mean_emb * bond_mask.unsqueeze(-1))
        bde_mean_count.scatter_add_(1, bi_count, masked_count)
        bde_mean_count = bde_mean_count.clamp(min=1)
        bde_mean_val = bde_mean_agg / bde_mean_count
        bde_mean_val = bde_mean_val[:, :n_bonds, :]  # [B, n_bonds, 2]

        return bde_deviation + bde_mean_val

    @classmethod
    def from_npz(cls, npz_path: str, device='cpu') -> 'BDEPredictor':
        """Load model with pre-trained weights from .npz file."""
        data = np.load(npz_path)

        atom_emb = data['atom_embedding/embeddings']
        bond_emb = data['bond_embedding/embeddings']
        num_atom_types, hidden_dim = atom_emb.shape
        num_bond_types = bond_emb.shape[0]
        output_dim = data['bde_mean/embeddings'].shape[1]

        num_messages = sum(1 for k in data.files
                          if 'concat_dense/dense/kernel' in k and k.startswith('edge_update'))

        model = cls(num_atom_types=num_atom_types, num_bond_types=num_bond_types,
                     hidden_dim=hidden_dim, num_messages=num_messages, output_dim=output_dim)

        # Embeddings (no transpose)
        model.atom_embedding.weight.data = torch.from_numpy(atom_emb.copy())
        model.bond_embedding.weight.data = torch.from_numpy(bond_emb.copy())
        model.bde_mean.weight.data = torch.from_numpy(data['bde_mean/embeddings'].copy())

        # Edge updates
        for i in range(num_messages):
            prefix = 'edge_update/concat_dense' if i == 0 else f'edge_update_{i}/concat_dense'
            eu = model.edge_updates[i].concat_dense
            eu.dense1.weight.data = torch.from_numpy(data[f'{prefix}/dense/kernel'].T.copy())
            eu.dense1.bias.data = torch.from_numpy(data[f'{prefix}/dense/bias'].copy())
            eu.dense2.weight.data = torch.from_numpy(data[f'{prefix}/dense_1/kernel'].T.copy())
            eu.dense2.bias.data = torch.from_numpy(data[f'{prefix}/dense_1/bias'].copy())

        # Node updates
        for i in range(num_messages - 1):
            cd_prefix = 'node_update/concat_dense' if i == 0 else f'node_update_{i}/concat_dense'
            nu_prefix = 'node_update' if i == 0 else f'node_update_{i}'
            nu = model.node_updates[i]

            nu.message_dense.dense1.weight.data = torch.from_numpy(data[f'{cd_prefix}/dense_2/kernel'].T.copy())
            nu.message_dense.dense1.bias.data = torch.from_numpy(data[f'{cd_prefix}/dense_2/bias'].copy())
            nu.message_dense.dense2.weight.data = torch.from_numpy(data[f'{cd_prefix}/dense_3/kernel'].T.copy())
            nu.message_dense.dense2.bias.data = torch.from_numpy(data[f'{cd_prefix}/dense_3/bias'].copy())

            nu.update_dense_1.weight.data = torch.from_numpy(data[f'{nu_prefix}/dense/kernel'].T.copy())
            nu.update_dense_1.bias.data = torch.from_numpy(data[f'{nu_prefix}/dense/bias'].copy())
            nu.update_dense_2.weight.data = torch.from_numpy(data[f'{nu_prefix}/dense_1/kernel'].T.copy())
            nu.update_dense_2.bias.data = torch.from_numpy(data[f'{nu_prefix}/dense_1/bias'].copy())

        # Output head
        model.bde_no_mean.weight.data = torch.from_numpy(data['bde_no_mean/kernel'].T.copy())

        model.eval()
        return model.to(device)
