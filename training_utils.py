"""Shared training utilities for Route-DQN and ReaSyn-DQN."""

import json
import os
import pathlib
import pickle
import random

import torch


def resolve_device(cfg):
    """Resolve torch device from config (cfg.device + cfg.gpu)."""
    device_str = cfg.device
    if device_str == 'cuda':
        if torch.cuda.is_available():
            gpu = cfg.get('gpu', 0)
            device_str = f'cuda:{gpu}'
        else:
            device_str = 'cpu'
    return torch.device(device_str)


def setup_experiment_dirs(project_root, cfg):
    """Create Experiments/models/ dirs, return (exp_dir, model_save_dir, prefix)."""
    exp_dir = pathlib.Path(project_root) / 'Experiments'
    exp_dir.mkdir(exist_ok=True)
    model_save_dir = exp_dir / 'models'
    model_save_dir.mkdir(exist_ok=True)
    prefix = f"{cfg.exp_name}_{cfg.trial}"
    return exp_dir, model_save_dir, prefix


def load_molecules(cfg, project_root):
    """Load SMILES list from mol_json / init_mol_path / init_mol config.

    Handles offset and num_molecules. Returns list of SMILES.
    """
    if cfg.get('mol_json'):
        path = pathlib.Path(cfg.mol_json)
        if not path.is_absolute():
            path = pathlib.Path(project_root) / path
        with open(path) as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            smiles = [d['smiles'] for d in data]
        else:
            smiles = list(data)
    elif cfg.get('init_mol_path'):
        with open(cfg.init_mol_path) as f:
            smiles = [line.strip() for line in f if line.strip()]
    else:
        smiles = list(cfg.get('init_mol', []))

    offset = cfg.get('init_mol_offset', 0)
    if offset > 0:
        smiles = smiles[offset:]

    num = cfg.get('num_molecules')
    if num is not None:
        if num <= len(smiles):
            smiles = random.sample(smiles, num)
        else:
            smiles = [smiles[i % len(smiles)] for i in range(num)]

    return smiles


# --- Checkpoint utilities (merged from checkpoint_utils.py) ---

def save_checkpoint(path, **kwargs):
    """Atomic checkpoint save."""
    tmp = str(path) + '.tmp'
    torch.save(kwargs, tmp)
    os.replace(tmp, str(path))


def save_pickle(path, data):
    """Atomic pickle save."""
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, str(path))
