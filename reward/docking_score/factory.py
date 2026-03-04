"""Docking scorer factory functions."""

import json
import pathlib


# Project root: reward/docking_score/factory.py -> ../../ (project root)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def make_dock_config(cfg_reward):
    """Return dock config dict for worker processes. None if not needed."""
    if cfg_reward.name == 'qed':
        return None
    target = cfg_reward.get('target', None)
    if target is None:
        if cfg_reward.name in ('dock', 'dock_deprecated'):
            raise ValueError("reward.target must be set for dock mode")
        return None  # multi without dock

    target_dir = _PROJECT_ROOT / 'Data' / 'docking_targets' / target
    config_path = target_dir / 'config.json'
    receptor_path = target_dir / 'receptor.pdbqt'
    if not config_path.exists():
        raise FileNotFoundError(f"Target config not found: {config_path}")

    with open(config_path) as f:
        tgt_cfg = json.load(f)

    return {
        'receptor_pdbqt': str(receptor_path),
        'center_x': cfg_reward.get('center_x') or tgt_cfg['center_x'],
        'center_y': cfg_reward.get('center_y') or tgt_cfg['center_y'],
        'center_z': cfg_reward.get('center_z') or tgt_cfg['center_z'],
        'size_x': cfg_reward.get('size_x', 22.5),
        'size_y': cfg_reward.get('size_y', 22.5),
        'size_z': cfg_reward.get('size_z', 22.5),
    }


def make_dock_scorer(cfg_reward, num_workers=4):
    """Create dock scorer from reward config. Returns None if not needed.

    Uses cfg_reward.scoring_method to select backend:
      'dock' (default) -> UniDockScorer (real Uni-Dock GPU docking)
      'proxy'          -> ProxyDockAdapter (MPNN/SVM/RF proxy models)
    """
    if cfg_reward.name == 'qed':
        return None
    scoring_method = cfg_reward.get('scoring_method', 'dock')

    if scoring_method == 'proxy':
        target = cfg_reward.get('target', None)
        if target is None:
            raise ValueError("reward.target must be set for proxy scoring")
        # Multi-target proxy (e.g. gsk3b_jnk3 -> [gsk3b, jnk3])
        targets = cfg_reward.get('targets', None)
        if targets is not None:
            from .proxy import MultiProxyDockAdapter
            target_list = list(targets)
            adapter = MultiProxyDockAdapter(target_list, device='cuda')
            print(f"  Multi-proxy scorer: targets={target_list}")
            return adapter
        from .proxy import ProxyDockAdapter
        adapter = ProxyDockAdapter(target, device='cuda')
        print(f"  Proxy scorer: target={target}")
        return adapter

    # Default: real Uni-Dock docking
    dock_config = make_dock_config(cfg_reward)
    if dock_config is None:
        return None

    from .unidock import UniDockScorer
    unidock_bin = cfg_reward.get('unidock_bin', None)
    scorer = UniDockScorer(num_workers=num_workers, unidock_bin=unidock_bin,
                           **dock_config)
    target = cfg_reward.target
    print(f"  Dock scorer: target={target}, "
          f"center=({dock_config['center_x']}, "
          f"{dock_config['center_y']}, {dock_config['center_z']})")
    return scorer
