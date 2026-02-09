#!/usr/bin/env python
"""
V3 FULL LR Tuning - Job C: lr=1e-4, ReduceLROnPlateau (patience=3).

This wrapper patches the scheduler in train_v3() to use ReduceLROnPlateau
instead of the default CosineAnnealingWarmRestarts.
"""
import sys
import os
import types
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_reactions.link_prediction.hypergraph_link_predictor_v3 as v3_module

# Monkey-patch: replace the scheduler creation in train_v3
# We intercept SequentialLR creation to replace it with ReduceLROnPlateau
_original_SequentialLR = torch.optim.lr_scheduler.SequentialLR

class ReduceLROnPlateauWrapper:
    """Wraps ReduceLROnPlateau to match the .step() call pattern of SequentialLR.

    train_v3() calls scheduler.step() once per batch (step-level scheduling).
    We convert this to epoch-level by tracking steps and only stepping once per epoch.
    """
    def __init__(self, optimizer, patience=3, factor=0.5, min_lr=1e-6):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=factor,
            min_lr=min_lr,
        )
        self._step_count = 0
        self._epoch_val_auc = 0.0
        self._optimizer = optimizer

    def step(self, *args, **kwargs):
        # No-op for step-level calls (train_v3 calls this per batch)
        self._step_count += 1

    def step_epoch(self, val_auc):
        """Call this at epoch level with validation metric."""
        self.scheduler.step(val_auc)

    def get_last_lr(self):
        return [pg['lr'] for pg in self._optimizer.param_groups]


_plateau_scheduler = [None]

def _patched_SequentialLR(optimizer, schedulers, milestones):
    """Replace SequentialLR with ReduceLROnPlateau."""
    sched = ReduceLROnPlateauWrapper(optimizer, patience=3, factor=0.5, min_lr=1e-6)
    _plateau_scheduler[0] = sched
    return sched

# Apply the patch
torch.optim.lr_scheduler.SequentialLR = _patched_SequentialLR

# Also patch the evaluate function to feed val_auc to the plateau scheduler
_original_evaluate = v3_module.evaluate

def _patched_evaluate(*args, **kwargs):
    metrics = _original_evaluate(*args, **kwargs)
    # Feed val AUC to the plateau scheduler
    if _plateau_scheduler[0] is not None and 'roc_auc' in metrics:
        _plateau_scheduler[0].step_epoch(metrics['roc_auc'])
    return metrics

v3_module.evaluate = _patched_evaluate

if __name__ == '__main__':
    print("=" * 60)
    print("Job C: lr=1e-4, ReduceLROnPlateau (patience=3, factor=0.5)")
    print("=" * 60)

    v3_module.train_v3(
        data_dir='Data/uspto_full',
        n_epochs=80,
        batch_size=512,
        lr=1e-4,
        model_size='medium',
        device='auto',
        num_workers=8,
        save_dir='model_reactions/checkpoints/full_C',
        pretrained_encoder='model_reactions/checkpoints/pretrained_encoder.pt',
        use_cf_ncn=True,
        cf_ncn_k=200,
        knn_cache='Data/precomputed/knn_full_k200.pkl',
        use_ncn=True,
        early_stop_patience=15,
    )
