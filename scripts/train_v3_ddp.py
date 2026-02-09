#!/usr/bin/env python
"""
DDP (Distributed Data Parallel) launcher for V3 link predictor training.

Supports both single-node multi-GPU and multi-node training via torchrun or SLURM.

Usage:
    # Single node with torchrun:
    torchrun --nproc_per_node=2 scripts/train_v3_ddp.py --data_dir Data/uspto_full ...

    # Multi-node with SLURM (see scripts/train_v3_ddp.sh):
    srun python scripts/train_v3_ddp.py --data_dir Data/uspto_full ...
"""

import os
import sys
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist


def init_distributed():
    """Initialize distributed training from environment variables.

    Works with both torchrun and SLURM srun.
    """
    # torchrun sets these
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # SLURM sets these
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        print("Warning: No distributed environment detected, running single-GPU")
        return 0, 1, 0

    # Set MASTER_ADDR/MASTER_PORT for SLURM if not set
    if 'MASTER_ADDR' not in os.environ:
        import subprocess
        node_list = os.environ.get('SLURM_NODELIST', 'localhost')
        # Parse the first node from SLURM_NODELIST
        result = subprocess.run(
            ['scontrol', 'show', 'hostnames', node_list],
            capture_output=True, text=True
        )
        master_addr = result.stdout.strip().split('\n')[0]
        os.environ['MASTER_ADDR'] = master_addr

    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"DDP initialized: rank={rank}, world_size={world_size}, "
              f"local_rank={local_rank}, master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description='DDP training for Hypergraph Link Predictor v3')
    parser.add_argument('--data_dir', type=str, default='Data/uspto')
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Per-GPU batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Base learning rate (will be scaled by world_size if --lr_scale is set)')
    parser.add_argument('--lr_scale', action='store_true',
                        help='Scale lr linearly by world_size (lr *= world_size)')
    parser.add_argument('--model_size', type=str, default='medium',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='model_reactions/checkpoints')
    parser.add_argument('--pretrained_encoder', type=str, default=None)
    parser.add_argument('--use_ncn', action='store_true')
    parser.add_argument('--ncn_update_freq', type=int, default=5)
    parser.add_argument('--use_cf_ncn', action='store_true')
    parser.add_argument('--cf_ncn_k', type=int, default=200)
    parser.add_argument('--cf_ncn_update_freq', type=int, default=5)
    parser.add_argument('--knn_cache', type=str, default=None,
                        help='Path to precomputed kNN graph')
    parser.add_argument('--use_topo', action='store_true')
    parser.add_argument('--use_sketch', action='store_true')
    parser.add_argument('--sketch_dim', type=int, default=128)
    parser.add_argument('--use_3d', action='store_true')
    parser.add_argument('--conformer_cache', type=str, default=None)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    args = parser.parse_args()

    # Initialize distributed
    rank, world_size, local_rank = init_distributed()
    ddp = (world_size > 1)

    # Scale learning rate by world_size
    effective_lr = args.lr
    if args.lr_scale and world_size > 1:
        effective_lr = args.lr * world_size
        if rank == 0:
            print(f"LR scaling: {args.lr} * {world_size} = {effective_lr}")

    # Import and run training
    from model_reactions.link_prediction.hypergraph_link_predictor_v3 import train_v3

    train_v3(
        data_dir=args.data_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=effective_lr,
        model_size=args.model_size,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        pretrained_encoder=args.pretrained_encoder,
        use_3d=args.use_3d,
        conformer_cache=args.conformer_cache,
        use_ncn=args.use_ncn,
        ncn_update_freq=args.ncn_update_freq,
        use_topo=args.use_topo,
        use_sketch=args.use_sketch,
        sketch_dim=args.sketch_dim,
        use_cf_ncn=args.use_cf_ncn,
        cf_ncn_k=args.cf_ncn_k,
        cf_ncn_update_freq=args.cf_ncn_update_freq,
        knn_cache=args.knn_cache,
        early_stop_patience=args.early_stop_patience,
        ddp=ddp,
        local_rank=local_rank,
    )

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
