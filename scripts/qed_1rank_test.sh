#!/bin/bash
set -euo pipefail
cd /shared/data1/Users/l1062811/git/DA-MolDQN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/l1062811/data/envs/rl4/bin/python -u main_hpc.py \
    +experiment=ablation_rw \
    reward=qed \
    'reward.reward_weight=[1.0]' \
    trial=3014 \
    iteration=5000 \
    max_steps_per_episode=20 \
    eps_threshold=1.0 \
    eps_decay=0.968 \
    discount_factor=1.0 \
    init_mol_path=Data/zinc_10000.txt \
    num_init_mol=256 \
    starter=null \
    backend=gloo \
    max_batch_size=32768 \
    min_batch_size=128 \
    maintain_OH=null \
    'cache=[]' \
    dqn.replay_buffer_size=256000 \
    save_reward_freq=25 \
    save_model_freq=250 \
    save_path_freq=250
