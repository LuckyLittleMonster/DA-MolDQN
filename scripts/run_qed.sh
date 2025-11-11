#!/bin/bash
#SBATCH -p maple
#SBATCH --account=app
#SBATCH --mem=420G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=1
#SBATCH -o test.out
#SBATCH -N 1
#SBATCH --ntasks-per-node 32

export SINGULARITY_CACHEDIR=/shared/data1/Users/${USER}/singularity

srun -u singularity run --nv --bind /shared/data1:/shared/data1 --bind /shared/scratch1:/shared/scratch1 hpc.sif python main_hpc.py --experiment zinc_qed --trial 10 --note 'fixed optimizer' --iteration 5000 --init_mol_path ./Data/zinc_10000.txt --gpu_list 0 --num_init_mol 256 --starter slurm --max_steps_per_episode 20 --reward qed --eps_decay 0.968 --max_batch_size 512 --reward_weight 1.0
#srun -u singularity run --nv --bind /shared/data1:/shared/data1 --bind /shared/scratch1:/shared/scratch1 hpc.sif python main_hpc.py --experiment zinc_qed --trial 8 --note 'hpc 2 nodes 64 processes' --iteration 20000 --init_mol_path ./Data/zinc_10000.txt --gpu_list 0 --num_init_mol 256 --starter slurm --max_steps_per_episode 20 --reward qed --eps_decay 0.992 --max_batch_size 512 --reward_weight 1.0

