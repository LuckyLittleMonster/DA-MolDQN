#!/bin/bash
#SBATCH -p maple
#SBATCH --account=app
#SBATCH --mem=320G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=1
#SBATCH -o test.out
#SBATCH -N 2
#SBATCH --ntasks-per-node 32

export SINGULARITY_CACHEDIR=/shared/data1/Users/${USER}/singularity

srun -u singularity run --nv --bind /shared/data1:/shared/data1 --bind /shared/scratch1:/shared/scratch1 hpc.sif python main_hpc.py --experiment maple_anti --trial 2 --note 'fix opt, 2 node, maple anti, old optimizer' --iteration 2500 --init_mol_path ./Data/anti_400.txt --gpu_list 0 --num_init_mol 256 --starter slurm --max_steps_per_episode 10 --reward bde_ip --eps_decay 0.968 --max_batch_size 512 --cache bde --maintain_OH exist

