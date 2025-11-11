#!/bin/bash
#SBATCH -p maple
#SBATCH --account=app
#SBATCH --mem=420G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=64
#SBATCH -o test.out
#SBATCH -N 2

export SINGULARITY_CACHEDIR=/shared/data1/Users/${USER}/singularity
singularity run --nv --bind /shared/data1:/shared/data1 --bind /shared/scratch1:/shared/scratch1 /shared/data1/Projects/CSE_HPC/apps/singularity/aarch64/nvidia_tensorflow/tensorflow-20.4-tf2-py3.sif bash torchrun.sh
#singularity run --nv --bind /shared/data1:/shared/data1 --bind /shared/scratch1:/shared/scratch1 /shared/data1/Projects/CSE_HPC/apps/singularity/aarch64/nvidia_tensorflow/tensorflow-20.4-tf2-py3.sif bash qed_ft.sh
#singularity run --nv --bind /shared/data1:/shared/data1 --bind /shared/scratch1:/shared/scratch1 /shared/data1/Projects/CSE_HPC/apps/singularity/aarch64/nvidia_tensorflow/tensorflow-20.4-tf2-py3.sif bash maple_test.sh


#singularity run --nv --bind /shared/data1:/shared/data1 --bind /shared/scratch1:/shared/scratch1 /shared/data1/Projects/CSE_HPC/apps/singularity/aarch64/nvidia_tensorflow/tensorflow-20.4-tf2-py3.sif bash qed_test.sh
