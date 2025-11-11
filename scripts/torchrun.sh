export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/shared/data1/Users/l1062811/envs/hpc310/include:/shared/data1/Users/l1062811/envs/hpc310/include/python3.10/:/shared/data1/Users/l1062811/envs/hpc310/lib/python3.10/site-packages/numpy/core/include/:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/shared/data1/Users/l1062811/envs/hpc310/lib:$LD_LIBRARY_PATH

export PATH=/home/l1062811/.local/bin:/apps/slurm/aarch64/24.05.4-maple/bin:/apps/slurm/aarch64/24.05.4-maple/sbin:$PATH

#python bde_predictor.py
#torchrun --nnodes=1 --nproc-per-node=32 main_hpc.py --experiment maple --trial 5 --iteration 2500 --init_mol_path ./Data/anti_400.txt --gpu_list 0 --num_init_mol 256 --starter torchrun --max_steps_per_episode 10 --reward bde_ip --eps_decay 0.968 --max_batch_size 512 --cache bde --maintain_OH exist
#torchrun --nnodes=1 --nproc-per-node=32 main_hpc.py --experiment maple_qed --trial 2 --iteration 2500 --init_mol_path ./Data/anti_400.txt --gpu_list 0 --num_init_mol 256 --starter torchrun --max_steps_per_episode 10 --reward qed --eps_decay 0.968 --max_batch_size 512 --reward_weight 1.0
#torchrun --nnodes=1 --nproc-per-node=32 main_hpc.py --experiment zinc_qed --trial 1 --iteration 2500 --init_mol_path ./Data/zinc_10000.txt --gpu_list 0 --num_init_mol 256 --starter torchrun --max_steps_per_episode 10 --reward qed --eps_decay 0.968 --max_batch_size 512 --reward_weight 1.0
#torchrun --nnodes=1 --nproc-per-node=32 main_hpc.py --experiment zinc_qed --trial 3 --note 'hpc' --iteration 5000 --init_mol_path ./Data/zinc_10000.txt --gpu_list 0 --num_init_mol 256 --starter torchrun --max_steps_per_episode 20 --reward qed --eps_decay 0.968 --max_batch_size 512 --reward_weight 1.0
torchrun --nnodes=1 --nproc-per-node=32 main_hpc.py --experiment zinc_qed --trial 6 --note 'hpc' --iteration 5000 --init_mol_path ./Data/zinc_10000.txt --gpu_list 0 --num_init_mol 256 --starter torchrun --max_steps_per_episode 20 --reward qed --eps_decay 0.968 --max_batch_size 512 --reward_weight 1.0
