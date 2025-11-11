export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/shared/data1/Users/l1062811/envs/hpc310/include:/shared/data1/Users/l1062811/envs/hpc310/include/python3.10/:/shared/data1/Users/l1062811/envs/hpc310/lib/python3.10/site-packages/numpy/core/include/:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/shared/data1/Users/l1062811/envs/hpc310/lib:$LD_LIBRARY_PATH

export PATH=/home/l1062811/.local/bin:/apps/slurm/aarch64/24.05.4-maple/bin:/apps/slurm/aarch64/24.05.4-maple/sbin:$PATH

#python bde_predictor.py
#torchrun --nnodes=1 --nproc-per-node=1 main_hpc.py --experiment maple_5ft --trial 0 --iteration 2000 --init_mol_path ./Data/anti_400.txt --gpu_list 0 --num_init_mol 1 --starter torchrun --max_steps_per_episode 10 --reward bde_ip --eps_decay 0.968 --max_batch_size 128 --cache bde --maintain_OH exist --checkpoint maple_5 --eps_threshold 0.5
for i in $(seq 0 16); do
    echo $i
	python main_hpc.py --experiment maple_qedft3 --trial $i --init_mol_start $i --iteration 20000 --init_mol_path ./Data/anti_400.txt --gpu_list 0 --num_init_mol 1 --max_steps_per_episode 10 --reward qed --eps_decay 0.996 --reward_weight 1.0 --checkpoint maple_qed_2 --eps_threshold 0.5 &
    if (( (i + 1) % 16 == 0 )); then
        wait
    fi
done
