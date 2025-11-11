export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/shared/data1/Users/l1062811/envs/hpc310/include:/shared/data1/Users/l1062811/envs/hpc310/include/python3.10/:/shared/data1/Users/l1062811/envs/hpc310/lib/python3.10/site-packages/numpy/core/include/:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/shared/data1/Users/l1062811/envs/hpc310/lib:$LD_LIBRARY_PATH

export PATH=/home/l1062811/.local/bin:/apps/slurm/aarch64/24.05.4-maple/bin:/apps/slurm/aarch64/24.05.4-maple/sbin:$PATH

for i in $(seq 0 256); do
    echo $i
	python main_hpc.py --experiment maple_qedtest3 --trial $i --init_mol_start $i --iteration 20 --init_mol_path ./Data/zinc_10000.txt --gpu_list 0 --num_init_mol 1 --max_steps_per_episode 20 --reward qed --reward_weight 1.0 --checkpoint ZincQED_1 --eps_threshold 0.0 &
    if (( (i + 1) % 16 == 0 )); then
        wait
    fi
done
