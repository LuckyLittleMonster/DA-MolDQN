run(){
    torchrun --nproc-per-node=16 main_dist.py --trial 700$1 --iteration 80000 --max_steps_per_episode 10 --init_mol_path ./Data/anti_400.txt --discount_factor 1.0 --backend gloo --gpu_list 0 1 2 3 --torch_num_threads 1 --num_init_mol 16 --record_path --update_episodes 1 --torchrun --init_mol_start $1 1>>log/700$1.log 2>>log/700$1.err
    sleep 20
    # echo ""
}

for (( i = $1; i < $2 ; i+=16 )); do
    # echo $(expr $i + 1)
    run $i &
    # run $i 0 25640 &
    # run $(expr $i + 1) 1 25641 &
    # run $(expr $i + 2) 2 25642 &
    # run $(expr $i + 3) 3 25643 &
    wait
done