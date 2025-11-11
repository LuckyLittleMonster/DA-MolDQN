run(){
	torchrun --nproc-per-node=16 main_dist.py --trial 700$1 --iteration 80000 --max_steps_per_episode 10 --init_mol_path ./Data/anti_400.txt --discount_factor 1.0 --backend gloo --gpu_list 0 1 2 3 --torch_num_threads 1 --num_init_mol 16 --record_path --update_episodes 1 --torchrun --init_mol_start $1 1>>log/700$1.log 2>>log/700$1.err
	sleep 20
	# echo "":
}
runS(){
	srun --error=log/1600$1_%t.err -N 1 --ntasks-per-node=16 -p cypress_a100 --gres=gpu:4 -l -u python main_dist.py --trial 1600$1 --iteration 80000 --max_steps_per_episode 10 --init_mol_path ./Data/anti_400.txt --discount_factor 1.0 --backend gloo --gpu_list 0 1 2 3 --torch_num_threads 1 --num_init_mol 16 --record_path --update_episodes 1 --slurm
}
for (( i = $1; i < $2 ; i+=64 )); do
	# echo $(expr $i + 1)
	runS $i &
	runS $(expr $i + 16) &
	runS $(expr $i + 32) &
	runS $(expr $i + 48) &
	wait
	sleep 1
done
