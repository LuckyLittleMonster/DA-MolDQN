Install Guide

1. Build rdkit from source code. (https://www.rdkit.org/docs/Install.html)
2. Make sure to set up RDBASE.
export RDBASE=/home/hqin/git/rdkit-Release_2025_03_5 
export PYTHONPATH=$RDBASE 
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH

(rl4) ➜  rdkit-Release_2025_03_5 python
Python 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import rdkit
>>> rdkit.__version__
'2025.03.5'
>>> rdkit.__file__
'/home/hqin/git/rdkit-Release_2025_03_5/rdkit/__init__.py'
>>> 

3. Compile c++ code:
cd src && mkdir build && cd build
cmake ..
make

4. Install packages
pip install tqdm seaborn nfp
# install alfabet (BDE-db2 is an extended version)
noglob pip install tensorflow-addons[tensorflow]
git clone https://github.com/patonlab/BDE-db2.git


python main_hpc.py --experiment test --trial 1 --init_mol_start 0 --iteration 2000 --init_mol_path  ./Data/anti_pub.txt --gpu_list 0 --num_init_mol 1 --max_steps_per_episode 10 --reward qed --eps_decay 0.968 --max_batch_size 128 --cache bde --maintain_OH exist --checkpoint trial_573 --eps_threshold 0.5 --init_method=file://$PWD/tem/sharedfile --starter fork