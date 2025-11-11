# Install Guide

## 1. Build RDKit from source code

Follow the official installation guide: https://www.rdkit.org/docs/Install.html

## 2. Set up RDBASE environment variables

```bash
export RDBASE=/path/to/rdkit/rdkit-Release_2025_03_5 
export PYTHONPATH=$RDBASE 
export LD_LIBRARY_PATH=$RDBASE/lib:$LD_LIBRARY_PATH
```

Verify the installation:

```python
>>> import rdkit
>>> rdkit.__version__
'2025.03.5'
>>> rdkit.__file__
'/path/to/rdkit/rdkit-Release_2025_03_5/rdkit/__init__.py'
>>> 
```

## 3. Compile C++ code

```bash
cd src && mkdir build && cd build
cmake ..
make
```

## 4. Install packages

```bash
pip install tqdm seaborn nfp
# install alfabet (BDE-db2 is an extended version)
noglob pip install tensorflow-addons[tensorflow]
```

## 5. Initialize git submodules

```bash
git submodule update --init --recursive
```

This will initialize the BDE-db2 submodule.

## 6. Run the main script

```bash
python main_hpc.py --experiment test --trial 1 --init_mol_start 0 --iteration 2000 --init_mol_path  ./Data/anti_pub.txt --gpu_list 0 --num_init_mol 1 --max_steps_per_episode 10 --reward qed --eps_decay 0.968 --max_batch_size 128 --cache bde --maintain_OH exist --checkpoint trial_573 --eps_threshold 0.5 --init_method=file://$PWD/tem/sharedfile --starter fork
```
