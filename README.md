# DA-MolDQN (dev 2.0)

RL-based molecular optimization. The codebase is organized as:

- `csrc/` — C++ extension (`cenv`): action enumeration + incremental Morgan fingerprint.
- `src/` — Python package: `trainer`, `agent`, `environment`, `launch/` (dist launchers),
  `models/`, `reward/` (qed / plogp / sa / bde / ip), `persistence/` (Recorder).
- `configs/` — Hydra config (`launcher/`, `reward/`, `preset/` groups).
- `train.py` / `finetune.py` / `testing.py` — entry points.

## 1. Environment

No source build of RDKit and no RDBASE/boost-python setup are required — everything
comes from conda-forge (the C++ extension links conda's `librdkit-dev`).

```bash
conda create -n da-moldqn -c conda-forge \
  python=3.11 rdkit librdkit-dev libboost-devel libboost-python-devel \
  pytorch numpy scikit-learn pandas six tqdm psutil hydra-core omegaconf \
  cmake make cxx-compiler
conda activate da-moldqn
```

## 2. Build the C++ extension

```bash
cd csrc && cmake -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX && cmake --build build -j4
cp build/cenv.so cenv.so   # import as csrc.cenv
cd ..
```

## 3. Initialize submodules (optional, for BDE-db2)

```bash
git submodule update --init --recursive
```

## 4. Run

Hydra drives configuration. Switch reward / launcher via config groups, and override
any key on the command line.

```bash
# Train (single process)
python train.py reward=qed init_mol_path=./Data/anti_pub.txt \
  num_init_mol=1 max_steps_per_episode=10 iteration=2000 \
  experiment=test trial=1

# Multi-process (fork)
python train.py launcher=fork mp_world_size=2 reward=qed init_mol='[CCO,CCN]' num_init_mol=2

# Finetune from a checkpoint
python finetune.py reward=qed checkpoint=test_1 experiment=test trial=2

# Testing (no training; eps=0; generate best molecules)
python testing.py reward=qed checkpoint=test_1 experiment=test trial=2

# Smoke run
python train.py reward=qed +preset=smoke init_mol='[CCO]'
```

Reward groups: `reward={qed,bde_ip,plogp}`. Launcher groups:
`launcher={single,torchrun,slurm,fork}`.

## Output

Each run writes everything under a single folder `Experiments/{experiment}_{trial}/`:

```
Experiments/{experiment}_{trial}/
  config.yaml                       # resolved run config
  checkpoints/model_dqn.pth  model_target_dqn.pth
  {experiment}_{trial}.pickle.gz    # all ranks' metrics + paths, merged + gzip-compressed
```
