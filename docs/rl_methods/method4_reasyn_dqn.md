# Method 4: RL + ReaSyn DQN

## Purpose

Methods 1-2 had too-large action spaces (jumping problem). Method 3 had too-small action spaces (limited modifiable positions). Method 4 seeks the middle ground: use NVIDIA's ReaSyn model to generate synthesizable analogs of the current molecule as actions. ReaSyn produces molecules that are structurally similar to the input (Tanimoto ≥ 0.3) and come with guaranteed synthesis routes. This should provide a "Goldilocks" action space — local enough for RL to exploit, but broad enough for optimization.

## Main Approach

### Architecture

**State Representation**:
- Morgan fingerprint (4096-bit, radius=2) + step fraction (1-d) = 4097-dim
- Step fraction decreases from 1.0 to 0.0 over the episode, enabling time-aware learning

**Action Space — ReaSyn FastSampler**:
- At each step, ReaSyn generates top_k=20 synthesizable analogs of the current molecule
- Actions = [current_mol (stay)] + [20 ReaSyn analogs] = 21 candidates
- Each analog comes with a synthesis route (building blocks + reaction templates)
- Epsilon-greedy selection: random with probability ε, else argmax Q-value

**FastSampler Configuration (L1a mode)**:
- `use_fp16: true` — half-precision inference
- `max_branch_states: 8` — beam search width
- `skip_editflow: true` — skip diffusion refinement (speed optimization)
- `rxn_product_limit: 1` — one product per reaction
- `max_evolve_steps: 4`, `num_cycles: 1` — reduced from full 8×4=32 cycles
- `expansion_width: 16`, `search_width: 8`
- Time limit: 30s per molecule

**Q-Network — MLPDQN (1.1M parameters)**:
```
Linear(4097, 256) → ReLU
Linear(256, 256) → ReLU
Linear(256, 1)
```

**Training Algorithm**:
- Standard DQN with uniform replay buffer (capacity 50K)
- Bellman update: Q(s) = reward + γ × max Q(s')
- Target network: Polyak averaging τ = 0.995
- Learning rate: 1e-4, Batch size: 64
- Gradient steps per episode: 4
- Epsilon decay: 0.98 per episode (1.0 → ~0.01)

### Incremental Action Cache

To avoid redundant ReaSyn calls:
- Stores (SMILES → [(neighbor_smi, score)]) mappings
- Min cached actions: 5, Max neighbors: 200
- Similarity-based cache lookup: reuse neighbors of fingerprint-similar molecules
- Achieves 90%+ cache hit rate by late training

### Code Location

- `rl/reasyn/rl/train.py` (49.3KB) — ReaSynTrainer main loop
- `rl/reasyn/rl/episode.py` — run_episode (per-molecule episode execution)
- `rl/reasyn/rl/actions.py` — get_reasyn_actions, get_reasyn_actions_full
- `rl/reasyn/rl/replay_buffer.py` — deque-based replay buffer
- `rl/reasyn/rl/action_cache.py` — IncrementalActionCache
- `rl/reasyn/rl/dqn.py` — MLPDQN, make_observation
- `rl/reasyn/sampler/sampler_fast.py` — FastSampler (ReaSyn inference)
- `configs/method/reasyn.yaml`

## Results

**ADRB2 Docking (500 episodes, 64 init molecules)**:

| Metric | Value |
|--------|-------|
| Final mean reward | 3.60 |
| Best docking score | -8.356 (proxy) |
| Oracle calls | 8,729 |
| Cache molecules | 7,311 |
| Cache edges | 118,494 |
| Loss (converged) | 0.0057 |
| Time per episode | 15-23s |

**Learning curve**: Mean reward increases from 0 → 3.6 over 500 episodes; loss converges to 0.005.

**Comparison to GP+ReaSyn baseline** (proxy sEH, seed=42):

| Metric | GP+ReaSyn | ReaSyn-DQN |
|--------|-----------|------------|
| Global best | -12.20 | -8.36 (different target, but pattern holds) |
| Top-10 mean | -11.67 | — |
| Scaffolds | 83 | — |
| Synthesizability | 99.5% | 100% (by construction) |

ReaSyn-DQN consistently underperforms GP+ReaSyn on matched targets.

## Why It Failed

### Problem 1: Init-Mol Dependency

ReaSyn-DQN's performance is heavily dependent on the quality of starting molecules. The agent navigates from each init molecule via ReaSyn analogs, which are constrained to be structurally similar (Tanimoto ≥ 0.3). If the init molecule is in a low-scoring region of chemical space, the agent can only reach nearby (also low-scoring) molecules. It cannot "teleport" to a high-scoring scaffold that is structurally distant.

In contrast, GP maintains a diverse population and uses crossover to combine fragments from different scaffolds, enabling long-range jumps in chemical space while keeping individual mutations local.

### Problem 2: Action Space Too Conservative

ReaSyn analogs are designed to be synthesizable and structurally similar. This means:
- Top-20 analogs typically have Tanimoto 0.3-0.7 to the input molecule
- The agent can only make incremental structural changes per step
- Over 5 steps: total structural distance from init molecule is limited
- Cannot discover novel scaffolds or make the large structural changes needed for high docking scores

### Problem 3: Representation Bottleneck

Morgan fingerprints (4096-bit) are 2D topological counts:
- Two fingerprint-similar molecules can have vastly different docking scores (different 3D conformations)
- Two fingerprint-dissimilar molecules can dock equally well (similar 3D shapes via different scaffolds)
- The Q-function is trained on a blurred landscape where the gradient signal is misleading

### Problem 4: GP+ReaSyn Is Already Near-Optimal

When ReaSyn provides the action space, the optimization landscape is relatively smooth (all molecules are synthesizable analogs). In this regime:
- **GP's strategy**: Random crossover + mutation → ReaSyn projection → score → keep top-K
- **RL's strategy**: Learned Q-function → greedy action selection → score → replay buffer learning

GP's random exploration + hard selection is remarkably effective because:
1. ReaSyn already constrains the space to synthesizable molecules
2. Population-based search maintains diversity naturally
3. Crossover enables long-range exploration that single-molecule RL cannot

### Problem 5: Computational Cost

- ReaSyn inference: 15-30s per molecule (even in L1a fast mode)
- 20 actions × 64 molecules × 5 steps × 500 episodes = millions of potential inference calls
- Cache mitigates this (90%+ hit rate), but initial exploration is expensive
- GP achieves comparable results with simpler, faster oracle calls

### Fundamental Insight

ReaSyn-DQN successfully provides a "Goldilocks" action space — not too jumpy, not too constrained. The local smoothness that RL needs is present. Yet RL still cannot beat GP. This reveals that **the problem is not the action space design — it is that RL's sequential decision-making formulation adds no value** when:
1. The scoring function (oracle) is a black box with no learnable temporal structure
2. Population-based methods already explore efficiently via crossover + selection
3. The oracle budget (10K calls) is insufficient for RL to learn a useful policy

This conclusion motivated the shift to Method 5 (using RL to guide GP, rather than replace it) and ultimately to abandoning RL entirely in favor of accelerating GP+ReaSyn (FastSynOpt).
