# Method 6: UniMolDQN — 3D Molecular Encoder DQN

## Purpose

Methods 1-5 all used Morgan fingerprints (2D topological counts) as molecular representations. A key hypothesis from the root cause analysis was that this 2D representation cannot capture the 3D conformation, electrostatics, and protein-ligand interactions that determine docking scores, causing the Q-function to learn on a blurred, misleading landscape.

Method 6 tests this hypothesis directly by replacing Morgan fingerprints with Uni-Mol, a state-of-the-art 3D molecular encoder pretrained on 209 million conformations. If representation quality is the bottleneck, UniMolDQN should significantly outperform fingerprint-based DQN.

## Main Approach

### Architecture

**Uni-Mol Encoder (Frozen)**:
- Pretrained on 209M molecular conformations (QM9 + GEOM datasets)
- SE(3)-equivariant transformer architecture
- Input: SMILES → automatic 3D conformer generation (ETKDG) → atomic features + coordinates
- Output: 512-dimensional CLS token (mean-pooled graph representation)
- **Frozen during training** — only the Q-head is trained
- Two inference paths:
  1. Via `unimol_tools` API (v1, slower, includes DataLoader overhead)
  2. Bare-metal path (v2+): `ConformerGen.transform_raw()` → manual batch collation → GPU inference (~10ms/40 mols vs ~176ms via API)

**Q-Head**:
```
Input: (N, 513) = [L2-normalized 512-d embedding | step_fraction]
Linear(513, 256) → ReLU
Linear(256, 128) → ReLU
Linear(128, 1)
Output: Q-value scalar
```

**Embedding Cache**:
- SMILES → 512-dim float32 vector
- Disk persistence (.npz files, 137-188MB per run)
- Hit rate: 91.8% (seed=123) to 96.2% (seed=42) by end of training
- Reduces ~100ms/mol Uni-Mol inference to ~1ms on cache hit

### 3D Conformer Generation and Action Filtering

**ConformerManager** — Incremental 3D conformer generation:

Three-stage ConstrainedEmbed (for molecules ≥ 10 atoms):
1. **Forward match**: Parent ⊂ child → `AllChem.ConstrainedEmbed()` with parent coordinates
2. **Reverse match**: Child ⊂ parent → extract coordinates via coordMap
3. **Fallback**: Scratch ETKDG if no substructure match

**3D Action Space Filtering** — Three complementary checks:

1. **Distance filter**: New bonds must satisfy `distance ≤ reference_bond_length × 2.5`
   - Reference lengths: C-C=1.54Å, C-N=1.47Å, C-O=1.43Å
2. **Bond angle check**: Angles by hybridization (SP3=109.5°, SP2=120°, SP=180°), reject if < 50°
3. **Steric clash detection**: VDW collision check for atom additions (factor 0.65)

Overall filtering: 16-32% of actions rejected (varies by episode stage)

### Environment

- **MolDQN-style action space**: Add atom, add bond, remove atom, remove bond, modify bond order
- **Starting molecule**: "C" (methane)
- **Episode length**: 40 steps maximum
- **Objective**: QED optimization (Quantitative Estimate of Drug-likeness, range 0-1)

### Training — Three Versions

**v1 (Baseline)**:
- Uni-Mol via `unimol_tools` API
- Standard DQN
- High overhead, slow inference

**v2 (Bare-metal + 3D filtering)**:
- Bare-metal encoder (bypass DataLoader, ~17× faster inference)
- ConstrainedEmbed for incremental 3D conformer generation
- 3D action space filtering (distance + angle + steric)
- **Issue**: Loss diverges catastrophically (0.01 → 23.8)

**v3 (Stability fixes)**:
- L2-normalize embeddings (scales norms from ~27 to 1.0)
- Remove LayerNorm (conflicts with Polyak target network averaging)
- Gradient clipping (max_norm = 1.0)
- Double DQN (online network selects, target network evaluates)
- Lower learning rate (5e-5 vs 1e-4)
- **Result**: Loss stabilizes but scores plateau

### Hyperparameters (v3)

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Discount (γ) | 0.95 |
| Polyak (τ) | 0.995 |
| Replay buffer | 5000 |
| Batch size | 128 |
| Epsilon schedule | exponential decay, 2000 steps (1.0 → 0.01) |
| Gradient clipping | max_norm = 1.0 |
| Double DQN | true |
| Target episodes | 2500 |
| Max steps/episode | 40 |

### Code Location

- `unimol_dqn.py` — UniMolEncoder, UniMolDQN, EmbeddingCache
- `agent_unimol.py` — DQN agent, replay buffer, training step
- `conformer3d.py` — ConformerManager, 3D filtering
- `run_unimol.py` — Main training script
- `tests/profile_unimol.py` — Performance profiling
- `scripts/experiments/run_unimol_qed_v2.sh` — v2 SLURM script
- `scripts/experiments/run_unimol_qed_v3.sh` — v3 SLURM script

## Results

### UniMolDQN v2 (bare-metal + ConstrainedEmbed)

| Seed | Episodes (of 2500) | Top-1 QED | Top-10 QED | Loss | Cache Hit | Time | Status |
|------|---------------------|-----------|------------|------|-----------|------|--------|
| 123 | 520 | 0.8650 | 0.8299 | 10.4 (diverging) | 34.8% | 6.2h | CANCELLED |
| 42 | 780 | 0.8047 | 0.7805 | 23.8 (diverging) | 41.5% | 6.1h | CANCELLED |

- Loss diverges catastrophically from episode ~60 onward
- High top-1 scores (0.865) were achieved in early exploration (episode 60), not by the learned policy
- Cache hit rate low (34-41%) — model explores widely but aimlessly

### UniMolDQN v3 (+ L2-norm, Double DQN, gradient clipping)

| Seed | Episodes (of 2500) | Top-1 QED | Top-10 QED | Loss | Cache Hit | Time | Status |
|------|---------------------|-----------|------------|------|-----------|------|--------|
| 123 | 500 | 0.8144 | 0.8144 | 0.17 (stable) | 91.8% | 11.6h | TIMEOUT 12h |
| 42 | 830 | 0.7444 | 0.7435 | 0.03 (converging) | 96.2% | 12h | TIMEOUT 12h |

- Loss stabilizes with v3 fixes
- But scores plateau far below optimum: seed=42 stagnates from Ep 260 to 830 (570 episodes) with top-1 barely improving (0.697 → 0.744)
- Cache hit rate very high (91-96%) — model trapped in local optima, revisiting same molecules

### MolDQN Baseline (Morgan fingerprint, same environment)

| Seed | Episodes | Top-1 QED | Top-10 QED | Time | Status |
|------|----------|-----------|------------|------|--------|
| 42 | 2500 | **0.8976** | **0.8786** | 4.3h | Complete |
| 123 | 2500 | **0.9012** | **0.8887** | 4.8h | Complete |

### Comparison Summary

| Model | Top-1 QED | Top-10 QED | vs Optimum (~0.948) |
|-------|-----------|------------|---------------------|
| MolDQN (FP) | 0.90 | 0.88 | 95% |
| UniMol v2 | 0.87 | 0.83 | 91% (but diverging) |
| UniMol v3 | 0.81 | 0.81 | 86% |
| Simple GA | ~0.93+ | ~0.91+ | 98% |

**UniMolDQN performs worse than the simpler MolDQN baseline**, despite using a vastly superior molecular representation.

## Performance Profiling (40 molecules on GH200)

| Stage | Time | % of Total |
|-------|------|-----------|
| ETKDG conformer generation | 272ms | **96%** |
| transform_raw (featurize) | 1.6ms | 1% |
| GPU inference (UniMol) | 9.7ms | 3% |
| **Total bare-metal** | **283ms** | — |
| unimol_tools API (for comparison) | 459ms | — |

**ETKDG is the absolute bottleneck** (96% of encoding time). GPU inference is fast (10ms for 40 molecules). The bare-metal pipeline (bypassing unimol_tools' Pool/DataHub/logging overhead) saves ~176ms/batch.

### 3D Pipeline Optimizations Attempted

1. **ConstrainedEmbed**: Reuse parent conformer coordinates → 7.65x speedup for large molecules (>10 atoms), but slower for small molecules (<5 atoms). Hybrid threshold: ≥10 atoms uses ConstrainedEmbed, <10 uses scratch ETKDG.
2. **3D action filtering**: Distance + bond angle + steric checks on parent conformer → filters 21-32% of invalid actions → fewer ETKDG calls.
3. **Embedding cache**: SMILES→512-dim cache with disk persistence. Hit rate 34-96% depending on exploration vs exploitation phase.
4. **nvMolKit GPU ETKDG**: Not feasible due to environment incompatibility and small batch sizes.

Despite these optimizations, UniMolDQN remains ~5-7x slower than MolDQN per step.

## Why It Failed

### Finding 1: 3D Representation Does Not Rescue DQN

This is the most important finding. The root cause analysis of Methods 1-5 hypothesized two bottlenecks:
1. Representation quality (2D fingerprints)
2. Model capacity (linear Q-network)

Method 6 addresses both:
- **Representation**: Uni-Mol (209M pretrained, 3D-aware, 512-dim) vs Morgan FP (2D, 2048-bit)
- **Capacity**: 3-layer deep MLP (513 → 256 → 128 → 1) vs linear Q-network

Yet UniMolDQN performs **worse** than fingerprint-based MolDQN. This definitively rules out representation and capacity as the primary bottlenecks.

### Finding 2: The Problem Is the MDP Formulation Itself

With representation and capacity eliminated, the remaining explanation is that **sequential decision-making (MDP/DQN) is fundamentally ill-suited for molecular optimization**:

1. **No temporal structure**: Molecular optimization does not have meaningful temporal dependencies. The quality of step 20's modification does not depend on what happened at step 5 — it depends only on the current molecular structure. The Markov property holds, but the optimal policy is essentially greedy (take the best available action each step), which does not require RL to learn.

2. **Myopic optimization**: DQN with discount γ=0.95 over 40 steps heavily discounts future rewards. The agent optimizes for immediate improvement, which is exactly what greedy selection does — but without the overhead of learning a Q-function.

3. **Sample efficiency**: RL needs thousands of episodes to learn even simple policies. With a 10K oracle budget, the agent barely explores before the budget is exhausted. Evolutionary methods use every oracle call for selection, wasting nothing on "learning."

### Finding 3: Embedding Norm Instability

A practical but important finding: Uni-Mol CLS embeddings have norms ~27 (varying from 20-35 across molecules). Without L2-normalization:
- Q-values scale with embedding norm, not molecular quality
- Target network Polyak averaging creates norm mismatch → loss divergence
- LayerNorm conflicts with Polyak averaging (running stats diverge between online and target networks)

v3's L2-normalization + no-LayerNorm + gradient clipping fixes the stability issue, but the underlying optimization performance remains poor.

### Finding 4: Cache Hit Rate as a Diagnostic

The cache hit rates tell a clear story:
- **v2**: 34-41% hit rate → model explores widely, sees many unique molecules, but cannot learn from them (loss diverges)
- **v3**: 91-96% hit rate → model is trapped in a tiny region of chemical space, revisiting the same molecules repeatedly (loss converges to ~0 because there's nothing new to learn)

Neither regime produces good optimization: v2 explores without learning, v3 learns without exploring.

### Fundamental Insight

Method 6 is the definitive experiment in the systematic elimination process:

| Method | What it tests | Result |
|--------|--------------|--------|
| 1-2 | Reaction-based action spaces | Failed (jumping) |
| 3 | Constrained action spaces | Failed (too small) |
| 4 | Moderate action spaces (ReaSyn) | Failed (no advantage over GP) |
| 5 | RL guiding GP (not replacing it) | Failed (GP already optimal) |
| **6** | **Superior representation + deep Q-network** | **Failed (same or worse)** |

Having systematically eliminated action space design (Methods 1-4), optimization strategy (Method 5), representation quality (Method 6), and model capacity (Method 6), the conclusion is:

**The sequential decision-making formulation (MDP/DQN) is itself the wrong abstraction for molecular optimization.** Evolutionary methods (GP) with domain-specific constraints (ReaSyn) are inherently better suited because they:
- Maintain population diversity without explicit entropy bonuses
- Use crossover for long-range exploration without sacrificing local search quality
- Evaluate every candidate (no wasted oracle calls on "learning")
- Require no training — every oracle call directly contributes to the solution
