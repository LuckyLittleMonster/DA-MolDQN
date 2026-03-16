# RL Optimizing Synthesizable Molecules: Retrospective & Open Problems

## Core Hypothesis

If a molecule has good chemical properties, structurally similar molecules should also have good properties (local smoothness assumption). RL can exploit this by iteratively modifying existing molecules, which should be more efficient than de novo generation.

## Methods Attempted

### 1. RL + Hypergraph AIO
- **Approach**: Predict reactants and reaction products for state molecule
- **Problem**: Molecular "jumping" - mutations produce very different molecules, breaking RL's local optimization advantage
- **Status**: Failed

### 2. Template/Model Reactant Prediction + T5v2 Product Prediction
- **Approach**: Use templates or model to predict reactants, ReactionT5v2 to predict products
- **Problem**: Same jumping issue - large structural changes between steps destroy RL's advantage
- **Status**: Failed

### 3. RL Route Optimization (`rl/route/`)
- **Approach**: Given existing synthesis route, RL selects which reaction/BB to replace, then reconstructs the full route
- **Problem**: Dependent on init mol quality; mean scores significantly below GP
- **Status**: Partially working but underperforms

### 4. RL + ReaSyn (`rl/reasyn/`)
- **Approach**: DQN navigates synthesizable space using ReaSyn-generated analogs as actions
- **Problem**: Dependent on init mol quality; top scores approach GP with good init but mean is always lower
- **Status**: Partially working but underperforms

### 5. RL-Guided GP + ReaSyn (SynOpt)
- **Approach**: Dual-layer RL (Strategy + Operator) guides GA operator selection, ReaSyn projects offspring to synthesizable space
- **Problem**: GP+ReaSyn without RL performs equally well or better. RL adds exploitation pressure that causes premature convergence without improving results
- **Status**: Failed - RL guidance provides no benefit

### 6. UniMolDQN — 3D Molecular Encoder DQN
- **Approach**: Replace Morgan FP with Uni-Mol (frozen, 209M pretrained, 512-dim 3D-aware encoder). Deep Q-head (3-layer MLP) replaces linear Q.
- **Motivation**: Test whether 3D representation + model capacity can rescue DQN.
- **Problem**: Worse than fingerprint-based MolDQN on QED (top-1 0.81 vs 0.90). Loss divergence in v2; v3 stabilizes but plateaus in local optima.
- **Status**: Failed - 3D representation does not rescue DQN

## Common Failure Patterns

1. **Init mol dependency**: All RL methods require good starting molecules. Cannot discover novel scaffolds independently.
2. **Poor mean scores**: Even when top scores approach GP, the distribution of good molecules is narrow.
3. **Jumping problem (methods 1,2)**: Large structural changes between RL steps break the local smoothness assumption that makes RL valuable.
4. **Premature convergence (method 5)**: RL policy collapses entropy, reducing exploration below random GP baseline.
5. **GP already sufficient**: For methods where ReaSyn provides the action space, GP's random search + selection is already near-optimal.
6. **Q-value instability with continuous representations (method 6)**: UniMol embeddings (L2 norm ~27, range ±4.7) cause Q-value divergence via max-operator overestimation. Requires L2 normalization + gradient clipping + Double DQN to stabilize, but stabilized model still underperforms.
7. **Exploration-exploitation trap (method 6)**: Without stability fixes → loss diverges but explores widely (cache hit 34%); with fixes → loss converges but trapped in local optima (cache hit 96%). Neither regime produces good optimization.

## Root Cause Analysis

### ~~Problem 1: Representation Bottleneck~~ (DISPROVED by Method 6)

Initially hypothesized that Morgan fingerprints (2D) cannot capture 3D conformation and electrostatics. **Method 6 tested this directly** by replacing Morgan FP with Uni-Mol (209M pretrained, 3D SE(3)-equivariant encoder, 512-dim). Result: UniMolDQN performs **worse** than fingerprint-based MolDQN (top-1 QED 0.81 vs 0.90). Representation quality is NOT the bottleneck.

### ~~Problem 2: Model Capacity Bottleneck~~ (DISPROVED by Method 6)

Initially hypothesized that linear Q-network cannot model the non-linear property landscape. **Method 6 tested this directly** with a deep 3-layer MLP Q-head (513→256→128→1). Result: Same or worse performance. Model capacity is NOT the bottleneck.

### True Root Cause: The MDP Formulation Itself

With representation (Method 6) and optimization strategy (Method 5) eliminated as causes, the remaining explanation is fundamental:

**Sequential decision-making (MDP/DQN) is the wrong abstraction for molecular optimization.**

1. **No temporal structure**: Molecular optimization has no meaningful temporal dependencies. The quality of step 20's modification depends only on the current structure, not the path taken. The optimal policy is essentially greedy.
2. **Sample inefficiency**: RL wastes oracle budget on "learning" a policy, while evolutionary methods use every call for selection.
3. **Exploration-exploitation dilemma is solved better by populations**: GP maintains diversity through population-based search, crossover, and selection — without requiring learned entropy bonuses or epsilon schedules.
4. **DQN's max operator causes systematic Q-value overestimation**, especially harmful with continuous representations (embedding norm ~27 → loss divergence, fixed in v3 but still underperforms).

## GP+ReaSyn Baseline Results (Proxy sEH Docking, seed=42)

| Metric | Value |
|--------|-------|
| Global Best | -12.1999 |
| Global Top-10 | -11.6679 |
| Pop Mean | -10.8194 |
| Scaffolds | 83 |
| Oracle Calls | 10000 |
| ReaSyn Success Rate | 99.5% |
| Top-10 Diversity | 0.8476 |

This serves as the baseline to beat for any future RL approach.

## Conclusion: RL Is Not Viable for This Problem Class

After systematically testing 6 methods across 4 dimensions:

| Dimension | Methods | Result |
|-----------|---------|--------|
| Action space design | 1, 2, 3, 4 | All underperform GP |
| Optimization strategy | 5 (RL guides GP) | No benefit over GP alone |
| Representation quality | 6 (Uni-Mol 3D encoder) | Worse than fingerprints |
| Model capacity | 6 (deep Q-head) | No improvement |

**Evolutionary methods (GP) with domain-specific constraints (ReaSyn) are inherently better suited for molecular optimization.** Future work should focus on accelerating and improving GP+ReaSyn rather than attempting RL-based approaches.

## Code Status

### Method 5: RL+GP (SynOpt) - Independent Module
- `rl/guided_ga.py` - GAPolicy + PPOGAController
- `rl/strategy.py` - StrategyNetwork + StrategyController
- `rl/rewards.py` - Composite reward functions (v1 + v2)
- `rl/population.py` - Diverse selection + ScaffoldTracker
- `rl/reasyn_projector.py` - ReaSyn wrapper (lazy imports from rl/reasyn/)
- `scripts/synopt.py` - Main training script

### Method 6: UniMolDQN - 3D Encoder DQN
- `unimol_dqn.py` - UniMolEncoder (bare-metal), UniMolDQN, EmbeddingCache, MolDQNBaseline
- `agent_unimol.py` - DQN agent with Double DQN, gradient clipping, replay buffer
- `conformer3d.py` - ConformerManager (ConstrainedEmbed + 3D action filter)
- `run_unimol.py` - Main training script (QED comparison)
- `scripts/experiments/run_unimol_qed_v3.sh` - SLURM experiment scripts

### Methods 1-4: Earlier RL Systems
- `rl/reasyn/` - ReaSyn DQN (44 files, independent)
- `rl/route/` - Route DQN (7 files, depends on rl/template/)
- `rl/template/` - Template predictor (6 files, independent)

## Appendix: Pre-Implementation Research Notes (2026-03-12)

Initial research that motivated Method 6. Kept for reference; the experiment results above supersede these design notes.

- **3D encoder candidates evaluated**: Uni-Mol (selected), SchNet, DimeNet++, EGNN, PaiNN
- **RxnFlow sample efficiency analysis**: Short trajectories + action subsampling + dot-product architecture
- **DQN efficiency strategies considered**: N-step returns, PER, embedding cache, multi-conformer augmentation
- **Original Q-head design**: LayerNorm + Dropout → later removed due to Polyak target conflict (see method6 doc)

Full details: `docs/rl_methods/method6_unimol_dqn.md`

## Experiment Logs

- `Experiments/synopt/proxy_seh_rl_reasyn_s42/` - SynOpt (RL+GP+ReaSyn)
- `Experiments/synopt/proxy_seh_norl_reasyn_s42/` - GP+ReaSyn (no RL baseline)
- `Experiments/synopt/proxy_seh_reasyn_s42/` - Earlier SynOpt run
- `Experiments/synopt/proxy_seh_noreasyn_s42/` - GP only (no ReaSyn)
- Slurm logs: `Experiments/logs/synopt_proxy_717133.out`, `synopt_proxy_717201.out`
- UniMolDQN v2: `Experiments/logs/unimol_qed_v2_718624.out`, `unimol_qed_v2_718625.out`
- UniMolDQN v3: `Experiments/logs/unimol_qed_v3_718682.out`, `unimol_qed_v3_718683.out`
