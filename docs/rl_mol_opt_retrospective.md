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

## Common Failure Patterns

1. **Init mol dependency**: All RL methods require good starting molecules. Cannot discover novel scaffolds independently.
2. **Poor mean scores**: Even when top scores approach GP, the distribution of good molecules is narrow.
3. **Jumping problem (methods 1,2)**: Large structural changes between RL steps break the local smoothness assumption that makes RL valuable.
4. **Premature convergence (method 5)**: RL policy collapses entropy, reducing exploration below random GP baseline.
5. **GP already sufficient**: For methods where ReaSyn provides the action space, GP's random search + selection is already near-optimal.

## Root Cause Analysis

### Problem 1: Representation Bottleneck (Fingerprint Limitations)

Morgan fingerprints are 2D topological substructure counts. They fundamentally cannot capture:
- **3D conformation**: Docking scores depend on molecular shape in protein pocket
- **Electrostatic properties**: Charge distribution, H-bond patterns
- **Flexibility**: Conformational entropy, rotatable bonds in context
- **Protein-ligand interactions**: Shape complementarity, induced fit

Consequence: Two fingerprint-similar molecules can have vastly different docking scores (different 3D conformations), and two fingerprint-dissimilar molecules can dock equally well (similar 3D shapes). The Q-function is trained on a blurred, misleading landscape.

### Problem 2: Model Capacity Bottleneck (Linear DQN)

Linear Q(s,a) = w . phi(s,a) cannot learn non-linear value functions. The molecular property landscape is highly non-linear - small structural changes can cause large property jumps (activity cliffs). A linear model sees these as noise and learns only coarse trends.

### Combined Effect

DQN can neither perceive the landscape accurately (bad representation) nor model it faithfully (insufficient capacity). This reduces RL to near-random search, which cannot compete with GP's simpler but more robust "random mutation + hard selection" strategy.

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

## Requirements for a Viable RL Approach

1. **3D-aware molecular representation**: Must capture conformation, shape, electrostatics. Candidates: Uni-Mol, SchNet/DimeNet, 3D pharmacophore descriptors, or learned representations from docking-score prediction.

2. **Sufficient model capacity**: Non-linear Q-network (deep MLP, GNN, or Transformer) that can model the complex property landscape.

3. **Stable action space**: Actions must produce structurally similar molecules (small step size) to maintain RL's advantage. ReaSyn analogs satisfy this but may be too conservative.

4. **Sample efficiency**: Must learn useful policy within the oracle budget (typically 10K calls). Pre-training on property prediction or transfer from related targets could help.

## Code Status

### RL+GP (SynOpt) - Independent Module
- `rl/guided_ga.py` - GAPolicy + PPOGAController
- `rl/strategy.py` - StrategyNetwork + StrategyController
- `rl/rewards.py` - Composite reward functions (v1 + v2)
- `rl/population.py` - Diverse selection + ScaffoldTracker
- `rl/reasyn_projector.py` - ReaSyn wrapper (lazy imports from rl/reasyn/)
- `scripts/synopt.py` - Main training script
- `tests/` - 27 tests, all passing

These files are 95% independent from the existing RL systems (`rl/reasyn/`, `rl/route/`). Only `reasyn_projector.py` has lazy dependency on `rl.reasyn.models` and `rl.reasyn.rl.actions`.

### Existing RL Systems
- `rl/reasyn/` - ReaSyn DQN (44 files, independent)
- `rl/route/` - Route DQN (7 files, depends on rl/template/)
- `rl/template/` - Template predictor (6 files, independent)

No reverse dependencies: existing RL code does not import from RL+GP modules.

## Research: Next-Generation DQN Architecture (2026-03-12)

### Candidate 3D-Aware Encoders

| Model | Embedding Dim | Pretrained Data | 安装方式 | 优势 | 劣势 |
|-------|--------------|-----------------|---------|------|------|
| **Uni-Mol** | 512 | 209M conformations (QM9+GEOM) | `pip install unimol-tools` | 最强预训练，CLS token直接用，SE(3)-equivariant | ~100ms/mol (需缓存) |
| SchNet | 128 | QM9 | PyG内置 | 简单连续卷积 | 预训练数据少，无CLS token |
| DimeNet++ | 256 | QM9 | PyG内置 | 方向信息(angular) | 计算较重 |
| EGNN | 可配置 | 无预训练 | 需自实现 | 简洁equivariant | 需从零训练 |
| PaiNN | 128 | QM9 | PyG内置 | 等变消息传递 | 预训练范围窄 |

**推荐**: Uni-Mol — 预训练规模最大(209M)，SMILES输入自动生成3D构象(ETKDG)，CLS token提供分子级512维embedding。

### RxnFlow 样本效率机制分析

RxnFlow (ICLR 2025) 使用标准 TB loss（**非 SubTB**），其样本效率来源：

1. **极短轨迹 (max 3步)**: TB 梯度方差与轨迹长度正相关，短轨迹=低方差=快收敛
2. **Action space 子采样 + 重要性权重**: 从120万BB中采样1%，通过 `log(1/sampling_ratio)` 校正logit得到无偏梯度估计，计算量降100x
3. **Dot-product 架构**: `state_emb @ block_emb.T` 一次矩阵乘法同时打分所有BB，BB embedding可预计算/缓存
4. **EMA sampling model (τ=0.9)**: 解耦探索与学习，防止分布坍缩
5. **Random action prob (10%)**: 保持探索多样性

关键参数: `num_from_policy=64`, `num_training_steps=10000`, `max_len=3`, `sampling_tau=0.9`

### DQN 对应的样本效率策略

RxnFlow 的 SubTB/短轨迹机制是 GFlowNet 特有的。DQN 框架下对应方案：

- **N-step returns**: 多步TD目标，一条轨迹生成多个不同步长的训练信号
- **Prioritized Experience Replay (PER)**: 高TD-error样本获得更多训练
- **Embedding 缓存**: Uni-Mol ~100ms/mol，必须缓存避免重复计算
- **多构象数据增强**: 同一SMILES的不同3D构象 → 不同embedding → 更多训练信号

### 架构方案

```
UniMolDQN:
  Encoder: Uni-Mol (frozen, 512-dim CLS token, pretrained 209M conformations)
  Q-head: 513 → 512 → 256 → 128 → 1 (LayerNorm + ReLU + Dropout)

Training: Double DQN + PER + N-step returns
Interface: 与现有 gnn_dqn.py 的 frozen encoder + Q-head 模式完全一致
```

### 6. UniMolDQN — 3D Molecular Encoder (2026-03-14)

- **Approach**: Replace fingerprint with Uni-Mol 3D encoder (frozen, 512-dim CLS token). Deep Q-head replaces linear Q.
- **Motivation**: Address both root causes — 3D representation + non-linear capacity.

**v2** (bare-metal encoder + ConstrainedEmbed):

| Seed | Episodes (of 2500) | top1 QED | top10 QED | Loss | Cache Hit | Status |
|------|---------------------|----------|-----------|------|-----------|--------|
| 123 | 520 | 0.8650 | 0.8299 | 10.4 (diverging) | 34.8% | CANCELLED 6h |
| 42 | 780 | 0.8047 | 0.7805 | 23.8 (diverging) | 41.5% | CANCELLED 6h |

**v3** (+ L2-norm + no LayerNorm + grad_clip=1.0 + Double DQN + lr=5e-5):

| Seed | Episodes (of 2500) | top1 QED | top10 QED | Loss | Cache Hit | Status |
|------|---------------------|----------|-----------|------|-----------|--------|
| 123 | 500 | 0.8144 | 0.8144 | 0.17 (stable) | 91.8% | TIMEOUT 12h |
| 42 | 830 | 0.7444 | 0.7435 | 0.03 (converging) | 96.2% | TIMEOUT 12h |

**Analysis**:
- v2: Loss diverges catastrophically (0.01 → 23.8). High top1 (0.865) achieved via early exploration (Ep 60), not learned policy. Q-network fails to converge.
- v3: Loss stabilizes with L2-norm + Double DQN, but scores plateau well below QED optimum (~0.948). seed=42 stagnates from Ep 260–830 (top1: 0.697→0.744).
- Both versions are far below simple GA baselines (QED top1 ~0.93+).
- 3D encoder did not rescue RL: the core issue may not be representation quality alone.

**Conclusion**: Even with a 3D-aware encoder (Uni-Mol, 209M pretrained) and stabilized training (Double DQN, L2-norm, gradient clipping), DQN-based molecular optimization underperforms simple evolutionary methods on QED. This suggests the problem is more fundamental than representation — likely related to the sequential decision-making formulation itself being poorly suited to molecular optimization.

## Experiment Logs

- `Experiments/synopt/proxy_seh_rl_reasyn_s42/` - SynOpt (RL+GP+ReaSyn)
- `Experiments/synopt/proxy_seh_norl_reasyn_s42/` - GP+ReaSyn (no RL baseline)
- `Experiments/synopt/proxy_seh_reasyn_s42/` - Earlier SynOpt run
- `Experiments/synopt/proxy_seh_noreasyn_s42/` - GP only (no ReaSyn)
- Slurm logs: `Experiments/logs/synopt_proxy_717133.out`, `synopt_proxy_717201.out`
- UniMolDQN v2: `Experiments/logs/unimol_qed_v2_718624.out`, `unimol_qed_v2_718625.out`
- UniMolDQN v3: `Experiments/logs/unimol_qed_v3_718682.out`, `unimol_qed_v3_718683.out`
