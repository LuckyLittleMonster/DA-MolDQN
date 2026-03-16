# Method 5: RL-Guided GP + ReaSyn (SynOpt)

## Purpose

Methods 1-4 attempted to use RL as the primary optimization engine. All underperformed GP+ReaSyn. Method 5 takes a fundamentally different approach: instead of replacing GP, use RL to **guide** GP's operator selection. The hypothesis was that RL could learn when to explore vs exploit, which crossover type to use, which mutation operator to apply, and how aggressively to project molecules to synthesizable space — improving GP's efficiency without replacing its core search mechanism.

## Main Approach

### Dual-Layer RL Architecture

**Layer 1 — Strategy Controller** (`rl/strategy.py`):
- **Input**: 11-dim population statistics
  - Score statistics: mean, std, min, max, median
  - Tanimoto diversity, scaffold ratio
  - Stagnation counter, oracle progress
  - ReaSyn success rate, top-K improvement rate
- **Output**: 5-dim strategy vector ∈ [0,1]⁵
  - `explore_weight`: exploration vs exploitation pressure
  - `scaffold_preserve`: scaffold conservation strength
  - `diversity_target`: diversity pressure in selection
  - `mutation_intensity`: conservative vs aggressive mutations
  - `projection_tolerance`: ReaSyn similarity threshold
- **Network**: 2-layer MLP (11 → 64 → 64 → 5) + sigmoid output, ~5,318 params
- **Training**: PPO, updated every K=10 generations
- **Role**: High-level meta-policy that adapts GA behavior to population state

**Layer 2 — Operator Controller** (`rl/guided_ga.py`):
- **Input**: Per-offspring context (parent fingerprints + scores + strategy vector)
- **Output**: Per-offspring operator selections
  - Parent B: Attention-based selection (data-dependent, replacing random selection)
  - Crossover type: Ring vs non-ring (2-way categorical)
  - Mutation operator: 7 types + no-op (8-way categorical)
- **Network**: Morgan FP (2048) encoder + attention mechanism + 3 categorical heads + value head, ~1.52M params
- **Training**: PPO, updated every U=5 generations
- **Strategy conditioning**: Context embedding layer accepts strategy vector from Layer 1

### Reward Function — 5-Component Composite (v2)

| Component | Weight Range | Definition |
|-----------|-------------|------------|
| Improvement | 0.25–0.60 | tanh(parent_best - offspring_score) |
| Novelty | 0.05–0.30 | 1 - max(Tanimoto to population) |
| Validity | 0.05–0.15 | +1 valid, -1 invalid |
| Diversity | 0.05–0.15 | Structural distance proxy |
| Synth Fidelity | 0.20–0.25 | Tanimoto(original, ReaSyn-projected) |

Curriculum scheduling based on oracle progress:
- Early (0-30%): explore-heavy weights
- Middle (30-70%): balanced
- Late (70-100%): exploit-heavy weights

### Population Management (`rl/population.py`)

- Diversity-preserving selection (not just truncation)
- Greedy algorithm mixing score + distance (weight 0.3 for diversity)
- ScaffoldTracker: Counts unique Murcko scaffolds per generation

### ReaSyn Projector (`rl/reasyn_projector.py`)

- FastSampler wrapper for synthesis projection
- Projects each GA offspring to its nearest synthesizable analog
- Guarantees 100% synthesizable output
- Returns: (projected_smi, similarity, synthesis_route)
- Caching for repeated molecules

### Training Loop (from `scripts/synopt.py`)

```
For each generation (1 to max_generations):
  1. Compute 11-dim population statistics
  2. [Every 10 gens] Update Strategy via PPO → strategy vector
  3. Condition Operator on strategy vector
  4. [Operator] Select parent pairs + crossover type + mutation op
  5. [joblib parallel] Execute crossover + mutation → offspring
  6. [ReaSyn] Project offspring to synthesizable space
  7. [Oracle] Score molecules (proxy docking, UniDock, or QED)
  8. Compute 5-component composite rewards
  9. [Every 5 gens] Update Operator via PPO
  10. Population update (diversity-aware or truncation selection)
  11. Log metrics
```

### Code Location

- `scripts/synopt.py` (955 lines) — Full training script (fully implemented)
- `rl/strategy.py` — Strategy Layer (designed but not implemented)
- `rl/guided_ga.py` — Operator Layer (designed but not implemented)
- `rl/rewards.py` — Composite reward functions (designed but not implemented)
- `rl/population.py` — Diversity-preserving selection (designed but not implemented)
- `rl/reasyn_projector.py` — ReaSyn wrapper (designed but not implemented)
- `docs/synopt.md` — Architecture specification
- `docs/rl_guided_ga.md` — RL-GA motivation document
- `docs/plans/2026-03-11-dual-layer-rl-synopt.md` — 2079-line implementation plan
- `tests/test_strategy.py`, `tests/test_rewards.py`, etc. — Test files (exist, modules missing)

**Note**: The RL modules were designed and specified in detail but not fully implemented. The training script `synopt.py` was written to support both RL-guided and baseline (no-RL) modes. The baseline mode was used for comparison experiments.

## Results

**Four comparison runs** (proxy sEH docking, seed=42, 10K oracle budget):

| Experiment | RL | ReaSyn | Global Best | Top-10 Mean | Pop Mean | Scaffolds | Oracle Calls |
|------------|:---:|:---:|-----------|----------|----------|-----------|------------|
| **GP+ReaSyn (no RL)** | - | ✓ | **-12.1999** | **-11.6679** | **-10.8194** | 83 | 10000 |
| GP+ReaSyn (with RL) | ✓ | ✓ | -12.1999 | -11.4097 | -10.6421 | 65 | 6364 |
| GP+ReaSyn (earlier) | ✓ | ✓ | -11.9322 | -11.6160 | -10.8389 | 73 | 7432 |
| GP only (no ReaSyn) | - | - | -11.3922 | -11.1940 | -10.4413 | 83 | 6530 |

### Key Findings

1. **No RL benefit**: GP+ReaSyn without RL achieves the same best score (-12.1999) with full oracle budget
2. **Top-10 degradation**: RL guidance worsens top-10 mean by 0.26 (-11.67 → -11.41)
3. **Population mean loss**: RL reduces population mean by 0.18 (-10.82 → -10.64)
4. **Early convergence**: RL runs terminate early (6364 vs 10000 oracle calls) due to stagnation detection
5. **Scaffold collapse**: RL+ReaSyn discovers 65 scaffolds vs 83 without RL (22% fewer)
6. **ReaSyn is the key ingredient**: GP+ReaSyn (-12.20) >> GP-only (-11.39), confirming ReaSyn's value

## Why It Failed

### Problem 1: RL Adds Exploitation Pressure That Hurts Exploration

The RL operator policy learns to select actions that improve the immediate reward (offspring quality). This sounds beneficial, but in practice:
- The policy concentrates on crossover types and mutation operators that worked recently
- This reduces the diversity of genetic operations applied to the population
- Entropy collapse: the policy becomes deterministic too quickly
- Result: fewer scaffolds discovered (65 vs 83), narrower search

In contrast, GP's random operator selection maintains exploration naturally. Random selection is close to optimal when the operator landscape is approximately uniform — which it is for GA on ReaSyn's well-behaved action space.

### Problem 2: Composite Reward Misalignment

The 5-component reward function introduces subjective weightings:
- How much should "novelty" matter vs "improvement"?
- Should "synth fidelity" be rewarded separately from the oracle score?
- Curriculum scheduling adds another layer of hyperparameters

These design choices can easily misalign the reward signal with the true objective (best docking score). The curriculum schedule may shift exploration pressure at the wrong time, causing early convergence.

### Problem 3: GP+ReaSyn Is Already Near-Optimal

This is the most fundamental issue. When ReaSyn constrains all offspring to be synthesizable, the optimization landscape is well-behaved:
- Population-based search maintains diversity automatically
- Truncation selection provides strong exploitation pressure
- Random crossover + mutation explores broadly
- ReaSyn projection ensures synthesizability

There is simply **no room for RL to add value**. The baseline is already operating near the Pareto frontier of exploration vs exploitation for this action space.

### Problem 4: Sample Inefficiency

RL needs many episodes to learn useful policies. With only ~280 generations (10K oracle budget / ~36 offspring per gen), the Strategy and Operator controllers do not have enough training signal:
- Strategy updates every 10 gens: only ~28 updates total
- Operator updates every 5 gens: only ~56 updates total
- PPO requires multiple epochs per update for stability

The policy barely has time to move away from its initialization before the oracle budget is exhausted.

### Fundamental Insight

Method 5 is the definitive experiment: it tests whether RL can improve GP's performance when given the perfect tool (ReaSyn action space) and perfect information (population statistics). The answer is **no**. This eliminates the hypothesis that RL's failure in Methods 1-4 was due to poor action space design. The problem is more fundamental:

**For molecular optimization with a black-box oracle, population-based evolutionary methods are inherently superior to sequential decision-making (RL).** GP's population diversity, crossover-based recombination, and hard selection are better suited to this problem class than RL's policy learning, which requires temporal structure that molecular optimization does not provide.
