# Method 3: RL Route Optimization (Route-DQN)

## Purpose

Methods 1-2 failed because reaction-based actions produce large structural jumps, breaking RL's local optimization advantage. Route-DQN takes a different approach: instead of predicting new reactions, it starts with an existing synthesis route and uses RL to select which building block to replace at each position. This constrains the action space to local modifications within a fixed synthetic plan, avoiding the jumping problem.

## Main Approach

### Architecture

**State Representation — Route Encoding**:
- `RouteStateEncoder`: Encodes entire synthesis routes into 256-dim embeddings
- Per-step features: intermediate molecule fingerprint (4096-d Morgan) + template embedding (128-d) + block embedding (64-d)
- Two-layer MLP (4288 → hidden → 256) with mean-pooling over route steps
- Captures the full synthetic context, not just the final product

**Hierarchical Two-Level Action Space**:

- **Level 1 (Q_pos)**: Select which route position to modify
  - Input: route embedding (256-d)
  - Output: Q-values for each modifiable position + EXTEND/TRUNCATE virtual positions
  - Masking: Uni-molecular reaction positions masked (cannot swap building blocks)

- **Level 2 (Q_bb)**: Select replacement building block for chosen position
  - Input: (route_emb, position_fp, template_emb)
  - Bilinear scoring: `scoring_emb (64-d) @ block_embs (10K × 64-d).T`
  - Precomputed block embeddings for O(1) inference

**Three-Level Action Masking**:
```
10,000 building blocks (full library)
  ↓ L1: Route structure mask (uni-reactions not modifiable)
~3-5 modifiable positions per route
  ↓ L2: Template compatibility mask (precomputed, 145K valid pairs)
~150 compatible BBs per position
  ↓ L3: Cascade validation (forward-execute to verify downstream reactions work)
~50-100 valid BBs
  ↓ Q_bb scoring → selected BB
```

**Q-Network — Total 3.77M parameters**:
- RouteStateEncoder: ~1.2M params
- PositionPolicyNetwork (Q_pos): ~70K params (256 → 128 → n_positions)
- RouteBBScoringNetwork (Q_bb): ~2.3M params (context encoder + condition MLP + block encoder)

**Training Algorithm**:
- Double DQN with Prioritized Experience Replay (PER)
  - Alpha = 0.6 (priority exponent), Beta annealing 0.4 → 1.0
  - TD-error based priority updates
- Polyak target network averaging (τ = 0.995)
- Optional reward shaping: F = γ × QED(s') - QED(s)
- Action subsampling: 512 BBs from compatible set (computational tractability)
- Learning rate: 1e-4, Discount: 0.9

### Route Generation Methods

Four methods to generate initial synthesis routes:
1. **Random Forward**: Random template + BB selection → N steps (75% success, 0.1s/64 mols)
2. **Retro (Reversed Templates)**: DFS on inverted SMARTS templates (84.4% success, 425ms/mol)
3. **AiZynth**: External retrosynthesis tool results (JSON input)
4. **PaRoutes**: Pre-computed PARoutes database

### Cascade Validation

The key engineering challenge: when replacing a BB at position i, all downstream reactions must still work. Cascade validation forward-executes the route from position i to verify feasibility.

- Parallelized with 32 workers (multiprocessing, fork context)
- Fast-forward mode: SanitizeMol only, RunReactants(maxProducts=1)
- Last-step shortcut: no downstream validation needed
- **Performance**: 26× speedup (17.3s → 542ms for 48 routes)

### Code Location

- `rl/route/route_dqn.py` (1398 lines) — Hierarchical DQN agent, SumTree PER
- `rl/route/policy_network.py` — RouteStateEncoder, Q_pos, Q_bb networks
- `rl/route/route_environment.py` — RL environment (step, reset, cascade validation)
- `rl/route/train.py` — RouteTrainer episode loop
- `rl/route/route.py` — RouteStep, SynthesisRoute data structures
- `rl/route/retro_decompose.py` — 4 route generation methods
- `configs/method/route.yaml`

## Results

**Route-DQN with DRD2 Docking Reward (500 episodes)**:

| Config | Mean Reward | Final Reward | Best QED | Best Reward |
|--------|-------------|--------------|----------|-------------|
| Baseline | 0.514 | 0.355 | 0.633 | 2.844 |
| + Experimental (PER + reward shaping + Double DQN) | 0.807 | 0.688 | 0.660 | 2.934 |

**QED-Only (100 episodes, retro routes)**:
- Best QED: 0.938
- Best mean reward: 2.603
- Cascade validation: 583ms/step

**Comparison to GP+ReaSyn baseline** (proxy sEH):
- GP+ReaSyn: global best = -12.20, top-10 = -11.67
- Route-DQN: operates on DRD2/QED — not directly comparable on sEH docking, but overall pattern shows underperformance

## Why It Failed

### Problem 1: Severely Constrained Optimization Space

Each synthesis route has only 1.4–2.0 modifiable positions on average:
- Retro routes: 2.2 steps average, 1.4 modifiable positions
- Random routes: 3.0 steps average, 2.0 modifiable positions
- Uni-molecular reactions cannot be modified (no BB to swap)

With so few degrees of freedom, the agent can only make marginal improvements. The space of reachable molecules from a single route is tiny compared to the full chemical space.

### Problem 2: Init-Mol Dependency

Route-DQN requires good starting routes. If the initial molecule has poor properties, the agent can only swap BBs within the same synthetic plan — it cannot discover entirely new scaffolds or reaction sequences. This makes the method heavily dependent on the quality of initial molecules from the seed set.

### Problem 3: Cascade Validation Bottleneck

Even with 32-worker parallelization and 26× speedup, cascade validation still consumes ~80% of training time (~500ms/step). This limits the number of episodes that can be run within a reasonable time budget, reducing sample efficiency.

### Problem 4: Representation Bottleneck (Same as Methods 1-2)

- Morgan fingerprints (4096-d) in the route state encoder are still 2D topological counts
- Cannot capture 3D binding interactions that determine docking scores
- The route embedding (256-d) is a compressed, lossy representation of the synthesis plan

### Problem 5: GP Already Sufficient

When ReaSyn provides the action space (projecting molecules to synthesizable analogs), GP's random mutation + hard selection strategy is already near-optimal. Route-DQN's more structured but constrained approach cannot match GP's ability to freely explore chemical space while maintaining synthesizability.

### Fundamental Insight

Route-DQN successfully solves the jumping problem by constraining actions to BB swaps within an existing route. However, this solution introduces a new problem: **the optimization space is too small**. With only 1-2 modifiable positions per route, the agent cannot make the large structural changes needed to find high-scoring molecules. There is an inherent tension between action locality (needed for RL stability) and search breadth (needed for optimization quality). Route-DQN is on the wrong side of this tradeoff.
