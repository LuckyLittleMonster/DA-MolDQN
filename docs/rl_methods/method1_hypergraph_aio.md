# Method 1: RL + Hypergraph All-In-One (AIO)

## Purpose

Molecular optimization requires actions that correspond to valid chemical reactions. This method explored using a directed hypergraph neural network to predict both co-reactants and products in a single forward pass ("All-In-One"), providing the DQN agent with synthesizable actions at each step.

The hypothesis was that by coupling reaction prediction with RL-based sequential optimization, the agent could navigate chemical space via valid synthetic transformations, producing molecules that are both high-scoring and synthesizable.

## Main Approach

### Architecture

- **State representation**: Morgan fingerprint (2048-d, radius=2) concatenated with step number (1-d), normalized to unit norm
- **Action space**: DirectedHypergraphNeighborPredictor generates (co_reactant, product) tuples with uniform scores (1.0)
- **Q-network**: Linear MLP — `Linear(2049, hidden) → ReLU → Linear(hidden, 1)`
- **Reward**: QED weight (0.8) + SA score weight (0.2), with optional Tanimoto similarity bonus and MW penalty

### Reaction Handling

The directed hypergraph model uses graph link prediction to simultaneously predict:
1. What co-reactants can react with the current molecule
2. What products result from that reaction

Products are filtered by:
- Minimum Tanimoto similarity to reactant ≥ 0.2
- Molecular weight change |ΔMW| ≤ 200 Da

### Training

- Standard DQN with experience replay
- Epsilon-greedy exploration
- Episodes of 5 steps starting from seed molecules

### Code Location

- `hypergraph/action_generator.py` — AIOActionGenerator class
- `hypergraph/hypergraph_neighbor_predictor.py` — DirectedHypergraphNeighborPredictor
- `environment.py` (reactant_method='aio')
- `scripts/train_qed_hypergraph.sh`

## Results

- AIO reactant prediction accuracy: MRR = 0.013, Recall@10 = 0.031 (poor)
- Product diversity was low, and structural jumps between steps were large
- Q-network loss converged but learned policy was no better than random

## Why It Failed

### Primary Failure: The Jumping Problem

The hypergraph model generates products that are structurally far from the input molecule. Even with Tanimoto ≥ 0.2 and MW filters, the product molecules often have different scaffolds from the reactant. This means:

1. **Breaking RL's core assumption**: RL exploits local smoothness — if Q(s) is high, nearby states should also have predictable Q-values. When each action jumps to a structurally distant molecule, there is no local structure for the Q-function to exploit.

2. **Unstable Q-learning**: The state transitions are effectively random jumps in fingerprint space. The Q-function cannot learn meaningful temporal patterns because consecutive states share little structural similarity.

3. **Worse than random search**: Since each action is a large structural jump, the RL agent cannot build incrementally toward better molecules. A random search with the same number of oracle calls explores equally well.

### Secondary Failures

- **Representation bottleneck**: Morgan fingerprints are 2D topological counts — they cannot capture 3D conformation, electrostatics, or protein-ligand interactions that determine docking scores. The Q-function is trained on a blurred, misleading landscape.

- **Model capacity bottleneck**: The linear Q-network `Q(s,a) = w · φ(s,a)` cannot learn non-linear value functions. Molecular property landscapes exhibit activity cliffs (small structural changes → large property jumps), which a linear model treats as noise.

- **Poor reactant prediction**: With MRR of only 0.013, the hypergraph model cannot reliably identify productive co-reactants. Most predicted reactions are chemically implausible, leading to wasted oracle calls on invalid products.

### Fundamental Insight

The AIO approach conflates two problems: (1) predicting valid reactions and (2) optimizing molecular properties. The hypergraph model is too inaccurate for (1), and the resulting action space is too chaotic for (2). RL needs a stable, local action space to be effective — AIO provides neither.
