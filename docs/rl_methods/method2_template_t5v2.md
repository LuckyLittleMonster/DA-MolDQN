# Method 2: Template/Model Reactant Prediction + ReactionT5v2 Product Prediction

## Purpose

Method 1 (Hypergraph AIO) suffered from poor reaction prediction accuracy. This method improved the approach by decomposing reaction prediction into two stages: (1) a template-based or model-based reactant predictor, and (2) ReactionT5v2, a pretrained reaction transformer, for product prediction. The goal was to generate more chemically valid actions for the DQN agent while maintaining synthesizability.

## Main Approach

### Architecture

- **State representation**: Morgan fingerprint (2048-d, radius=2) + step number (same as Method 1)
- **Q-network**: Linear MLP (same as Method 1)

### Two-Stage Action Generation Pipeline

**Stage 1 — Reactant Prediction** (3 implementations tested):

1. **HypergraphReactantPredictor**: Uses trained HypergraphLinkPredictor with pre-computed embeddings for fast kNN retrieval. Returns top-k co-reactants with scores.

2. **FingerprintReactantPredictor**: Simple Morgan fingerprint similarity baseline. Computes Tanimoto similarity between current molecule and candidate building blocks.

3. **TemplateReactionPredictor**: RDKit SMARTS templates (deterministic, chemically valid). Matches current molecule substructures against 15K+ reaction templates, then identifies compatible building blocks.

**Stage 2 — Product Prediction**:

- **ReactionT5v2**: Pretrained reaction transformer
- Input format: `"REACTANT:{mol}.{co_reactant} REAGENT:"`
- Output: Product SMILES with confidence scores
- LRU cache to avoid redundant T5 inference

### Product Filtering

Products are filtered by multiple criteria:
- Min Tanimoto(reactant, product) ≥ 0.2
- Min Tanimoto(product, co_reactant) ≤ 0.9
- |MW(product) - MW(reactant)| ≤ 200 Da
- MW(product) / MW(reactant) ∈ [0.5, 1.3]
- Reject Silicon-containing products

### Hybrid Variant

A more sophisticated version was also tested:
- Stage 1: Template or model-based reactant retrieval
- Stage 2a: AIO product embedding search as initial retrieval
- Stage 2b: HypergraphLinkPredictorV3 re-ranks candidates using NCN + CF-NCN features
- Hybrid results: MRR = 0.025, Recall@10 = 0.061 (~2× improvement over AIO alone)

### Training Configuration

- 1000 episodes, 5000 iterations
- 64 molecules in parallel
- Max 5 steps per episode
- Top-K = 128 candidates (20 model + 108 template)

### Code Location

- `model_reactions/reactant_predictor.py` — ReactantPredictor implementations
- `model_reactions/product_prediction/product_predictor.py` — T5v2 wrapper
- `model_reactions/reaction_predictor.py` — Combined pipeline
- `model_reactions/hybrid_template_predictor.py` — Hybrid variant
- `template/template_predictor.py` — Template-based predictor
- `environment.py` (reactant_method='template', 'template_2model', 'hybrid')
- `scripts/train_hybrid_template.sh`

## Results

- Template-based: 100% chemical validity of products, but structural diversity still too high
- T5v2-based: ~48% chemical validity (hallucinated products)
- Hybrid: MRR improved to 0.025, but still insufficient for reliable optimization
- No successful convergence documented — method was abandoned before achieving production results

## Why It Failed

### Primary Failure: Same Jumping Problem

Despite the improved two-stage pipeline, this method suffers from the same fundamental issue as Method 1:

1. **Large structural changes between steps**: Even with RDKit SMARTS templates, reactions transform molecules significantly. Acetate + aniline → N-acetylaniline is a "small" reaction, but the abundance of templates and building blocks means the space of possible products is vast and structurally diverse.

2. **Q-function cannot learn**: When state transitions are large structural jumps, the Q-values for consecutive states are uncorrelated. The Q-network converges to a constant (mean value) rather than learning useful structure.

### T5v2 Hallucination Problem

ReactionT5v2 is a language model trained on reaction SMILES. It inherits typical language model failure modes:
- **Generates chemically invalid products** (~52% failure rate)
- **Hallucinated structures**: Products inconsistent with known chemistry
- **No reaction mechanism awareness**: Treats SMILES as text tokens, not chemical entities
- **Dead-end exploration**: Invalid products waste oracle calls and send the agent into unproductive regions of chemical space

### Representation and Capacity Bottlenecks (Same as Method 1)

- Morgan fingerprints cannot capture 3D properties relevant to docking
- Linear Q-network cannot model the non-linear property landscape
- Combined effect: RL reduces to near-random search

### Computational Cost

- T5v2 inference: 26.9s per step for 64 molecules (vs 1.3s for AIO)
- Template matching: 2-3s per step for 64 molecules
- The two-stage pipeline is 5-20× slower than AIO without meaningful quality improvement

### Fundamental Insight

Breaking reaction prediction into two stages improves chemical validity (especially with templates), but does not solve the jumping problem. The issue is not prediction accuracy — it is that **any reaction-based action space inherently produces large structural changes**, making RL's sequential optimization ineffective. This insight motivated the shift to local modification approaches (Methods 3-5) and non-RL methods (GP+ReaSyn).
