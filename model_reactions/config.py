"""Configuration for model_reactions."""

from dataclasses import dataclass, field


@dataclass
class ReactantPredictorConfig:
    """Configuration for Model 1: ReactantPredictor."""
    # Data
    data_dir: str = "Data/uspto"
    
    # Fingerprint settings
    fp_length: int = 2048
    fp_radius: int = 2
    
    # Retrieval settings
    top_k: int = 10
    
    # Database settings
    max_db_mols: int = 100000  # Max molecules to index


@dataclass 
class ProductPredictorConfig:
    """Configuration for Model 2: ProductPredictor (ReactionT5v2)."""
    model_name: str = "sagawa/ReactionT5v2-forward"
    input_max_length: int = 400
    output_max_length: int = 300
    num_beams: int = 5
    num_return_sequences: int = 3
    cache_size: int = 10000  # LRU cache capacity
    max_sub_batch: int = 64  # Max sequences per T5 forward pass (prevents OOM)


@dataclass
class RetroPredictorConfig:
    """Configuration for RetroPredictor (ReactionT5v2-retrosynthesis)."""
    model_name: str = "sagawa/ReactionT5v2-retrosynthesis"
    input_max_length: int = 300
    output_max_length: int = 400
    num_beams: int = 5
    num_return_sequences: int = 3
    cache_size: int = 10000  # LRU cache capacity
    max_sub_batch: int = 64  # Max sequences per T5 forward pass


@dataclass
class ReactionPredictorConfig:
    """Configuration for the combined predictor."""
    reactant_method: str = "hypergraph"  # 'hypergraph', 'seal', 'fingerprint'
    reactant_top_k: int = 20
    product_top_k: int = 3
    device: str = "auto"
    reactant_config: ReactantPredictorConfig = field(default_factory=ReactantPredictorConfig)
    product_config: ProductPredictorConfig = field(default_factory=ProductPredictorConfig)

    # Plan B: Fragment expansion - when T5 predicts multi-component products (P1.P2),
    # keep ALL valid fragments as action candidates instead of only the largest.
    expand_fragments: bool = False
    fragment_min_atoms: int = 2  # Minimum heavy atoms for a fragment to be kept

    # Plan C: Co-reactant MW penalty - prefer smaller co-reactants.
    # adjusted_score = score * exp(-alpha * max(0, MW - threshold))
    mw_penalty_alpha: float = 0.0  # 0 = disabled, typical: 0.003-0.01
    mw_penalty_threshold: float = 150.0  # MW below this gets no penalty
    co_reactant_oversample: int = 1  # Fetch oversample*top_k, re-rank, keep top_k

    # Product validation filter — removes chemically unreasonable T5v2 outputs
    # Thresholds calibrated against 700 real USPTO reactions (chemist-validated)
    filter_products: bool = True  # Master switch for product filtering
    filter_check_copy: bool = True  # Layer 1: reject product == co_reactant or reactant
    filter_min_tanimoto: float = 0.2  # Layer 2: min Tanimoto(reactant, product) — 0% real USPTO < 0.2
    filter_max_co_tanimoto: float = 0.9  # Layer 2: max Tanimoto(product, co_reactant)
    filter_max_mw_delta: float = 200.0  # Layer 3: max |MW(product) - MW(reactant)| Da
    filter_max_mw_ratio: float = 1.3  # Layer 3: max MW(product) / MW(reactant) — prevents MW explosion
    filter_min_mw_ratio: float = 0.5  # Layer 3: min MW(product) / MW(reactant) — prevents MW collapse
    filter_reject_si: bool = True  # Layer 4: reject products containing Si (protecting groups)
