"""Configuration for DAT (Dynamic Alpha Tuning) strategy."""

from dataclasses import dataclass


@dataclass
class DATConfig:
    """Configuration for Dynamic Alpha Tuning retriever.

    Attributes:
        dense_weight_default: Default weight for dense retrieval (0.0 to 1.0)
        sparse_weight_default: Default weight for sparse retrieval (0.0 to 1.0)
        top_k_dense: Number of results to retrieve from dense retrieval
        top_k_sparse: Number of results to retrieve from sparse retrieval
        use_dynamic_tuning: Whether to use LLM-based dynamic alpha tuning
        effectiveness_threshold: Minimum effectiveness score to consider (0.0 to 1.0)
        llm_model: LLM model to use for effectiveness scoring
        temperature: Temperature for LLM effectiveness scoring
    """

    dense_weight_default: float = 0.5
    sparse_weight_default: float = 0.5
    top_k_dense: int = 20
    top_k_sparse: int = 20
    use_dynamic_tuning: bool = True
    effectiveness_threshold: float = 0.3
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0.0 <= self.dense_weight_default <= 1.0):
            raise ValueError("dense_weight_default must be between 0.0 and 1.0")
        if not (0.0 <= self.sparse_weight_default <= 1.0):
            raise ValueError("sparse_weight_default must be between 0.0 and 1.0")
        if self.top_k_dense < 1:
            raise ValueError("top_k_dense must be at least 1")
        if self.top_k_sparse < 1:
            raise ValueError("top_k_sparse must be at least 1")
        if not (0.0 <= self.effectiveness_threshold <= 1.0):
            raise ValueError("effectiveness_threshold must be between 0.0 and 1.0")
