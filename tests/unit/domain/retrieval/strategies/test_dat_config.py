"""Unit tests for DAT configuration."""

import pytest

from ragprod.domain.retrieval.strategies.dat.config import DATConfig


def test_dat_config_defaults():
    """Test default configuration values."""
    config = DATConfig()

    assert config.dense_weight_default == 0.5
    assert config.sparse_weight_default == 0.5
    assert config.top_k_dense == 20
    assert config.top_k_sparse == 20
    assert config.use_dynamic_tuning is True
    assert config.effectiveness_threshold == 0.3
    assert config.llm_model == "gpt-4o-mini"
    assert config.temperature == 0.0


def test_dat_config_custom_values():
    """Test custom configuration values."""
    config = DATConfig(
        dense_weight_default=0.7,
        sparse_weight_default=0.3,
        top_k_dense=10,
        top_k_sparse=15,
        use_dynamic_tuning=False,
    )

    assert config.dense_weight_default == 0.7
    assert config.sparse_weight_default == 0.3
    assert config.top_k_dense == 10
    assert config.top_k_sparse == 15
    assert config.use_dynamic_tuning is False


def test_dat_config_validation_dense_weight():
    """Test validation of dense weight."""
    with pytest.raises(ValueError, match="dense_weight_default must be between 0.0 and 1.0"):
        DATConfig(dense_weight_default=1.5)

    with pytest.raises(ValueError, match="dense_weight_default must be between 0.0 and 1.0"):
        DATConfig(dense_weight_default=-0.1)


def test_dat_config_validation_sparse_weight():
    """Test validation of sparse weight."""
    with pytest.raises(ValueError, match="sparse_weight_default must be between 0.0 and 1.0"):
        DATConfig(sparse_weight_default=1.5)


def test_dat_config_validation_top_k():
    """Test validation of top_k values."""
    with pytest.raises(ValueError, match="top_k_dense must be at least 1"):
        DATConfig(top_k_dense=0)

    with pytest.raises(ValueError, match="top_k_sparse must be at least 1"):
        DATConfig(top_k_sparse=0)


def test_dat_config_validation_threshold():
    """Test validation of effectiveness threshold."""
    with pytest.raises(ValueError, match="effectiveness_threshold must be between 0.0 and 1.0"):
        DATConfig(effectiveness_threshold=1.5)
