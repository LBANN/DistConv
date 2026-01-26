"""Tests for ParallelStrategy validation."""

import pytest

from distconv import ParallelStrategy
from tests.utils import cleanup_parallel_strategy


def test_shard_dim_batch_dimension_rejected(device):
    """Test that batch dimension (0) cannot be used for sharding."""
    with pytest.raises(ValueError, match="Invalid shard_dim value: 0"):
        ParallelStrategy(num_shards=2, shard_dim=0, device_type=device.type)


def test_shard_dim_channel_dimension_rejected(device):
    """Test that channel dimension (1) cannot be used for sharding."""
    with pytest.raises(ValueError, match="Invalid shard_dim value: 1"):
        ParallelStrategy(num_shards=2, shard_dim=1, device_type=device.type)


def test_shard_dim_negative_value_rejected(device):
    """Test that negative dimension values are rejected."""
    with pytest.raises(ValueError, match="Invalid shard_dim value: -1"):
        ParallelStrategy(num_shards=2, shard_dim=-1, device_type=device.type)


def test_shard_dim_tuple_with_invalid_value_rejected(device):
    """Test that tuple with invalid dimension value is rejected."""
    with pytest.raises(ValueError, match="Invalid shard_dim value: 1"):
        ParallelStrategy(num_shards=(2, 2), shard_dim=(2, 1), device_type=device.type)


def test_shard_dim_valid_spatial_dimension_accepted(device):
    """Test that valid spatial dimension (2) is accepted."""
    # Should not raise any exception
    strategy = ParallelStrategy(num_shards=2, shard_dim=2, device_type=device.type)
    assert strategy.shard_dim == (2,)
    cleanup_parallel_strategy(strategy)


def test_shard_dim_valid_tuple_accepted(device):
    """Test that valid tuple of spatial dimensions is accepted."""
    # Should not raise any exception
    strategy = ParallelStrategy(num_shards=(2, 2), shard_dim=(2, 3), device_type=device.type)
    assert strategy.shard_dim == (2, 3)
    cleanup_parallel_strategy(strategy)
