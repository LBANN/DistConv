"""
Resource registry for managing ParallelStrategy/device_mesh access.

Separates compilation-friendly primitive metadata (DistConvSpec) from
runtime resources (device_mesh, process groups). Resources are registered
outside tracing and retrieved during tensor reconstruction.
"""

from typing import Dict, Tuple
from .spec import DistConvSpec

# Type alias for registry key
_SpecKey = Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]  # (rank, world_size, num_shards, shard_dim)

# Global registry mapping specs to their ParallelStrategy resources
_parallel_strategy_registry: Dict[_SpecKey, "ParallelStrategy"] = {}


def register_parallel_strategy(spec: DistConvSpec, parallel_strategy: "ParallelStrategy") -> None:
    """
    Register ParallelStrategy resources for a DistConvSpec.

    Call this outside tracing (e.g., in distribute() before creating DCTensor).
    Allows later reconstruction to access device_mesh without storing it in spec.

    Args:
        spec: DistConvSpec with primitive metadata
        parallel_strategy: ParallelStrategy with device_mesh and process groups
    """
    key = _make_key(spec)
    _parallel_strategy_registry[key] = parallel_strategy


def get_parallel_strategy(spec: DistConvSpec) -> "ParallelStrategy":
    """
    Look up registered ParallelStrategy for a DistConvSpec.

    Call this when needing device_mesh (e.g., in to_ddp() for redistribution).
    Raises RuntimeError if spec not registered.

    Args:
        spec: DistConvSpec to look up

    Returns:
        Registered ParallelStrategy with resources

    Raises:
        RuntimeError: If spec not found in registry
    """
    key = _make_key(spec)
    if key not in _parallel_strategy_registry:
        raise RuntimeError(
            f"ParallelStrategy not registered for spec: "
            f"rank={spec.rank}, world_size={spec.world_size}, "
            f"num_shards={spec.num_shards}, shard_dim={spec.shard_dim}. "
            "Call register_parallel_strategy() before using this spec."
        )
    return _parallel_strategy_registry[key]


def _make_key(spec: DistConvSpec) -> _SpecKey:
    """Create registry key from DistConvSpec."""
    return (spec.rank, spec.world_size, spec.num_shards, spec.shard_dim)


def clear_registry() -> None:
    """
    Clear all registered ParallelStrategies.

    Useful for testing or when starting fresh with new process group setup.
    """
    _parallel_strategy_registry.clear()
