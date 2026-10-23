"""
Serializable metadata specification for distributed convolution prototypes.
"""

import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class DistConvSpec:
    """
    Immutable, serializable distributed convolution specification.

    Contains only primitive types that can be passed through torch.compile.
    Process groups and device meshes must be managed outside this spec.
    """
    num_shards: Tuple[int, ...]       # (2,) for 2-way splitting
    shard_dim: Tuple[int, ...]        # (4,) for final spatial axis (-1)
    rank: int                          # Current rank ID
    world_size: int                    # Total number of ranks
    shard_ind: Tuple[int, ...]        # (0,) or (1,) for 2-rank setup
    is_periodic: Tuple[bool, ...]     # (False,) for non-periodic boundaries

    def neighbor_ranks(self, dim_index: int) -> Tuple[int, int]:
        """
        Compute left and right neighbor ranks for the given dimension index.

        Args:
            dim_index: Index into shard_dim/num_shards tuples

        Returns:
            (left_rank, right_rank) tuple
        """
        num_shards_dim = self.num_shards[dim_index]
        shard_ind_dim = self.shard_ind[dim_index]

        # Compute neighbor shard indices
        left_shard_ind = shard_ind_dim - 1
        right_shard_ind = shard_ind_dim + 1

        # Wrap for periodic boundaries
        if self.is_periodic[dim_index]:
            left_shard_ind = left_shard_ind % num_shards_dim
            right_shard_ind = right_shard_ind % num_shards_dim

        # Convert to ranks using row-major ordering
        # For 1D sharding, rank = shard_ind
        # For multi-D, this is simplified - full implementation would use proper row-major
        left_rank = (self.rank // self.num_shards[dim_index]) * self.num_shards[dim_index] + left_shard_ind
        right_rank = (self.rank // self.num_shards[dim_index]) * self.num_shards[dim_index] + right_shard_ind

        # Handle negative indices (for non-periodic)
        if left_shard_ind < 0:
            left_rank = -1  # Invalid, but will be checked by is_leftmost
        if right_shard_ind >= num_shards_dim:
            right_rank = -1  # Invalid, but will be checked by is_rightmost

        return left_rank, right_rank
