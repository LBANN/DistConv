import os

from .ddp import DistConvDDP
from .distconv import ParallelStrategy, forward_halo_exchange

# Conditional DCTensor import (rollout gate)
if os.getenv('DISTCONV_USE_EXPERIMENTAL', '0') == '1':
    # Use experimental compiler-compatible implementation
    from .experimental.dctensor import DCTensorExperimental as DCTensor
else:
    # Use original eager-only implementation
    from .distconv import DCTensor

__all__ = ["DistConvDDP", "DCTensor", "ParallelStrategy"]
