"""
Experimental prototypes for compiler-compatible DistConv representations.

WARNING: This is an experimental namespace for PR3 prototypes.
APIs are unstable and may change or be removed.
"""

from .spec import DistConvSpec
from .dctensor import DCTensorExperimental
from . import ops  # Register custom operators
from . import resources  # Resource registry

__all__ = ["DistConvSpec", "DCTensorExperimental", "ops", "resources"]
