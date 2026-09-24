import dataclasses
import math
import torch
import torch.distributed as dist
import torch.futures
from torch.nn.parallel import DistributedDataParallel

from .distconv import ParallelStrategy


@dataclasses.dataclass
class _DistConvCommHookState:
    """
    State for DistConv DDP communication hook.

    Attributes:
        process_group: DDP process group for gradient reduction.
        gradient_divisor: Number of data-parallel replicas (Q).
        world_size: Total number of ranks (W).
    """
    process_group: dist.ProcessGroup
    gradient_divisor: float
    world_size: int


def _distconv_comm_hook(
    state: _DistConvCommHookState,
    bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    DDP communication hook for DistConv gradient reduction.

    Achieves AVERAGE over data replicas (Q), SUM over spatial shards (W/Q).

    Algorithm:
        1. Pre-scale: divide by W to undo DDP's implicit averaging
        2. All-reduce: sum across all ranks
        3. Post-scale: multiply by W/Q to achieve correct normalization

    Result: synchronized_gradient = SUM(all_rank_gradients) / Q

    Args:
        state: Hook state with process group and scaling factors.
        bucket: DDP gradient bucket.

    Returns:
        Future[Tensor] that resolves to correctly scaled gradient buffer.
    """
    buf = bucket.buffer()

    # Step 1: Pre-scale to prepare for SUM reduction
    buf.div_(state.world_size)

    # Step 2: Asynchronous all-reduce (SUM)
    work = dist.all_reduce(
        buf,
        op=dist.ReduceOp.SUM,
        group=state.process_group,
        async_op=True
    )

    # Step 3: Get future and attach post-processing
    fut = work.get_future()

    def finalize(fut):
        # Extract result from future (returns list of tensors)
        result = fut.value()[0]

        # Apply final scaling: (W / Q)
        scale_factor = state.world_size / state.gradient_divisor
        result.mul_(scale_factor)

        return result

    return fut.then(finalize)


class DistConvDDP(DistributedDataParallel):
    def __init__(
        self,
        *args,
        grad_reduction_factor: float = None,
        parallel_strategy: ParallelStrategy = None,
        **kwargs,
    ):
        """
        Custom DistributedDataParallel class for DistConv that scales the gradients by the
        number of DDP ranks in the parallel strategy or a custom factor.

        Uses a bucket communication hook instead of per-parameter hooks for torch.compile()
        compatibility.

        Args:
            grad_reduction_factor (float, optional): Factor by which to divide the summed
                gradients (number of data-parallel replicas). If None, it will be set to
                the number of DDP ranks in the parallel_strategy. Defaults to None.
            parallel_strategy (ParallelStrategy, optional): The parallel strategy used to
                derive the grad_reduction_factor. Defaults to None.

        Raises:
            ValueError: If both grad_reduction_factor and parallel_strategy are None.
            ValueError: If grad_reduction_factor is not finite and positive.
            ValueError: If world_size is not evenly divisible by grad_reduction_factor.
        """
        # Initialize parent DDP first
        super().__init__(*args, **kwargs)

        # Determine gradient divisor (Q) - number of data-parallel replicas
        if grad_reduction_factor is None:
            if parallel_strategy is None:
                raise ValueError(
                    "Either grad_reduction_factor or parallel_strategy must be provided"
                )
            gradient_divisor = parallel_strategy.ddp_ranks
        else:
            gradient_divisor = grad_reduction_factor

        # Validate gradient divisor
        if not (isinstance(gradient_divisor, (int, float)) and
                gradient_divisor > 0 and
                math.isfinite(gradient_divisor)):
            raise ValueError(
                f"grad_reduction_factor must be finite and positive, got {gradient_divisor}"
            )

        # Get process group info
        pg = self.process_group
        world_size = dist.get_world_size(pg)

        # Validate topology compatibility
        if world_size % gradient_divisor != 0:
            raise ValueError(
                f"Process group world size ({world_size}) must be evenly divisible "
                f"by gradient_divisor ({gradient_divisor})"
            )

        # Store for external access (backward compatibility)
        # Note: DDP already scales gradients by world size, so we multiply by W/Q
        self.grad_reduction_factor = world_size / gradient_divisor
        self._gradient_divisor = gradient_divisor

        # Register bucket communication hook
        state = _DistConvCommHookState(
            process_group=pg,
            gradient_divisor=gradient_divisor,
            world_size=world_size
        )

        self.register_comm_hook(state, _distconv_comm_hook)
