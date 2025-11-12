import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .distconv import ParallelStrategy


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

        Args:
            grad_reduction_factor (float, optional): Factor by which to scale the gradients.
            If None, it will be set to the number of DDP ranks in the parallel_strategy.
            Defaults to None.
            parallel_strategy (ParallelStrategy, optional): The parallel strategy used to
            set the grad_reduction_factor. Defaults to None.

        Raises:
            Exception: If both grad_reduction_factor and parallel_strategy are None.
        """
        super().__init__(*args, **kwargs)

        # Set the grad reduction factor using the parallel strategy if needed
        self.grad_reduction_factor = grad_reduction_factor
        if grad_reduction_factor is None:
            if parallel_strategy is None:
                raise Exception(
                    "Either grad_reduction_factor or parallel_strategy must be provided"
                )
            self.grad_reduction_factor = parallel_strategy.ddp_ranks
        # Note: DDP already scales the gradients by the world size
        self.grad_reduction_factor = dist.get_world_size() / self.grad_reduction_factor

        def scale_grads_hook(param):
            if param.grad is not None:
                param.grad.mul_(self.grad_reduction_factor)

        # Register the hook to scale the gradients
        for param in self._module_parameters:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(scale_grads_hook)
