from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.utils._pytree import tree_map


class ParallelStrategy:
    """
    ParallelStrategy defines the strategy for distributing tensors across multiple devices
    for parallel computation. It includes the number of shards, the dimension along which
    the tensor is sharded, and the device mesh configuration.
    """

    def __init__(self, num_shards: int, shard_dim: int = 2, device_type: str = "cuda"):
        """
        Initialize the ParallelStrategy.

        Args:
            num_shards (int): The number of shards to divide the tensor into.
            shard_dim (int, optional): The dimension along which the tensor is sharded. Defaults to 2.
            device_type (str, optional): The device type to use with DeviceMesh. Defaults to "cuda".
        """
        self.num_shards = num_shards
        self.shard_dim = shard_dim

        world_size = dist.get_world_size()
        self.ddp_ranks = world_size // num_shards
        self.shard_ind = dist.get_rank() % num_shards

        self.device_mesh = init_device_mesh(
            device_type,
            mesh_shape=(self.ddp_ranks, num_shards),
            mesh_dim_names=("ddp", "dc"),
        )


def check_is_distconv_supported(
    tensor_shard_dim: int,
    tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
) -> None:
    """
    Check if the distributed convolution is supported with the given parameters.

    Args:
        tensor_shard_dim (int): The dimension along which the tensor is sharded.
        tensor (torch.Tensor): The input tensor.
        weight (torch.Tensor): The convolution kernel tensor.
        stride (List[int]): The stride of the convolution.
        padding (List[int]): The padding added to the input tensor.
        dilation (List[int]): The dilation applied to the kernel.

    Raises:
        Exception: If dilation is not 1.
        Exception: If input size is not divisible by stride.
        Exception: If kernel size is odd and padding is not equivalent to "same".
        Exception: If kernel size is even and padding is not zero.
        Exception: If kernel size is even and stride is not divisible by kernel size.
    """
    shard_dim = tensor_shard_dim - 2
    kernel_size = weight.size(tensor_shard_dim)
    if dilation[shard_dim] != 1:
        raise Exception("DistConv: dilation must be 1")
    if tensor.size(tensor_shard_dim) % stride[shard_dim] != 0:
        raise Exception("DistConv: input size must be divisible by stride")
    if kernel_size % 2 == 1:
        if (kernel_size // 2) != padding[shard_dim]:
            raise Exception(
                'DistConv: when kernel size is odd, padding must be equivalent to "same"'
            )
    else:
        if padding[shard_dim] != 0:
            raise Exception("DistConv: when kernel size is even, padding must be zero")
        if stride[shard_dim] % kernel_size != 0:
            raise Exception(
                "DistConv: when kernel size is even, stride must be divisble by kernel size"
            )


def forward_halo_exchange(
    tensor: torch.Tensor, halo_size: int, parallel_strategy: ParallelStrategy
) -> torch.Tensor:
    """
    Perform forward halo exchange for distributed convolution.

    Args:
        tensor (torch.Tensor): The input tensor to exchange halos for.
        halo_size (int): The size of the halo to exchange.
        parallel_strategy (ParallelStrategy): The parallel strategy containing shard information.

    Returns:
        torch.Tensor: The tensor including the exchanged halos.
    """
    # Check if halo exchange is needed
    if halo_size == 0:
        return tensor

    # Extract parallel strategy parameters
    shard_dim = parallel_strategy.shard_dim
    num_shards = parallel_strategy.num_shards
    shard_ind = parallel_strategy.shard_ind
    rank = dist.get_rank()

    # Prepare halos for sending and receiving
    inner_halo_minus = tensor.narrow(shard_dim, 0, halo_size)
    inner_halo_plus = tensor.narrow(shard_dim, -halo_size, halo_size)
    halo_minus = torch.zeros_like(inner_halo_minus)
    halo_plus = torch.zeros_like(inner_halo_plus)

    # Define communication operations
    ops = []
    if shard_ind > 0:
        # Receive halo from the previous rank and send their halo back
        ops += [
            dist.P2POp(dist.irecv, halo_minus, rank - 1),
            dist.P2POp(dist.isend, inner_halo_minus.contiguous(), rank - 1),
        ]
    if shard_ind < (num_shards - 1):
        # Send halo to the next rank and receive their halo
        ops += [
            dist.P2POp(dist.isend, inner_halo_plus.contiguous(), rank + 1),
            dist.P2POp(dist.irecv, halo_plus, rank + 1),
        ]

    # Execute communication operations
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # Concatenate received halos with the original tensor
    tensor_with_halo = torch.cat([halo_minus, tensor, halo_plus], dim=shard_dim)

    return tensor_with_halo


def backward_halo_exchange(
    tensor: torch.Tensor, halo_size: int, parallel_strategy: ParallelStrategy
) -> torch.Tensor:
    """
    Perform backward halo exchange for distributed convolution.

    Args:
        tensor (torch.Tensor): The input tensor to exchange halos for.
        halo_size (int): The size of the halo to exchange.
        parallel_strategy (ParallelStrategy): The parallel strategy containing shard information.

    Returns:
        torch.Tensor: The tensor including halo contributions.
    """
    # Check if halo exchange is needed
    if halo_size == 0:
        return tensor

    # Extract parallel strategy parameters
    shard_dim = parallel_strategy.shard_dim
    num_shards = parallel_strategy.num_shards
    shard_ind = parallel_strategy.shard_ind
    rank = dist.get_rank()

    # Prepare halos for sending and receiving
    send_halo_minus = tensor.narrow(shard_dim, 0, halo_size)
    send_halo_plus = tensor.narrow(shard_dim, -halo_size, halo_size)
    recv_halo_minus = torch.zeros_like(send_halo_minus)
    recv_halo_plus = torch.zeros_like(send_halo_plus)

    # Define communication operations
    ops = []
    if shard_ind > 0:
        # Receive halo from previous rank and send their halo back
        ops += [
            dist.P2POp(dist.irecv, recv_halo_minus, rank - 1),
            dist.P2POp(dist.isend, send_halo_minus.contiguous(), rank - 1),
        ]
    if shard_ind < (num_shards - 1):
        # Send halo to the next rank and receive their halo
        ops += [
            dist.P2POp(dist.isend, send_halo_plus.contiguous(), rank + 1),
            dist.P2POp(dist.irecv, recv_halo_plus, rank + 1),
        ]

    # Execute communication operations
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # Accumulate received halos into the inner tensor
    inner_tensor = tensor.narrow(
        shard_dim, halo_size, tensor.size(shard_dim) - 2 * halo_size
    )
    inner_halo_minus = inner_tensor.narrow(shard_dim, 0, halo_size)
    inner_halo_plus = inner_tensor.narrow(shard_dim, -halo_size, halo_size)
    inner_halo_minus.add_(recv_halo_minus)
    inner_halo_plus.add_(recv_halo_plus)

    return inner_tensor


def distconv_forward(func: Callable, args: Tuple, kwargs: Dict) -> "DCTensor":
    """
    Perform the forward pass of the distributed convolution.

    Args:
        func (Callable): The convolution function to be applied.
        args (Tuple): The arguments to the convolution function.
        kwargs (Dict): The keyword arguments to the convolution function.

    Returns:
        DCTensor: The result of the convolution wrapped in a DCTensor.
    """
    # Convert args to a list for easier manipulation
    args = list(args)

    # Unpack the necessary arguments
    tensor, weight, bias, stride, padding, dilation = args[:6]

    # Extract the parallel strategy and shard dimension from the input tensor
    parallel_strategy = tensor._parallel_strategy
    shard_dim = parallel_strategy.shard_dim

    # Unwrap the underlying tensor from the DCTensor
    torch_tensor = tensor._tensor

    # Check if the distributed convolution is supported with the given parameters
    check_is_distconv_supported(
        shard_dim, torch_tensor, weight, stride, padding, dilation
    )

    # Determine the halo size for halo exchange
    kernel_size = weight.size(shard_dim)
    halo_size = kernel_size // 2 if (kernel_size % 2 == 1) else 0

    # Perform forward halo exchange to prepare the tensor for convolution
    tensor_with_halo = forward_halo_exchange(torch_tensor, halo_size, parallel_strategy)

    # Save the tensor with its halo for the backward pass.
    tensor._tensor_with_halo = tensor_with_halo
    tensor._tensor = tensor_with_halo.narrow(
        shard_dim, halo_size, tensor.size(shard_dim)
    )

    # Update the arguments with the tensor including halos and adjusted padding
    args[0] = tensor_with_halo
    padding[shard_dim - 2] = 0
    args[4] = padding

    # Perform the convolution operation
    out_tensor = func(*args, **kwargs)

    # Wrap the output tensor in a DCTensor and return it
    return DCTensor(out_tensor, parallel_strategy)


def distconv_backward(
    func: Callable, args: Tuple, kwargs: Dict
) -> Tuple["DCTensor", torch.Tensor, torch.Tensor]:
    """
    Perform the backward pass of the distributed convolution.

    Args:
        func (Callable): The convolution function to be applied.
        args (Tuple): The arguments to the convolution function.
        kwargs (Dict): The keyword arguments to the convolution function.

    Returns:
        Tuple[DCTensor, torch.Tensor, torch.Tensor]: The gradients with respect to the input tensor, weight, and bias.
    """
    # Convert args to a list for easier manipulation
    args = list(args)

    # Unpack the necessary arguments
    grad_out_tensor, input_tensor, weight, bias_size, stride, padding, dilation = args[
        :7
    ]

    # Extract the parallel strategy and shard dimension from the gradient output tensor
    parallel_strategy = grad_out_tensor._parallel_strategy
    shard_dim = parallel_strategy.shard_dim

    # Unwrap the underlying tensors from the DCTensors
    grad_out_tensor = grad_out_tensor._tensor
    input_torch_tensor = input_tensor._tensor

    # Check if the distributed convolution is supported with the given parameters
    check_is_distconv_supported(
        shard_dim, input_torch_tensor, weight, stride, padding, dilation
    )

    # Determine the halo size for halo exchange
    kernel_size = weight.size(shard_dim)
    halo_size = kernel_size // 2 if (kernel_size % 2 == 1) else 0

    # Get the input tensor including halos if available, otherwise perform forward halo exchange
    if input_tensor._tensor_with_halo is not None:
        input_tensor_with_halo = input_tensor._tensor_with_halo
    else:
        input_tensor_with_halo = forward_halo_exchange(
            input_torch_tensor, halo_size, parallel_strategy
        )

    # Update the arguments with the gradient output tensor, input tensor including halos, and adjusted padding
    args[0] = grad_out_tensor
    args[1] = input_tensor_with_halo
    padding[shard_dim - 2] = 0
    args[5] = padding

    # Perform the backward convolution operation
    grad_in_tensor, grad_weight, grad_bias = func(*args, **kwargs)

    if grad_in_tensor is not None:
        # Perform backward halo exchange to accumulate halo contributions into the gradient input tensor
        grad_in_tensor = backward_halo_exchange(
            grad_in_tensor, halo_size, parallel_strategy
        )

        # Wrap the gradient input tensor in a DCTensor
        grad_in_tensor = DCTensor(grad_in_tensor, parallel_strategy)

    # Return the gradients with respect to the input tensor, weight, and bias
    return grad_in_tensor, grad_weight, grad_bias


class DCTensor(torch.Tensor):
    """
    A subclass of torch.Tensor used for representing spatially sharded tensors.
    """

    _tensor: torch.Tensor
    _tensor_with_halo: torch.Tensor = None
    _parallel_strategy: ParallelStrategy

    @staticmethod
    def __new__(
        cls, tensor: torch.Tensor, parallel_strategy: ParallelStrategy
    ) -> "DCTensor":
        """
        Create a new DCTensor instance.

        Args:
            tensor (torch.Tensor): The underlying tensor.
            parallel_strategy (ParallelStrategy): The parallel strategy for distributing the tensor.

        Returns:
            DCTensor: A new instance of DCTensor.
        """
        dc_tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )
        dc_tensor._tensor = tensor
        dc_tensor._parallel_strategy = parallel_strategy

        return dc_tensor

    @classmethod
    def from_shard(
        cls, tensor: torch.Tensor, parallel_strategy: ParallelStrategy
    ) -> "DCTensor":
        """
        Create a DCTensor from a sharded tensor.

        Args:
            tensor (torch.Tensor): The sharded tensor.
            parallel_strategy (ParallelStrategy): The parallel strategy for distributing the tensor.

        Returns:
            DCTensor: A new instance of DCTensor.
        """
        return _FromTensor.apply(tensor, parallel_strategy)

    @classmethod
    def distribute(
        cls, tensor: torch.Tensor, parallel_strategy: ParallelStrategy
    ) -> "DCTensor":
        """
        Shard a tensor according to the given parallel strategy.

        Args:
            tensor (torch.Tensor): The tensor to be sharded.
            parallel_strategy (ParallelStrategy): The parallel strategy for sharding the tensor.

        Returns:
            DCTensor: A new instance of DCTensor with the tensor sharded according to the parallel strategy.
        """
        dtensor = distribute_tensor(
            tensor,
            device_mesh=parallel_strategy.device_mesh["dc"],
            placements=[Shard(parallel_strategy.shard_dim)],
        )
        return cls(dtensor.to_local(), parallel_strategy)

    def to_ddp(self) -> torch.Tensor:
        """
        Convert the DCTensor to a simple distributed data parallel tensor, resharding as necessary.

        Returns:
            torch.Tensor: The tensor resharded to the batch dimension.
        """
        device_mesh = self._parallel_strategy.device_mesh["dc"]
        shard_dim = self._parallel_strategy.shard_dim
        dtensor = DTensor.from_local(
            _ToTensor.apply(self),
            device_mesh=device_mesh,
            placements=[Shard(shard_dim)],
        ).redistribute(device_mesh=device_mesh, placements=[Shard(0)])
        return dtensor.to_local()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        Custom __torch_dispatch__ implementation for DCTensor.
        Intercepts forward/backward convolution ops and performs distributed convolution.
        For other ops, applies the parent class implementation.

        Args:
            func (Callable): The function to be dispatched.
            types (Tuple): The types of the arguments.
            args (Tuple, optional): The positional arguments for the function. Defaults to ().
            kwargs (Dict, optional): The keyword arguments for the function. Defaults to None.

        Returns:
            Any: The result of the dispatched function.
        """
        if kwargs is None:
            kwargs = {}

        if func is torch.ops.aten.convolution.default:
            return distconv_forward(func, args, kwargs)
        elif func is torch.ops.aten.convolution_backward.default:
            return distconv_backward(func, args, kwargs)

        def unwrap(t):
            if isinstance(t, DCTensor):
                assert (
                    self._parallel_strategy == t._parallel_strategy
                ), "Parallel strategy mismatch"
                return t._tensor
            else:
                return t

        def wrap(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, DCTensor):
                return DCTensor(t, self._parallel_strategy)
            else:
                return t

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

    def __repr__(self) -> str:
        """
        Return a string representation of the DCTensor.

        Returns:
            str: A string representation of the DCTensor.
        """
        return super().__repr__(tensor_contents=f"{self._tensor}")


class _FromTensor(Function):
    """
    Convert a torch.Tensor to a DCTensor.

    Args:
        tensor (torch.Tensor): The input tensor to be converted.
        parallel_strategy (ParallelStrategy): The parallel strategy for distributing the tensor.

    Returns:
        DCTensor: The converted DCTensor.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, parallel_strategy: ParallelStrategy):
        return DCTensor(tensor, parallel_strategy)

    @staticmethod
    def backward(ctx, grad: DCTensor):
        return _ToTensor.apply(grad), None


class _ToTensor(Function):
    """
    Convert a DCTensor back to a torch.Tensor.

    Args:
        dc_tensor (DCTensor): The DCTensor to be converted.

    Returns:
        torch.Tensor: The converted torch.Tensor.
    """

    @staticmethod
    def forward(ctx, dc_tensor: DCTensor):
        ctx.parallel_strategy = dc_tensor._parallel_strategy
        return dc_tensor._tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _FromTensor.apply(grad, ctx.parallel_strategy)
