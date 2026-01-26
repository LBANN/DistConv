from copy import copy
from math import prod
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.utils._pytree import tree_map


def infer_contiguous_format(tensor: torch.Tensor) -> torch.memory_format:
    """
    Infer the contiguous memory format for a tensor.

    Returns channels_last or channels_last_3d if the tensor is already in that format but otherwise
    defaults to contiguous_format. This is used for preserving the memory format through various
    communication operations (halo exchange and DCTensor redistribution).
    """
    if tensor.dim() == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    elif tensor.dim() == 5 and tensor.is_contiguous(
        memory_format=torch.channels_last_3d
    ):
        return torch.channels_last_3d
    return torch.contiguous_format


class ParallelStrategy:
    """
    ParallelStrategy defines the strategy for distributing tensors across multiple devices
    for parallel computation. It includes the number of shards, the dimension along which
    the tensor is sharded, and the device mesh configuration.
    """

    def __init__(
        self, num_shards: tuple, shard_dim: tuple = (2,), device_type: str = "cuda"
    ):
        """
        Initialize the ParallelStrategy.

        Args:
            num_shards (list): The number of shards to divide the tensor into.
            shard_dim (list, optional): The dimensions along which the tensor is sharded. Defaults to 2.
            device_type (str, optional): The device type to use with DeviceMesh. Defaults to "cuda".
        """
        self.num_shards = num_shards
        self.shard_dim = shard_dim
        self.total_num_shards = prod(self.num_shards)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ddp_ind = self.rank // self.total_num_shards

        self.ddp_ranks = self.world_size // self.total_num_shards
        # Convert linear rank to multi-dimensional shard indices (row-major order)
        self.shard_ind = []
        linear_idx = self.rank % self.total_num_shards
        stride = self.total_num_shards
        for num_shards_i in self.num_shards:
            stride //= num_shards_i
            self.shard_ind.append(linear_idx // stride)
            linear_idx %= stride

        self.distconv_dim_names = tuple([f"dc{i}" for i in range(len(self.shard_dim))])
        mesh_shape = (self.ddp_ranks,) + self.num_shards
        mesh_dim_names = ("ddp",) + self.distconv_dim_names

        self.device_mesh = init_device_mesh(
            device_type,
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

    def shard_to_rank(self, shard_ind):
        if isinstance(shard_ind, int):
            shard_ind = (shard_ind,)
        assert len(shard_ind) == len(self.num_shards)
        rank = 0
        stride = 1
        for shard_ind_dim_i, num_shards_dim_i in zip(
            reversed(shard_ind), reversed(self.num_shards)
        ):
            # modify shard ind for periodicity
            if shard_ind_dim_i < 0:
                shard_ind_dim_i = num_shards_dim_i - 1
            if shard_ind_dim_i == num_shards_dim_i:
                shard_ind_dim_i = 0
            rank += shard_ind_dim_i * stride
            stride *= num_shards_dim_i
        return rank + self.ddp_ind * self.total_num_shards

    @property
    def num_shards(self):
        return self._num_shards

    @num_shards.setter
    def num_shards(self, value):
        if isinstance(value, int):
            self._num_shards = (value,)
        elif isinstance(value, (tuple, list)):
            self._num_shards = tuple(value)
        else:
            raise TypeError(f"Unexpected num_shards type {type(value)}")
        self.total_num_shards = prod(self._num_shards)

    @property
    def shard_dim(self):
        return self._shard_dim

    @shard_dim.setter
    def shard_dim(self, value):
        if isinstance(value, int):
            self._shard_dim = (value,)
        elif isinstance(value, (tuple, list)):
            self._shard_dim = tuple(value)
        else:
            raise TypeError(f"Unexpected shard_dim type {type(value)}")
        # Validate length matches num_shards if already set
        if hasattr(self, "_num_shards") and len(self._shard_dim) != len(
            self._num_shards
        ):
            raise ValueError(
                f"shard_dim length ({len(self._shard_dim)}) must match "
                f"num_shards length ({len(self._num_shards)})"
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
    tensor: torch.Tensor,
    halo_size: int,
    parallel_strategy: ParallelStrategy,
    dim_index: int,
    is_periodic: bool = False,
) -> torch.Tensor:
    """
    Perform forward halo exchange for distributed convolution.

    Args:
        tensor (torch.Tensor): The input tensor to exchange halos for.
        halo_size (int): The size of the halo to exchange.
        parallel_strategy (ParallelStrategy): The parallel strategy containing shard information.
        dim_index (int): Index into parallel_strategy.shard_dim specifying which
            sharding dimension to perform the halo exchange for.
        is_periodic (bool, optional): Whether to use periodic (circular) boundary
            conditions for this dimension. Defaults to False.

    Returns:
        torch.Tensor: The tensor including the exchanged halos.
    """
    # Check if halo exchange is needed
    if halo_size == 0:
        return tensor

    # Detect memory format to preserve throughout operations
    memory_format = infer_contiguous_format(tensor)

    # Extract parallel strategy parameters
    shard_dim = parallel_strategy.shard_dim[dim_index]
    num_shards = parallel_strategy.num_shards[dim_index]
    shard_ind = parallel_strategy.shard_ind[dim_index]

    # Prepare halos for sending and receiving
    inner_halo_minus = tensor.narrow(shard_dim, 0, halo_size)
    inner_halo_plus = tensor.narrow(shard_dim, -halo_size, halo_size)
    halo_minus = torch.zeros_like(inner_halo_minus)
    halo_plus = torch.zeros_like(inner_halo_plus)

    # Define communication operations
    ops = []
    shard_minus = copy(parallel_strategy.shard_ind)
    shard_minus[dim_index] -= 1
    shard_plus = copy(parallel_strategy.shard_ind)
    shard_plus[dim_index] += 1
    minus_rank = parallel_strategy.shard_to_rank(shard_minus)
    plus_rank = parallel_strategy.shard_to_rank(shard_plus)
    if shard_ind > 0:
        # Receive halo from the previous rank and send their halo back
        ops += [
            dist.P2POp(dist.irecv, halo_minus, minus_rank),
            dist.P2POp(
                dist.isend,
                inner_halo_minus.contiguous(memory_format=memory_format),
                minus_rank,
            ),
        ]
    if shard_ind < (num_shards - 1) or is_periodic:
        # Send halo to the next rank and receive their halo
        ops += [
            dist.P2POp(
                dist.isend,
                inner_halo_plus.contiguous(memory_format=memory_format),
                plus_rank,
            ),
            dist.P2POp(dist.irecv, halo_plus, plus_rank),
        ]
    if shard_ind == 0 and is_periodic:
        ops += [
            dist.P2POp(dist.irecv, halo_minus, minus_rank),
            dist.P2POp(
                dist.isend,
                inner_halo_minus.contiguous(memory_format=memory_format),
                minus_rank,
            ),
        ]

    # Execute communication operations
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Concatenate received halos with the original tensor
    tensor_with_halo = torch.cat([halo_minus, tensor, halo_plus], dim=shard_dim)

    return tensor_with_halo


def backward_halo_exchange(
    tensor: torch.Tensor,
    halo_size: int,
    parallel_strategy: ParallelStrategy,
    dim_index: int,
    is_periodic: bool = False,
) -> torch.Tensor:
    """
    Perform backward halo exchange for distributed convolution.

    Args:
        tensor (torch.Tensor): The input tensor to exchange halos for.
        halo_size (int): The size of the halo to exchange.
        parallel_strategy (ParallelStrategy): The parallel strategy containing shard information.
        dim_index (int): Index into parallel_strategy.shard_dim specifying which
            sharding dimension to perform the halo exchange for.
        is_periodic (bool, optional): Whether to use periodic (circular) boundary
            conditions for this dimension. Defaults to False.

    Returns:
        torch.Tensor: The tensor including halo contributions.
    """
    # Check if halo exchange is needed
    if halo_size == 0:
        return tensor

    # Detect memory format to preserve throughout operations
    memory_format = infer_contiguous_format(tensor)

    # Extract parallel strategy parameters
    shard_dim = parallel_strategy.shard_dim[dim_index]
    num_shards = parallel_strategy.num_shards[dim_index]
    shard_ind = parallel_strategy.shard_ind[dim_index]

    # Prepare halos for sending and receiving
    send_halo_minus = tensor.narrow(shard_dim, 0, halo_size)
    send_halo_plus = tensor.narrow(shard_dim, -halo_size, halo_size)
    recv_halo_minus = torch.zeros_like(send_halo_minus)
    recv_halo_plus = torch.zeros_like(send_halo_plus)

    # Define communication operations
    ops = []
    shard_minus = copy(parallel_strategy.shard_ind)
    shard_minus[dim_index] -= 1
    shard_plus = copy(parallel_strategy.shard_ind)
    shard_plus[dim_index] += 1
    minus_rank = parallel_strategy.shard_to_rank(shard_minus)
    plus_rank = parallel_strategy.shard_to_rank(shard_plus)
    if shard_ind > 0:
        # find neighbouring shard, and which gpu it belongs to
        # Receive halo from previous rank and send their halo back
        ops += [
            dist.P2POp(dist.irecv, recv_halo_minus, minus_rank),
            dist.P2POp(
                dist.isend,
                send_halo_minus.contiguous(memory_format=memory_format),
                minus_rank,
            ),
        ]
    if shard_ind < (num_shards - 1) or is_periodic:
        # Send halo to the next rank and receive their halo
        ops += [
            dist.P2POp(
                dist.isend,
                send_halo_plus.contiguous(memory_format=memory_format),
                plus_rank,
            ),
            dist.P2POp(dist.irecv, recv_halo_plus, plus_rank),
        ]
    if shard_ind == 0 and is_periodic:
        ops += [
            dist.P2POp(dist.irecv, recv_halo_minus, minus_rank),
            dist.P2POp(
                dist.isend,
                send_halo_minus.contiguous(memory_format=memory_format),
                minus_rank,
            ),
        ]

    # Execute communication operations
    if ops:
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
    is_periodic = tensor._is_periodic
    for i, shard_dim_i in enumerate(shard_dim):
        if is_periodic[i]:
            assert padding[shard_dim_i - 2] == 0, (
                "Cannot zero-pad a tensor marked for periodic padding on the shard dimension"
            )
            padding[shard_dim_i - 2] = tensor._periodic_shard_padding[i]

    # Unwrap the underlying tensor from the DCTensor
    torch_tensor = tensor._tensor

    # Check if the distributed convolution is supported with the given parameters
    tensor_with_halo = torch_tensor
    halo_sizes = []
    for i, shard_dim_i in enumerate(shard_dim):
        check_is_distconv_supported(
            shard_dim_i, torch_tensor, weight, stride, padding, dilation
        )

        # Determine the halo size for halo exchange
        kernel_size = weight.size(shard_dim_i)
        halo_size = kernel_size // 2 if (kernel_size % 2 == 1) else 0
        halo_sizes.append(halo_size)

        # Perform forward halo exchange to prepare the tensor for convolution
        tensor_with_halo = forward_halo_exchange(
            tensor_with_halo, halo_size, parallel_strategy, i, is_periodic[i]
        )

        # Save the tensor with its halo for the backward pass.
        tensor._tensor_with_halo = tensor_with_halo

        # Update the arguments with the tensor including halos and adjusted padding
        args[0] = tensor_with_halo
        padding[shard_dim_i - 2] = 0
        args[4] = padding

    tensor._tensor = tensor_with_halo
    for i, shard_dim_i in enumerate(shard_dim):
        tensor._tensor = tensor._tensor.narrow(
            shard_dim_i, halo_sizes[i], tensor.size(shard_dim_i)
        )

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
    is_periodic = input_tensor._is_periodic
    for i, shard_dim_i in enumerate(shard_dim):
        if is_periodic[i]:
            assert padding[shard_dim_i - 2] == 0, (
                "Cannot zero-pad a tensor marked for periodic padding on the shard dimension"
            )
            padding[shard_dim_i - 2] = input_tensor._periodic_shard_padding[i]

    # Unwrap the underlying tensors from the DCTensors
    grad_out_tensor = grad_out_tensor._tensor
    input_torch_tensor = input_tensor._tensor

    # Check if the distributed convolution is supported with the given parameters
    halo_sizes = []
    for i, shard_dim_i in enumerate(shard_dim):
        check_is_distconv_supported(
            shard_dim_i, input_torch_tensor, weight, stride, padding, dilation
        )

        # Determine the halo size for halo exchange
        kernel_size = weight.size(shard_dim_i)
        halo_size = kernel_size // 2 if (kernel_size % 2 == 1) else 0
        halo_sizes.append(halo_size)
        padding[shard_dim_i - 2] = 0

    # Get the input tensor including halos if available, otherwise perform forward halo exchange
    if input_tensor._tensor_with_halo is not None:
        input_tensor_with_halo = input_tensor._tensor_with_halo
    else:
        input_tensor_with_halo = input_torch_tensor
        for i, shard_dim_i in enumerate(shard_dim):
            input_tensor_with_halo = forward_halo_exchange(
                input_tensor_with_halo,
                halo_sizes[i],
                parallel_strategy,
                i,
                is_periodic[i],
            )

    # Update the arguments with the gradient output tensor, input tensor including halos, and adjusted padding
    args[0] = grad_out_tensor
    args[1] = input_tensor_with_halo
    args[5] = padding

    # Perform the backward convolution operation
    grad_in_tensor, grad_weight, grad_bias = func(*args, **kwargs)

    if grad_in_tensor is not None:
        for i, shard_dim_i in enumerate(shard_dim):
            # Perform backward halo exchange to accumulate halo contributions into the gradient input tensor
            grad_in_tensor = backward_halo_exchange(
                grad_in_tensor, halo_sizes[i], parallel_strategy, i, is_periodic[i]
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
    _is_periodic: Tuple[bool, ...] = ()
    _periodic_shard_padding: Tuple[int, ...] = ()

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
        num_shard_dims = len(parallel_strategy.shard_dim)
        dc_tensor._is_periodic = tuple(False for _ in range(num_shard_dims))
        dc_tensor._periodic_shard_padding = tuple(0 for _ in range(num_shard_dims))

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
        # Preserve memory format through distribution
        memory_format = infer_contiguous_format(tensor)
        placements = [Shard(i) for i in parallel_strategy.shard_dim]
        device_mesh = parallel_strategy.device_mesh[
            parallel_strategy.distconv_dim_names
        ]
        dtensor = distribute_tensor(
            tensor,
            device_mesh=device_mesh,
            placements=placements,
        )
        local_tensor = dtensor.to_local()
        # DTensor may not preserve memory format, so convert back if needed
        if memory_format != torch.contiguous_format:
            local_tensor = local_tensor.contiguous(memory_format=memory_format)
        return cls(local_tensor, parallel_strategy)

    def to_ddp(self) -> torch.Tensor:
        """
        Convert the DCTensor to a simple distributed data parallel tensor, resharding as necessary.

        Returns:
            torch.Tensor: The tensor resharded to the batch dimension.
        """
        # Preserve memory format through redistribution
        memory_format = infer_contiguous_format(self._tensor)
        device_mesh = self._parallel_strategy.device_mesh[
            self._parallel_strategy.distconv_dim_names
        ]
        placements = [Shard(i) for i in self._parallel_strategy.shard_dim]
        dtensor = DTensor.from_local(
            _ToTensor.apply(self),
            device_mesh=device_mesh,
            placements=placements,
        ).redistribute(
            device_mesh=device_mesh, placements=[Shard(0)] * device_mesh.ndim
        )
        local_tensor = dtensor.to_local()
        # DTensor may not preserve memory format, so convert back if needed
        if memory_format != torch.contiguous_format:
            local_tensor = local_tensor.contiguous(memory_format=memory_format)
        return local_tensor

    def to_replicate(self) -> torch.Tensor:
        """
        Convert the DCTensor to a simple replicated tensor.

        Returns:
            torch.Tensor: The full tensor.
        """
        # Preserve memory format through redistribution
        memory_format = infer_contiguous_format(self._tensor)
        device_mesh = self._parallel_strategy.device_mesh[
            self._parallel_strategy.distconv_dim_names
        ]
        placements = [Shard(i) for i in self._parallel_strategy.shard_dim]
        dtensor = DTensor.from_local(
            _ToTensor.apply(self),
            device_mesh=device_mesh,
            placements=placements,
        ).redistribute(
            device_mesh=device_mesh, placements=[Replicate()] * device_mesh.ndim
        )
        local_tensor = dtensor.to_local()
        # DTensor may not preserve memory format, so convert back if needed
        if memory_format != torch.contiguous_format:
            local_tensor = local_tensor.contiguous(memory_format=memory_format)
        return local_tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Custom __torch_function__ implementation for DCTensor.
        Intercepts F.pad when padding_mode='circular' to handle distributed circular padding.

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

        # Intercept F.pad when padding_mode='circular'
        if func is torch.nn.functional.pad:
            input_tensor = args[0] if args else None
            mode = kwargs.get("mode", "constant")

            if isinstance(input_tensor, DCTensor) and mode == "circular":
                return cls._handle_circular_pad(func, args, kwargs)

        # For other functions, use default behavior
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def _handle_circular_pad(cls, func, args, kwargs):
        """
        Handle circular padding for DCTensor by applying normal padding to non-shard dimensions
        and marking the shard dimension for circular handling during conv operations.

        Args:
            func (Callable): The F.pad function.
            args (Tuple): The arguments to F.pad.
            kwargs (Dict): The keyword arguments to F.pad.

        Returns:
            DCTensor: The padded tensor with shard dimension marked for circular handling.
        """
        input_tensor = args[0]
        pad = args[1] if len(args) > 1 else kwargs.get("pad")

        parallel_strategy = input_tensor._parallel_strategy
        shard_dim = parallel_strategy.shard_dim
        pad_list = list(pad)
        shard_padding = [
            0,
        ] * len(shard_dim)
        is_periodic = [
            False,
        ] * len(shard_dim)

        # Calculate padding indices for shard dimension
        ndim = input_tensor.dim()
        for i, shard_dim_i in enumerate(shard_dim):
            shard_pad_start_idx = 2 * (ndim - 1 - shard_dim_i)
            shard_pad_end_idx = shard_pad_start_idx + 1

            # Extract and store shard dimension padding
            if len(pad_list) > shard_pad_end_idx:
                shard_pad_minus = pad_list[shard_pad_start_idx]
                shard_pad_plus = pad_list[shard_pad_end_idx]

                assert shard_pad_minus == shard_pad_plus, (
                    "Periodic padding must be symmetric on sharded dimension"
                )
                shard_padding[i] = shard_pad_minus
                is_periodic[i] = True

                # Disable padding on shard dimension for F.pad
                pad_list[shard_pad_start_idx] = 0
                pad_list[shard_pad_end_idx] = 0
            else:
                shard_padding[i] = 0
                is_periodic[i] = False

        # Call F.pad with modified padding (shard dim padding disabled)
        new_args = (_ToTensor.apply(input_tensor), tuple(pad_list)) + args[2:]
        partial_padded_tensor = func(*new_args, **kwargs)

        # Create result DCTensor with periodic flag and stored shard padding
        result: DCTensor = _FromTensor.apply(partial_padded_tensor, parallel_strategy)
        result._is_periodic = tuple(is_periodic)
        result._periodic_shard_padding = tuple(shard_padding)

        return result

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
                assert self._parallel_strategy == t._parallel_strategy, (
                    "Parallel strategy mismatch"
                )
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
