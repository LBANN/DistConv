from typing import Callable, Dict, List, Tuple
import itertools
from copy import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.functional import pad
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.nn.functional import pad
from torch.utils._pytree import tree_map


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.

    Credit: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/utils.py
    """
    return tuple(x for x in reversed(t) for _ in range(n))


class ParallelStrategy:
    """
    ParallelStrategy defines the strategy for distributing tensors across multiple devices
    for parallel computation. It includes the number of shards, the dimension along which
    the tensor is sharded, and the device mesh configuration.
    """

    def __init__(
        self,
        num_shards: int,
        shard_dim: int = 2,
        device_type: str = "cuda",
        is_periodic: bool = False,
        space_ndim: int = None,
    ):
        """
        Initialize the ParallelStrategy.

        Args:
            num_shards (int): The number of shards to divide the tensor into.
            shard_dim (int, optional): The dimension along which the tensor is sharded. Defaults to 2.
            device_type (str, optional): The device type to use with DeviceMesh. Defaults to "cuda".
            is_periodic (bool, optional): When true, adds checks to do circular padding on all boundaries.
            space_ndim (int, optional): When periodic, need to define spatial dimension to assign
                which boundaries are recieved by P2POp and which boundaries are on same rank.
        """
        self.num_shards = num_shards
        self.shard_dim = shard_dim
        self.space_ndim = space_ndim
        self.is_periodic = is_periodic
        if is_periodic:
            self.padding_mode = "circular"
        else:
            self.padding_mode = "constant"
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ddp_ranks = self.world_size // self.num_shards

        self.set_shard_inds()

        self.device_mesh = init_device_mesh(
            device_type,
            mesh_shape=(self.ddp_ranks, num_shards),
            mesh_dim_names=("ddp", "dc"),
        )

    def _check_gpu_map(self):
        expected_rank = self.find_rank_from_shard(self.shard_ind)
        assert (
            expected_rank == self.rank
        ), f"expected rank {expected_rank} does not match actual rank {self.rank} for shard {self.shard_ind}"

    def find_rank_from_shard(self, shard_ind):
        rank_to_ddp_index = self.rank // self.num_shards
        expected_rank = self.shard_to_gpu_map[str(shard_ind)]
        expected_rank += rank_to_ddp_index * self.num_shards
        return expected_rank

    def set_shard_inds(self):
        self.nonshard_dim = []
        if self.is_periodic:
            for i in range(self.space_ndim):
                if i == (self.shard_dim - 2):
                    pass
                else:
                    self.nonshard_dim.append(i + 2)

        self.shard_ind = self.rank % self.num_shards

        self.shard_to_gpu_map = {}
        for i in range(self.world_size):
            mesh_str = f"{i//self.ddp_ranks}"
            self.shard_to_gpu_map[mesh_str] = i // self.ddp_ranks

        self._check_gpu_map()


def check_is_distconv_transpose_supported(
    tensor_shard_dim: int,
    tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
) -> None:
    """
    Additional check if the distributed transpose convolution is supported with the given parameters.

    Args:
        tensor_shard_dim (int): The dimension along which the tensor is sharded.
        tensor (torch.Tensor): The input tensor.
        weight (torch.Tensor): The convolution kernel tensor.
        stride (List[int]): The stride of the convolution.
        padding (List[int]): The padding added to the input tensor.
        dilation (List[int]): The dilation applied to the kernel.

    Raises:
        Exception: If kernel size is even.
    """
    shard_dim = tensor_shard_dim - 2
    kernel_size = weight.size(tensor_shard_dim)
    if kernel_size % 2 == 0:
        raise Exception(
            f"DistConv Transpose: even kernel ({kernel_size}) not supported currently."
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
    if dilation[0] != 1:
        raise Exception(f"DistConv: dilation[0] ({dilation}) must be 1")
    if tensor.size(tensor_shard_dim) % stride[0] != 0:
        raise Exception(
            f"DistConv: input size ({tensor.shape}) must be divisible by stride ({stride})"
        )
    if kernel_size % 2 == 1:
        if (kernel_size // 2) != padding[0]:
            raise Exception(
                f'DistConv: when kernel size is odd, padding ({(kernel_size // 2)}) must be  equivalent to "same", but is {padding}, weight size is {weight.shape} for shard {tensor_shard_dim}'
            )
    else:
        if padding[0] != 0:
            raise Exception("DistConv: when kernel size is even, padding must be zero")
        if stride[0] % kernel_size != 0:
            raise Exception(
                f"DistConv: when kernel size is even ({weight.shape}), stride ({stride}) must be divisble by kernel size"
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
    is_periodic = parallel_strategy.is_periodic
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

    if is_periodic:
        ops = []
        if shard_ind == 0:
            # Receive halo from the previous rank and send their halo back
            shard_rhs = num_shards - 1
            rank_rhs = parallel_strategy.find_rank_from_shard(shard_rhs)
            ops += [
                dist.P2POp(dist.irecv, halo_minus, rank_rhs),
                dist.P2POp(dist.isend, inner_halo_minus.contiguous(), rank_rhs),
            ]
        if shard_ind == (num_shards - 1):
            # Receive halo from the previous rank and send their halo back
            shard_lhs = 0
            rank_lhs = parallel_strategy.find_rank_from_shard(shard_lhs)
            ops += [
                dist.P2POp(dist.irecv, halo_plus, rank_lhs),
                dist.P2POp(dist.isend, inner_halo_plus.contiguous(), rank_lhs),
            ]

        # check if P2POps on this rank exist (only true for outer boundary ranks)
        if len(ops) > 0:
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
    is_periodic = parallel_strategy.is_periodic
    nonshard_dim = parallel_strategy.nonshard_dim
    space_ndim = tensor.ndim - 2  # minus batch and channel dimensions
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

    if is_periodic:
        ops = []
        if shard_ind == 0:
            # Receive halo from the previous rank and send their halo back
            shard_rhs = num_shards - 1
            rank_rhs = parallel_strategy.find_rank_from_shard(shard_rhs)
            ops += [
                dist.P2POp(dist.irecv, recv_halo_minus, rank_rhs),
                dist.P2POp(dist.isend, send_halo_minus.contiguous(), rank_rhs),
            ]
        if shard_ind == (num_shards - 1):
            # Receive halo from the previous rank and send their halo back
            shard_lhs = 0
            rank_lhs = parallel_strategy.find_rank_from_shard(shard_lhs)
            ops += [
                dist.P2POp(dist.irecv, recv_halo_plus, rank_lhs),
                dist.P2POp(dist.isend, send_halo_plus.contiguous(), rank_lhs),
            ]

        # check if P2POps on this rank exist (only true for outer boundary ranks)
        if len(ops) > 0:
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

    # Crop the non-sharded edges to extract inner tensor
    pad_map = [-halo_size] * 2 * space_ndim
    pad_map[(2 * (shard_dim - 2))] = 0
    pad_map[(2 * (shard_dim - 2)) + 1] = 0
    inner_tensor = pad(inner_tensor, pad_map[::-1])
    for i in range(space_ndim):
        inner_halo_minus = inner_tensor.narrow(i + 2, 0, halo_size)
        inner_halo_plus = inner_tensor.narrow(i + 2, -halo_size, halo_size)
        outer_halo_minus = tensor.narrow(i + 2, 0, halo_size)
        outer_halo_plus = tensor.narrow(i + 2, -halo_size, halo_size)
        for j in range(space_ndim):
            if i == j:
                continue
            outer_halo_minus = outer_halo_minus.narrow(
                j + 2, halo_size, outer_halo_minus.size(j + 2) - 2 * halo_size
            )
            outer_halo_plus = outer_halo_plus.narrow(
                j + 2, halo_size, outer_halo_plus.size(j + 2) - 2 * halo_size
            )

        # When boundaries are not zero padded, accumulate periodic data into inner tensor
        if i + 2 in nonshard_dim and is_periodic:
            inner_halo_minus.add_(outer_halo_plus)
            inner_halo_plus.add_(outer_halo_minus)
    # Corners of sharded-dim need additional treatment in 3D
    if space_ndim == 3 and is_periodic:
        inner_tensor = accumulate_nonsharded_corners_3d(
            tensor, inner_tensor, halo_size, parallel_strategy
        )
    return inner_tensor


def get_corner_3d(
    tensor: torch.Tensor,
    halo_size: int,
    space_ndim: int,
    nonshard_dim: List[int],
    narrow_keys: List[bool],
):
    """
    Loop over each nonsharded dimension to find periodic corner data.

    Args:
        tensor (torch.Tensor): The tensor to narrow.
        halo_size (int): Size of halo to narrow.
        space_ndim (int): Number of spatial dimensions.
        nonshard_dim (list): Index of spatial dimensions which are not sharded.
        narrow_keys (list): Minus or positive side to narrow from (i.e. left or right) for each nonsharded dim.

    Returns:
        corner_tensor (torch.Tensor): Tensor narrowed to contain only corner values.
    """
    assert len(narrow_keys) == len(nonshard_dim)
    corner_tensor = tensor
    nonshard_counter = 0
    for i in range(space_ndim):
        if i + 2 in nonshard_dim:
            if narrow_keys[nonshard_counter] == True:
                start = -halo_size
            else:
                start = 0
            corner_tensor = corner_tensor.narrow(i + 2, start, halo_size)
            nonshard_counter += 1
    return corner_tensor


def accumulate_nonsharded_corners_3d(
    outer_tensor: torch.Tensor,
    inner_tensor: torch.Tensor,
    halo_size: int,
    parallel_strategy: ParallelStrategy,
):
    """
    Accumulates nonsharded corner values from outer tensor into corners of inner tensor for periodic convolutions.

    Args:
        outer_tensor (torch.Tensor): Original tensor.
        inner_tensor (torch.Tensor): Inner tensor after cropping the halo from each dimension.
        halo_size (int): Halo size to crop during narrowing.
        parallel_strategy (ParallelStrategy): Class containing information on sharded/non-sharded dimensions

    Returns:
        inner_tensor (torch.Tensor): Inner tensor with nonsharded corner values accumulated.
    """
    shard_dim = parallel_strategy.shard_dim
    nonshard_dim = parallel_strategy.nonshard_dim
    space_ndim = parallel_strategy.space_ndim
    assert space_ndim == 3

    # Pre-apply narrowing to sharded dimension such that the outer corners are the correct shape.
    outer_tensor = outer_tensor.narrow(
        shard_dim, halo_size, outer_tensor.size(shard_dim) - 2 * halo_size
    )
    # Initialise dictionary with positive and negative edge for 'inner' and 'outer' tensors
    # Assign False and True represent minus and plus edge of tensor (e.g. left and right)
    keys = [False, True]
    _combo = list(itertools.product(keys, keys))
    inner_dict = {}
    outer_dict = {}
    for key_i in keys:
        inner_dict[key_i] = {}
        outer_dict[key_i] = {}
        for key_j in keys:
            inner_dict[key_i][key_j] = {}
            outer_dict[key_i][key_j] = {}

    # Find corners for each combination of top/bottom and left/right
    for key_set_i in _combo:
        inner_dict[key_set_i[0]][key_set_i[1]] = get_corner_3d(
            inner_tensor, halo_size, space_ndim, nonshard_dim, key_set_i
        )
        outer_dict[key_set_i[0]][key_set_i[1]] = get_corner_3d(
            outer_tensor, halo_size, space_ndim, nonshard_dim, key_set_i
        )

    # Accumulate corner gradients into tensor
    for key_set_i in _combo:
        inner_dict[key_set_i[0]][key_set_i[1]].add_(
            outer_dict[not key_set_i[0]][not key_set_i[1]]
        )
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
    tensor, weight, bias, stride, padding, dilation, transpose, output_padding = args[
        :8
    ]

    # when doing double-backprop (e.g. torch.autograd.grad(...).backwards())
    # need to use core pytorch functionality
    if (
        weight.shape[-1] * dilation[-1] == tensor.shape[-1]
    ):  # NOTE: not rigorously tested
        args[0] = args[0].to_replicate()
        args[1] = args[1].to_replicate()
        return DCTensor(func(*args, **kwargs), tensor._parallel_strategy)

    # Extract the parallel strategy and shard dimension from the input tensor
    parallel_strategy = tensor._parallel_strategy
    shard_dim = parallel_strategy.shard_dim
    is_periodic = parallel_strategy.is_periodic
    shard_ind = parallel_strategy.shard_ind
    world_size = parallel_strategy.world_size

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

    # Update the arguments with the tensor including halos and adjusted padding
    padding_orig = copy(padding)
    output_padding_orig = copy(output_padding)
    if transpose:
        output_padding = forward_transpose_output_padding(
            tensor,
            dilation,
            stride,
            padding_orig,
            output_padding,
            kernel_size,
            parallel_strategy,
        )
    padding[shard_dim - 2] = 0
    tensor_with_halo = pad(
        tensor_with_halo,
        _reverse_repeat_tuple(padding, 2),
        mode=parallel_strategy.padding_mode,
    )
    for i in range(0, parallel_strategy.space_ndim):
        if transpose:
            padding[i] = dilation[i] * (kernel_size - 1)
        else:
            padding[i] = 0
    args[0] = tensor_with_halo
    args[4] = padding
    args[7] = output_padding

    # Save the tensor with its halo for the backward pass.
    tensor._tensor_with_halo = tensor_with_halo
    tensor._tensor = tensor_with_halo.narrow(
        shard_dim, halo_size, tensor.size(shard_dim)
    )

    # Perform the convolution operation
    out_tensor = func(*args, **kwargs)

    # handle output cropping for transpose
    if transpose:
        out_tensor = forward_transpose_padding(
            out_tensor,
            padding,
            dilation,
            stride,
            padding_orig,
            kernel_size,
            parallel_strategy,
        )

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
    (
        grad_out_tensor,
        input_tensor,
        weight,
        bias_size,
        stride,
        padding,
        dilation,
        transpose,
        output_padding,
    ) = args[:9]

    # Extract the parallel strategy and shard dimension from the gradient output tensor
    parallel_strategy = grad_out_tensor._parallel_strategy
    shard_dim = parallel_strategy.shard_dim
    is_periodic = parallel_strategy.is_periodic
    shard_ind = parallel_strategy.shard_ind
    world_size = parallel_strategy.world_size

    # Unwrap the underlying tensors from the DCTensors
    grad_out_tensor = grad_out_tensor._tensor
    input_torch_tensor = input_tensor._tensor

    # Check if the distributed convolution is supported with the given parameters
    check_is_distconv_supported(
        shard_dim, input_torch_tensor, weight, stride, padding, dilation
    )
    if transpose:
        check_is_distconv_transpose_supported(
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
    padding_orig = copy(padding)
    output_padding_orig = copy(output_padding)
    for i in range(0, parallel_strategy.space_ndim):
        if transpose:
            padding[i] = dilation[i] * (kernel_size - 1)
        else:
            padding[i] = 0
    padding[shard_dim - 2] = 0
    output_padding[shard_dim - 2] = 0

    if transpose:
        grad_out_tensor, padding = backward_transpose_padding(
            grad_out_tensor,
            input_tensor_with_halo,
            padding,
            dilation,
            stride,
            padding_orig,
            output_padding_orig,
            kernel_size,
            parallel_strategy,
        )
    args[0] = grad_out_tensor
    args[1] = input_tensor_with_halo
    args[5] = padding
    args[8] = output_padding

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


def forward_transpose_output_padding(
    input_tensor: torch.Tensor,
    dilation: List[int],
    stride: List[int],
    padding_orig: List[int],
    output_padding: List[int],
    kernel_size: int,
    parallel_strategy: ParallelStrategy,
) -> List[int]:
    """
    Modify output_padding in sharded dimension to be passed into transpose convolution.
    This adds an additional N rows/columns to the tensor with zeros, based on equations specified in
    https://arxiv.org/abs/1603.07285.
    This does not apply to the last shard along the sharded dimension, as this is naturally handled by
    the original output_padding.

    Args:
        input_tensor (torch.Tensor): Tensor whose size will be used to calculate the required output padding.
        dilation (list): Dilation applied to convolution kernel.
        stride (list): Stride used for convolution.
        padding_orig (list): Original padding defined in convolution layer, before being modified within distconv.
        output_padding (list): Output padding added to last row and column after transpose convolution to give
            the output tensor the correct shape.
        kernel_size (int): Width of convolution kernel.
        parallel_strategy (ParallelStrategy): The parallel strategy for distributing the tensor.

    Returns:
        List[int]: Output padding to be applied to the local tensor after convolution.
    """
    shard_dim = parallel_strategy.shard_dim
    world_size = parallel_strategy.world_size
    shard_ind = parallel_strategy.shard_ind

    for dim_i in range(input_tensor.ndim - 2):
        if dim_i == shard_dim - 2:
            if shard_ind + 1 < world_size:
                updated_padding = (
                    dilation[dim_i] * (kernel_size - 1) - padding_orig[dim_i]
                )
                output_padding[dim_i] += (
                    input_tensor.size(shard_dim) + 2 * updated_padding - kernel_size
                ) % stride[shard_dim - 2]
    return output_padding


def forward_transpose_padding(
    out_tensor: torch.Tensor,
    padding: List[int],
    dilation: List[int],
    stride: List[int],
    padding_orig: List[int],
    kernel_size: int,
    parallel_strategy: ParallelStrategy,
) -> torch.Tensor:
    """
    Crops tensor output from transpose convolution function to enforce shape to match reference (unsharded) solution.
    Based on equations specified in https://arxiv.org/abs/1603.07285.

    Args:
        out_tensor (torch.Tensor): Tensor after forward transpose convolution.
        padding (list): Padding passed into forward transpose convolution.
        dilation (list): Dilation applied to convolution kernel.
        stride (list): Stride used for convolution.
        padding_orig (list): Original padding defined in convolution layer, before being modified within distconv.
        kernel_size (int): Width of convolution kernel.
        parallel_strategy (ParallelStrategy): The parallel strategy for distributing the tensor.

    Returns:
        torch.Tensor: Cropped tensor with shape matching reference solution.
    """
    shard_dim = parallel_strategy.shard_dim
    world_size = parallel_strategy.world_size
    shard_ind = parallel_strategy.shard_ind

    pad_map = [0, 0] * (out_tensor.ndim - 2)
    for dim_i in range(out_tensor.ndim - 2):
        pad_map[(dim_i * 2)] -= (
            dilation[dim_i]
            * (kernel_size - 1 - padding_orig[dim_i])
            * (stride[dim_i] - 1)
        )
        pad_map[(dim_i * 2) + 1] -= (
            dilation[dim_i]
            * (kernel_size - 1 - padding_orig[dim_i])
            * (stride[dim_i] - 1)
        )

    # for layout e.g. NCHW, padding input should be (W0,W1,H0,H1)...
    pad_map = pad_map[::-1]
    out_tensor = pad(out_tensor, pad=pad_map, mode=parallel_strategy.padding_mode)
    return out_tensor


def backward_transpose_padding(
    grad_out_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    padding: List[int],
    dilation: List[int],
    stride: List[int],
    padding_orig: List[int],
    output_padding_orig: List[int],
    kernel_size: int,
    parallel_strategy: ParallelStrategy,
) -> [torch.Tensor, List[int]]:
    """
    Pads tensor before input to backward transpose convolution function to enforce shape to match reference (unsharded) solution.
    Based on equations specified in https://arxiv.org/abs/1603.07285.

    Args:
        grad_out_tensor (torch.Tensor): Output gradient tensor.
        input_tensor (torch.Tensor): Input tensor from distconv_forward.
        padding (list): Padding passed into forward transpose convolution.
        dilation (list): Dilation applied to convolution kernel.
        stride (list): Stride used for convolution.
        padding_orig (list): Original padding defined in convolution layer, before being modified within distconv.
        output_padding_orig (list): Original output padding defined in convolution layer, before being modified within distconv.
        kernel_size (int): Width of convolution kernel.
        parallel_strategy (ParallelStrategy): The parallel strategy for distributing the tensor.

    Returns:
        torch.Tensor: Cropped tensor with shape matching reference solution.
        List[int]: New padding arguments to be passed into backward transpose convolution.
    """

    shard_dim = parallel_strategy.shard_dim
    world_size = parallel_strategy.world_size
    shard_ind = parallel_strategy.shard_ind

    pad_map = [0, 0] * (grad_out_tensor.ndim - 2)
    output_pad_map = [0, 0] * (grad_out_tensor.ndim - 2)
    for dim_i in range(grad_out_tensor.ndim - 2):
        if dim_i == shard_dim - 2:
            if shard_ind < world_size - 1:
                # For strided transpose convolution, the shape is ambiguous.
                # See relationship 14 in paper in docstring.
                updated_padding = (
                    dilation[dim_i] * (kernel_size - 1) - padding_orig[dim_i]
                )
                output_pad_map[dim_i * 2] -= (
                    input_tensor.size(shard_dim) + 2 * updated_padding - kernel_size
                ) % stride[shard_dim - 2]
            else:
                # For last shard in dimension, we already have the original output padding.
                output_pad_map[dim_i * 2] -= output_padding_orig[shard_dim - 2]
            pad_map[dim_i * 2] += dilation[dim_i] * (kernel_size - 1)
            pad_map[dim_i * 2 + 1] += dilation[dim_i] * (kernel_size - 1)
        else:
            # Pre-pad the unsharded dimensions and update the padding variable
            pad_map[(dim_i * 2)] += padding[dim_i]
            pad_map[(dim_i * 2) + 1] += padding[dim_i]
            # Overwrite the padding to be passed into func() so we do not duplicate it.
            padding[dim_i] = 0
        # Apply this to all strided transpose convolutions to obtain correct output shape
        # (see relationship 14 in paper above).
        pad_map[(dim_i * 2)] += (
            dilation[dim_i]
            * (kernel_size - 1 - padding_orig[dim_i])
            * (stride[dim_i] - 1)
        )
        pad_map[(dim_i * 2) + 1] += (
            dilation[dim_i]
            * (kernel_size - 1 - padding_orig[dim_i])
            * (stride[dim_i] - 1)
        )

    # for layout e.g. NCHW, padding input should be (W0,W1,H0,H1)...
    pad_map = pad_map[::-1]
    output_pad_map = output_pad_map[::-1]
    # order here is opposite of forward pass (where output padding is
    # handled first, within the convolution op)
    grad_out_tensor = pad(grad_out_tensor, pad=pad_map, mode="constant")
    # padding is constant, even when periodic padding is used as
    # periodicity accounted for in backward_halo_exchange
    check_backward_output_pad_map(output_pad_map)
    grad_out_tensor = pad(grad_out_tensor, pad=output_pad_map, mode="constant")
    return grad_out_tensor, padding


def check_backward_output_pad_map(output_pad_map: List[int]):
    """
    Check that the output padding applied to the grad_out_tensor during backward pass is negative (cropping).

    Args:
        output_pad_map (list): Output cropping to be applied to the grad_out_tensor, expected length is 2*ndims.

    Raises:
        Exception: If positive values found in padding map.
    """
    output_pad_map = torch.tensor(output_pad_map)
    assert torch.all(
        output_pad_map <= 0
    ), f"output_pad_map expected to be negative in backward pass ({output_pad_map})"


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

    def to_replicate(self, shape=None, stride=None) -> torch.Tensor:
        """
        Convert the DCTensor to a simple replicated tensor.

        Returns:
            torch.Tensor: The full tensor.
        """
        device_mesh = self._parallel_strategy.device_mesh["dc"]
        shard_dim = self._parallel_strategy.shard_dim
        dtensor = DTensor.from_local(
            _ToTensor.apply(self),
            device_mesh=device_mesh,
            placements=[Shard(shard_dim)],
            shape=shape,
            stride=stride,
        ).redistribute(device_mesh=device_mesh, placements=[Replicate()])
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
