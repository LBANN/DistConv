"""
Distributed normalization (group, layer, and instance norm) for DCTensor.

The kernels in the first section operate on local (unwrapped) tensors. ``groups`` is a list of
process groups, one per sharded dimension being normalized over; statistics are combined across
each group in turn. Since the combination is associative, reducing one mesh dimension at a time is
exact. The second section registers the DCTensor handlers that invoke these kernels.
"""

import inspect
from math import prod
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distconv import (
    DCTensor,
    ParallelStrategy,
    infer_contiguous_format,
    register_dispatch,
    register_function,
)
from .utils import signature_from_schema


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype for statistics (at least fp32)."""
    return torch.promote_types(dtype, torch.float32)


def _merge_var_mean(
    count: torch.Tensor, mean: torch.Tensor, m2: torch.Tensor, group: dist.ProcessGroup
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine per-shard (count, mean, M2) statistics across a process group using the parallel
    variance algorithm of Chan et al., which avoids the cancellation of a sum/sum-of-squares
    reduction.
    """
    stats = torch.stack([count, mean, m2])
    gathered = [torch.empty_like(stats) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, stats, group=group)
    counts, means, m2s = torch.stack(gathered).unbind(1)

    # Empty shards have NaN statistics and must not contribute
    nonempty = counts > 0
    means = torch.where(nonempty, means, 0)
    m2s = torch.where(nonempty, m2s, 0)

    total = counts.sum(0)
    mean = (counts * means).sum(0) / total
    m2 = m2s.sum(0) + (counts * (means - mean).square()).sum(0)
    return total, mean, m2


def sharded_var_mean(
    x: torch.Tensor, dims: Tuple[int, ...], groups: List[dist.ProcessGroup]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and biased variance of ``x`` over ``dims`` across all shards.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The global mean and variance (keepdim, accumulation dtype).
    """
    x = x.to(_acc_dtype(x.dtype))
    var, mean = torch.var_mean(x, dim=dims, keepdim=True, correction=0)
    count = torch.full_like(mean, prod(x.size(d) for d in dims))
    m2 = var * count
    for group in groups:
        count, mean, m2 = _merge_var_mean(count, mean, m2, group)
    return mean, m2 / count


def _sharded_input_grad(
    dxhat: torch.Tensor,
    xhat: torch.Tensor,
    rstd: torch.Tensor,
    dims: Tuple[int, ...],
    groups: List[dist.ProcessGroup],
) -> torch.Tensor:
    """
    Gradient of a normalization with respect to its input, given the gradient with respect to the
    normalized input ``dxhat``. The required means over ``dims`` are reduced across all shards.
    """
    count = prod(dxhat.size(d) for d in dims)
    sums = torch.stack(
        [
            dxhat.sum(dim=dims, keepdim=True),
            (dxhat * xhat).sum(dim=dims, keepdim=True),
            torch.full_like(rstd, count),
        ]
    )
    for group in groups:
        dist.all_reduce(sums, group=group)
    sum_dxhat, sum_dxhat_xhat, count = sums.unbind(0)
    return rstd * (dxhat - (sum_dxhat + xhat * sum_dxhat_xhat) / count)


def group_norm_forward(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    num_groups: int,
    eps: float,
    groups: List[dist.ProcessGroup],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Distributed equivalent of ``aten.native_group_norm``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The output, mean (N, G) and rstd (N, G).
    """
    N, C = x.shape[:2]
    xg = x.unflatten(1, (num_groups, C // num_groups))
    dims = tuple(range(2, xg.dim()))
    mean, var = sharded_var_mean(xg, dims, groups)
    rstd = torch.rsqrt(var + eps)

    out = ((xg - mean) * rstd).flatten(1, 2)
    affine_shape = (1, C) + (1,) * (x.dim() - 2)
    if weight is not None:
        out = out * weight.view(affine_shape)
    if bias is not None:
        out = out + bias.view(affine_shape)

    # Match the native op, which returns statistics in the input dtype
    return (
        out.to(x.dtype),
        mean.view(N, num_groups).to(x.dtype),
        rstd.view(N, num_groups).to(x.dtype),
    )


def group_norm_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    num_groups: int,
    output_mask: Sequence[bool],
    groups: List[dist.ProcessGroup],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Distributed equivalent of ``aten.native_group_norm_backward``. The weight and bias gradients
    are local partial sums, to be summed across shards by gradient reduction (e.g., DistConvDDP).
    """
    acc = _acc_dtype(x.dtype)
    N, C = x.shape[:2]
    xg = x.unflatten(1, (num_groups, C // num_groups)).to(acc)
    dyg = grad_out.unflatten(1, (num_groups, C // num_groups)).to(acc)
    stat_shape = (N, num_groups) + (1,) * (xg.dim() - 2)
    rstd = rstd.view(stat_shape).to(acc)
    xhat = (xg - mean.view(stat_shape).to(acc)) * rstd

    grad_in = grad_weight = grad_bias = None
    param_dtype = weight.dtype if weight is not None else x.dtype
    if output_mask[0]:
        dxhat = dyg
        if weight is not None:
            weight_shape = (num_groups, C // num_groups) + (1,) * (xg.dim() - 3)
            dxhat = dxhat * weight.view(weight_shape)
        dims = tuple(range(2, xg.dim()))
        grad_in = _sharded_input_grad(dxhat, xhat, rstd, dims, groups)
        grad_in = grad_in.flatten(1, 2).to(x.dtype)
    param_dims = (0,) + tuple(range(3, xg.dim()))
    if output_mask[1] and weight is not None:
        grad_weight = (dyg * xhat).sum(dim=param_dims).flatten().to(param_dtype)
    if output_mask[2]:
        grad_bias = dyg.sum(dim=param_dims).flatten().to(param_dtype)
    return grad_in, grad_weight, grad_bias


def layer_norm_forward(
    x: torch.Tensor,
    normalized_ndim: int,
    eps: float,
    groups: List[dist.ProcessGroup],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Distributed equivalent of ``aten.native_layer_norm`` (without affine parameters).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The output, mean and rstd.
    """
    dims = tuple(range(x.dim() - normalized_ndim, x.dim()))
    mean, var = sharded_var_mean(x, dims, groups)
    rstd = torch.rsqrt(var + eps)
    out = (x - mean) * rstd

    # Match the native op, which returns statistics in the accumulation dtype
    return out.to(x.dtype), mean, rstd


def layer_norm_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    normalized_ndim: int,
    groups: List[dist.ProcessGroup],
) -> torch.Tensor:
    """
    Distributed equivalent of ``aten.native_layer_norm_backward`` (without affine parameters).

    Returns:
        torch.Tensor: The gradient with respect to the input.
    """
    acc = _acc_dtype(x.dtype)
    rstd = rstd.to(acc)
    xhat = (x.to(acc) - mean.to(acc)) * rstd
    dims = tuple(range(x.dim() - normalized_ndim, x.dim()))
    return _sharded_input_grad(grad_out.to(acc), xhat, rstd, dims, groups).to(x.dtype)


# DCTensor handlers


def _local(tensor):
    """Unwrap a DCTensor to its local tensor, passing anything else through."""
    return tensor._tensor if isinstance(tensor, DCTensor) else tensor


def _shard_groups(
    parallel_strategy: ParallelStrategy, dim_indices: Iterable[int]
) -> List[dist.ProcessGroup]:
    """
    Get the process groups for the given sharding dimensions (indices into
    parallel_strategy.shard_dim), skipping dimensions that are not split.
    """
    # Mesh dimension 0 is the DDP dimension
    return [
        parallel_strategy.device_mesh.get_group(i + 1)
        for i in dim_indices
        if parallel_strategy.num_shards[i] > 1
    ]


@register_dispatch(torch.ops.aten.native_group_norm.default)
def distconv_group_norm_forward(
    func: Callable, args: Tuple, kwargs: Dict
) -> Tuple[DCTensor, torch.Tensor, torch.Tensor]:
    """
    Perform the forward pass of group normalization with statistics reduced across all shards.

    Args:
        func (Callable): The native group norm function (unused).
        args (Tuple): The arguments to the group norm function.
        kwargs (Dict): The keyword arguments to the group norm function.

    Returns:
        Tuple[DCTensor, torch.Tensor, torch.Tensor]: The output, mean, and rstd.
    """
    tensor, weight, bias, _, _, _, num_groups, eps = args
    parallel_strategy = tensor._parallel_strategy
    torch_tensor = tensor._tensor

    groups = _shard_groups(parallel_strategy, range(len(parallel_strategy.shard_dim)))
    out, mean, rstd = group_norm_forward(
        torch_tensor, _local(weight), _local(bias), num_groups, eps, groups
    )
    out = out.contiguous(memory_format=infer_contiguous_format(torch_tensor))
    return DCTensor(out, parallel_strategy), mean, rstd


@register_dispatch(torch.ops.aten.native_group_norm_backward.default)
def distconv_group_norm_backward(
    func: Callable, args: Tuple, kwargs: Dict
) -> Tuple[DCTensor, torch.Tensor, torch.Tensor]:
    """
    Perform the backward pass of group normalization with statistics reduced across all shards.

    Args:
        func (Callable): The native group norm backward function (unused).
        args (Tuple): The arguments to the group norm backward function.
        kwargs (Dict): The keyword arguments to the group norm backward function.

    Returns:
        Tuple[DCTensor, torch.Tensor, torch.Tensor]: The gradients with respect to the input
        tensor, weight, and bias. The weight and bias gradients are local partial sums.
    """
    grad_out, tensor, mean, rstd, weight, _, _, _, num_groups, output_mask = args
    parallel_strategy = tensor._parallel_strategy
    torch_tensor = tensor._tensor

    groups = _shard_groups(parallel_strategy, range(len(parallel_strategy.shard_dim)))
    grad_in, grad_weight, grad_bias = group_norm_backward(
        _local(grad_out),
        torch_tensor,
        _local(mean),
        _local(rstd),
        _local(weight),
        num_groups,
        output_mask,
        groups,
    )
    if grad_in is not None:
        grad_in = grad_in.contiguous(memory_format=infer_contiguous_format(torch_tensor))
        grad_in = DCTensor(grad_in, parallel_strategy)
    return grad_in, grad_weight, grad_bias


def _layer_norm_shard_groups(
    tensor: DCTensor, normalized_shape: Sequence[int]
) -> List[dist.ProcessGroup]:
    """
    Determine the process groups to reduce layer norm statistics across, i.e., those of the
    sharded dimensions being normalized over. normalized_shape must use the global (unsharded)
    size of sharded dimensions.

    Note: DCTensor does not track permutations of its dimensions, so this relies on the tensor's
    shard dimensions being accurate.

    Returns:
        List[dist.ProcessGroup]: The process groups to reduce statistics across.
    """
    parallel_strategy = tensor._parallel_strategy
    torch_tensor = tensor._tensor
    axis = torch_tensor.dim() - len(normalized_shape)

    dim_indices = []
    for i, shard_dim_i in enumerate(parallel_strategy.shard_dim):
        if shard_dim_i < axis or parallel_strategy.num_shards[i] == 1:
            continue

        # Gather the local sizes of all shards along this dimension to get the global size
        group = parallel_strategy.device_mesh.get_group(i + 1)
        global_size = torch.tensor(
            [torch_tensor.size(shard_dim_i)], dtype=torch.int64, device=torch_tensor.device
        )
        dist.all_reduce(global_size, group=group)
        global_size = global_size.item()

        if normalized_shape[shard_dim_i - axis] != global_size:
            raise ValueError(
                f"DistConv: layer norm normalized_shape {tuple(normalized_shape)} does not match "
                f"the global size ({global_size}) of sharded dimension {shard_dim_i}"
            )
        dim_indices.append(i)

    return _shard_groups(parallel_strategy, dim_indices)


@register_dispatch(torch.ops.aten.native_layer_norm.default)
def distconv_layer_norm_forward(
    func: Callable, args: Tuple, kwargs: Dict
) -> Tuple[DCTensor, torch.Tensor, torch.Tensor]:
    """
    Perform the forward pass of layer normalization, reducing statistics across shards when the
    normalized dimensions are sharded. Affine parameters are not supported.

    Args:
        func (Callable): The native layer norm function.
        args (Tuple): The arguments to the layer norm function.
        kwargs (Dict): The keyword arguments to the layer norm function.

    Returns:
        Tuple[DCTensor, torch.Tensor, torch.Tensor]: The output, mean, and rstd.

    Raises:
        NotImplementedError: If weight or bias is given.
    """
    tensor, normalized_shape, weight, bias, eps = args
    if weight is not None or bias is not None:
        raise NotImplementedError(
            "DistConv: layer norm with elementwise affine parameters is not supported. Use "
            "elementwise_affine=False, or GroupNorm(1, C) for normalization over channels and "
            "spatial dimensions with a per-channel affine."
        )
    parallel_strategy = tensor._parallel_strategy
    torch_tensor = tensor._tensor

    groups = _layer_norm_shard_groups(tensor, normalized_shape)
    if groups:
        out, mean, rstd = layer_norm_forward(
            torch_tensor, len(normalized_shape), eps, groups
        )
    else:
        # The normalized dimensions are not sharded, so the native op is exact
        out, mean, rstd = func(torch_tensor, normalized_shape, None, None, eps)
    out = out.contiguous(memory_format=infer_contiguous_format(torch_tensor))
    return DCTensor(out, parallel_strategy), mean, rstd


@register_dispatch(torch.ops.aten.native_layer_norm_backward.default)
def distconv_layer_norm_backward(
    func: Callable, args: Tuple, kwargs: Dict
) -> Tuple[DCTensor, None, None]:
    """
    Perform the backward pass of layer normalization, reducing statistics across shards when the
    normalized dimensions are sharded. Affine parameters are not supported.

    Args:
        func (Callable): The native layer norm backward function.
        args (Tuple): The arguments to the layer norm backward function.
        kwargs (Dict): The keyword arguments to the layer norm backward function.

    Returns:
        Tuple[DCTensor, None, None]: The gradient with respect to the input tensor (and no weight
        or bias gradients).
    """
    grad_out, tensor, normalized_shape, mean, rstd, _, _, output_mask = args
    if not output_mask[0]:
        return None, None, None
    parallel_strategy = tensor._parallel_strategy
    torch_tensor = tensor._tensor
    grad_out, mean, rstd = _local(grad_out), _local(mean), _local(rstd)

    groups = _layer_norm_shard_groups(tensor, normalized_shape)
    if groups:
        grad_in = layer_norm_backward(
            grad_out, torch_tensor, mean, rstd, len(normalized_shape), groups
        )
    else:
        grad_in, _, _ = func(
            grad_out,
            torch_tensor,
            normalized_shape,
            mean,
            rstd,
            None,
            None,
            [True, False, False],
        )
    grad_in = grad_in.contiguous(memory_format=infer_contiguous_format(torch_tensor))
    return DCTensor(grad_in, parallel_strategy), None, None


_INSTANCE_NORM_SIGNATURES = {
    F.instance_norm: inspect.signature(F.instance_norm),
    torch.instance_norm: signature_from_schema(torch.ops.aten.instance_norm.default),
}


@register_function(F.instance_norm, torch.instance_norm)
def distconv_instance_norm(
    func: Callable, types: Tuple, args: Tuple, kwargs: Dict
) -> DCTensor:
    """
    Perform instance normalization of a DCTensor as group normalization with one group per
    channel, which reduces statistics across shards. (The native implementation dispatches to
    backend-specific batch norm ops.) Normalizing with running statistics (use_input_stats=False)
    is purely local and falls back to the default behavior.

    Args:
        func (Callable): The F.instance_norm or torch.instance_norm function.
        types (Tuple): The types of the arguments.
        args (Tuple): The positional arguments to the function.
        kwargs (Dict): The keyword arguments to the function.

    Returns:
        DCTensor: The normalized tensor.
    """
    bound = _INSTANCE_NORM_SIGNATURES[func].bind(*args, **kwargs)
    bound.apply_defaults()
    input = bound.arguments["input"]
    if not (isinstance(input, DCTensor) and bound.arguments["use_input_stats"]):
        return NotImplemented
    weight, bias = bound.arguments["weight"], bound.arguments["bias"]
    running_mean = bound.arguments["running_mean"]
    running_var = bound.arguments["running_var"]
    momentum, eps = bound.arguments["momentum"], bound.arguments["eps"]

    N, C = input.shape[:2]
    HxW = prod(input.shape[2:])
    # Disable __torch_function__ so the (replicated) statistics are not converted to DCTensor
    with torch._C.DisableTorchFunctionSubclass():
        out, mean, rstd = torch.native_group_norm(input, weight, bias, N, C, HxW, C, eps)

    # Update the running statistics with the batch-averaged instance statistics
    if running_mean is not None or running_var is not None:
        with torch.no_grad():
            parallel_strategy = input._parallel_strategy
            count = torch.tensor(float(HxW), dtype=torch.float64, device=input.device)
            for group in _shard_groups(
                parallel_strategy, range(len(parallel_strategy.shard_dim))
            ):
                dist.all_reduce(count, group=group)
            count = count.item()
            if running_mean is not None:
                running_mean.mul_(1 - momentum).add_(
                    mean.mean(0).to(running_mean.dtype), alpha=momentum
                )
            if running_var is not None:
                var = rstd.double().pow(-2) - eps
                unbiased_var = var * count / max(count - 1, 1)
                running_var.mul_(1 - momentum).add_(
                    unbiased_var.mean(0).to(running_var.dtype), alpha=momentum
                )

    return out
