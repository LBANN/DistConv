"""
Distributed normalization (group and instance norm) for DCTensor. Layer norm is supported only
over unsharded dimensions, where it is local to each shard.

The kernels in the first section operate on local (unwrapped) tensors. ``groups`` is a list of
process groups, one per sharded dimension being normalized over; statistics are combined across
each group in turn. Since the combination is associative, reducing one mesh dimension at a time is
exact. The second section registers the DCTensor handlers that invoke these kernels.

Precision: the output and input gradient have the input dtype, and the parameter gradients have
the parameter dtype, as in native PyTorch. The computation and the statistics saved for the
backward pass are in at least float32 for any combination of dtypes, on every device. Native
PyTorch rounds the saved statistics to a reduced-precision input dtype in some cases (depending on
the op, device, and affine parameters), which corrupts the gradients when the variance is small
relative to the mean, so results are at least as accurate as native and may differ from it within
reduced-precision tolerance. Native PyTorch also rejects some mixed-dtype combinations that are
supported here.
"""

import inspect
from math import prod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distconv import (
    DCTensor,
    infer_contiguous_format,
    local_tensor,
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
    local_count = prod(x.size(d) for d in dims)
    m2 = var * local_count
    count = torch.full_like(mean, local_count)
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
    Distributed equivalent of ``aten.native_group_norm``. The computation and the returned
    statistics are in the accumulation dtype (at least fp32), and the output is in the input
    dtype.

    The statistics are saved for the backward pass, which reconstructs the normalized input from
    them. Unlike native group norm, they are never rounded to a reduced-precision input dtype,
    which badly corrupts the gradients when the variance is small relative to the mean (e.g., a
    bf16 mean of 1.0039 rounds to 1.0 while rstd is ~200), though the output is unaffected.

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

    return (
        out.to(x.dtype),
        mean.view(N, num_groups),
        rstd.view(N, num_groups),
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


# DCTensor handlers


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

    groups = parallel_strategy.shard_groups()
    out, mean, rstd = group_norm_forward(
        torch_tensor, weight, bias, num_groups, eps, groups
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

    groups = parallel_strategy.shard_groups()
    grad_in, grad_weight, grad_bias = group_norm_backward(
        local_tensor(grad_out),
        torch_tensor,
        mean,
        rstd,
        weight,
        num_groups,
        output_mask,
        groups,
    )
    if grad_in is not None:
        grad_in = grad_in.contiguous(
            memory_format=infer_contiguous_format(torch_tensor)
        )
        grad_in = DCTensor(grad_in, parallel_strategy)
    return grad_in, grad_weight, grad_bias


def _check_layer_norm_unsharded(tensor: DCTensor, normalized_shape: Sequence[int]):
    """
    Raise if any dimension normalized over by layer norm is sharded. Otherwise the statistics are
    local to each shard and the native op is exact.

    DCTensor does not track permutations, so after one the shard dimension may be stale. A
    sharded dimension permuted into the normalized dimensions is then still caught by the native
    op's shape validation, since normalized_shape gives its global size but the input has its
    local size.

    Raises:
        NotImplementedError: If a normalized dimension is sharded.
    """
    parallel_strategy = tensor._parallel_strategy
    ndim = tensor._tensor.dim()
    normalized_dims = range(max(ndim - len(normalized_shape), 0), ndim)
    sharded_dims = [
        dim
        for dim, num_shards in zip(
            parallel_strategy.shard_dim, parallel_strategy.num_shards
        )
        if num_shards > 1 and dim in normalized_dims
    ]
    if sharded_dims:
        raise NotImplementedError(
            f"DistConv: layer norm over sharded dimension(s) {sharded_dims} is not supported "
            "(only over unsharded dimensions, e.g., channels in a channels-last layout). To "
            "normalize over channels and spatial dimensions, use GroupNorm(1, C) instead (with a "
            "per-channel rather than elementwise affine; GroupNorm(1, C, affine=False) is "
            "equivalent to LayerNorm without one). If you need distributed layer norm, please "
            "open an issue at https://github.com/LBANN/DistConv/issues."
        )


@register_dispatch(torch.ops.aten.native_layer_norm.default)
def distconv_layer_norm_forward(
    func: Callable, args: Tuple, kwargs: Dict
) -> Tuple[DCTensor, torch.Tensor, torch.Tensor]:
    """
    Perform the forward pass of layer normalization over unsharded dimensions, which is local to
    each shard.

    Args:
        func (Callable): The native layer norm function.
        args (Tuple): The arguments to the layer norm function.
        kwargs (Dict): The keyword arguments to the layer norm function.

    Returns:
        Tuple[DCTensor, torch.Tensor, torch.Tensor]: The output, mean, and rstd.

    Raises:
        NotImplementedError: If a normalized dimension is sharded.
    """
    tensor, normalized_shape, weight, bias, eps = args
    _check_layer_norm_unsharded(tensor, normalized_shape)
    out, mean, rstd = func(tensor._tensor, normalized_shape, weight, bias, eps)
    return DCTensor(out, tensor._parallel_strategy), mean, rstd


@register_dispatch(torch.ops.aten.native_layer_norm_backward.default)
def distconv_layer_norm_backward(
    func: Callable, args: Tuple, kwargs: Dict
) -> Tuple[Optional[DCTensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Perform the backward pass of layer normalization over unsharded dimensions, which is local to
    each shard.

    Args:
        func (Callable): The native layer norm backward function.
        args (Tuple): The arguments to the layer norm backward function.
        kwargs (Dict): The keyword arguments to the layer norm backward function.

    Returns:
        Tuple[Optional[DCTensor], Optional[torch.Tensor], Optional[torch.Tensor]]: The gradients
        with respect to the input tensor, weight, and bias. The weight and bias gradients are
        local partial sums, to be summed across shards by gradient reduction (e.g., DistConvDDP).

    Raises:
        NotImplementedError: If a normalized dimension is sharded.
    """
    grad_out, tensor, normalized_shape, *rest = args
    _check_layer_norm_unsharded(tensor, normalized_shape)
    grad_in, grad_weight, grad_bias = func(
        local_tensor(grad_out), tensor._tensor, normalized_shape, *rest
    )
    if grad_in is not None:
        grad_in = DCTensor(grad_in, tensor._parallel_strategy)
    return grad_in, grad_weight, grad_bias


def _update_running_stat(
    running: torch.Tensor,
    batch_stat: torch.Tensor,
    momentum: float,
    nonempty: torch.Tensor,
):
    """
    Update a running statistic in place with a batch statistic, as in the native op. The update
    is computed in the dtype of the batch statistic (at least float32) and rounded to the running
    statistic's dtype once, as the native op does; rounding each term to a reduced-precision
    running statistic's dtype would compound the rounding errors.

    Args:
        running (torch.Tensor): The running statistic, updated in place.
        batch_stat (torch.Tensor): The batch statistic (at least float32).
        momentum (float): The weight of the batch statistic.
        nonempty (torch.Tensor): A boolean scalar tensor. If False (e.g., for a globally empty
            input, whose statistics are NaN), the running statistic is left unchanged. A tensor
            rather than a Python bool avoids host-device synchronization.
    """
    acc = torch.promote_types(running.dtype, batch_stat.dtype)
    updated = running.to(acc) * (1 - momentum) + batch_stat * momentum
    running.copy_(torch.where(nonempty, updated, running.to(acc)))


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
    channel, which reduces statistics across shards (the native implementation dispatches to
    backend-specific batch norm ops). Normalizing with running statistics (use_input_stats=False)
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

    # Call aten.native_group_norm rather than F.group_norm: under CUDA autocast, group_norm upcasts
    # its input to float32 (giving a float32 output), but instance norm keeps the input dtype, and
    # autocast does not affect native_group_norm.
    N, C = input.shape[:2]
    HxW = prod(input.shape[2:])
    out, mean, rstd = torch.native_group_norm(input, weight, bias, N, C, HxW, C, eps)

    # Update the running statistics with the batch-averaged instance statistics. As in the native
    # op, a globally empty spatial input (whose statistics are NaN) leaves them unchanged.
    if running_mean is not None or running_var is not None:
        with torch.no_grad():
            # Global spatial count, kept on the device (and created with a fill rather than a
            # host-to-device copy) to avoid synchronization
            count = torch.full((), HxW, dtype=torch.float64, device=input.device)
            for group in input._parallel_strategy.shard_groups():
                dist.all_reduce(count, group=group)
            nonempty = count > 0

            if running_mean is not None:
                _update_running_stat(running_mean, mean.mean(0), momentum, nonempty)
            if running_var is not None:
                var = rstd.double().pow(-2) - eps
                unbiased_var = var * count / (count - 1).clamp(min=1)
                _update_running_stat(
                    running_var, unbiased_var.mean(0), momentum, nonempty
                )

    return out
