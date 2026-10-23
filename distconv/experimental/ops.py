"""
Custom operators for halo exchange in distributed convolution.

These operators use primitive arguments (int, bool) instead of custom objects
to ensure compatibility with PyTorch's operator schema system.
"""

import torch
import torch.distributed as dist


@torch.library.custom_op("distconv_experimental::halo_exchange_fwd", mutates_args=())
def halo_exchange_fwd(
    tensor: torch.Tensor,
    shard_dim: int,
    halo_size: int,
    left_rank: int,
    right_rank: int,
    is_leftmost: bool,
    is_rightmost: bool,
) -> torch.Tensor:
    """
    Forward halo exchange: N → N + 2*halo_size.

    Exchanges boundary regions with neighboring ranks. Returns a new tensor
    with halos concatenated: [left_halo, tensor, right_halo].

    Args:
        tensor: Local tensor shard
        shard_dim: Dimension to exchange halos on
        halo_size: Width of halo region
        left_rank: Rank of left neighbor
        right_rank: Rank of right neighbor
        is_leftmost: True if no left neighbor (physical boundary)
        is_rightmost: True if no right neighbor (physical boundary)

    Returns:
        Tensor with shape increased by 2*halo_size on shard_dim
    """
    # Extract inner halos to send
    inner_halo_left = tensor.narrow(shard_dim, 0, halo_size)
    inner_halo_right = tensor.narrow(shard_dim, -halo_size, halo_size)

    # Allocate receive buffers (zero-filled for boundaries)
    halo_left = torch.zeros_like(inner_halo_left)
    halo_right = torch.zeros_like(inner_halo_right)

    # Build P2P operations
    ops = []
    if not is_leftmost:
        ops += [
            dist.P2POp(dist.irecv, halo_left, left_rank),
            dist.P2POp(dist.isend, inner_halo_left.contiguous(), left_rank),
        ]
    if not is_rightmost:
        ops += [
            dist.P2POp(dist.isend, inner_halo_right.contiguous(), right_rank),
            dist.P2POp(dist.irecv, halo_right, right_rank),
        ]

    # Execute P2P communication
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Concatenate: [halo_left, tensor, halo_right]
    return torch.cat([halo_left, tensor, halo_right], dim=shard_dim)


@halo_exchange_fwd.register_fake
def _(tensor, shard_dim, halo_size, left_rank, right_rank, is_leftmost, is_rightmost):
    """Fake implementation for shape inference."""
    output_shape = list(tensor.shape)
    output_shape[shard_dim] += 2 * halo_size
    return tensor.new_empty(output_shape)


def _setup_context_fwd(ctx, inputs, output):
    """Save metadata for backward pass."""
    tensor, shard_dim, halo_size, left_rank, right_rank, is_leftmost, is_rightmost = inputs
    ctx.shard_dim = shard_dim
    ctx.halo_size = halo_size
    ctx.left_rank = left_rank
    ctx.right_rank = right_rank
    ctx.is_leftmost = is_leftmost
    ctx.is_rightmost = is_rightmost


def _backward_fwd(ctx, grad_output):
    """Backward of forward halo exchange."""
    grad_input = torch.ops.distconv_experimental.halo_exchange_bwd(
        grad_output,
        ctx.shard_dim,
        ctx.halo_size,
        ctx.left_rank,
        ctx.right_rank,
        ctx.is_leftmost,
        ctx.is_rightmost,
    )
    # Return gradients for all inputs (None for non-tensor args)
    return grad_input, None, None, None, None, None, None


torch.library.register_autograd(
    "distconv_experimental::halo_exchange_fwd",
    _backward_fwd,
    setup_context=_setup_context_fwd,
)


@torch.library.custom_op("distconv_experimental::halo_exchange_bwd", mutates_args=())
def halo_exchange_bwd(
    grad_output: torch.Tensor,
    shard_dim: int,
    halo_size: int,
    left_rank: int,
    right_rank: int,
    is_leftmost: bool,
    is_rightmost: bool,
) -> torch.Tensor:
    """
    Backward halo exchange: N + 2*halo_size → N.

    Accumulates gradient contributions from neighboring halos. Returns
    fresh storage (does not mutate grad_output).

    Args:
        grad_output: Gradient with halos (shape N + 2*halo_size)
        shard_dim: Dimension halos are on
        halo_size: Width of halo region
        left_rank: Rank of left neighbor
        right_rank: Rank of right neighbor
        is_leftmost: True if no left neighbor
        is_rightmost: True if no right neighbor

    Returns:
        Gradient without halos (shape N), with halo contributions accumulated
    """
    # Extract halos to send back
    send_halo_left = grad_output.narrow(shard_dim, 0, halo_size)
    send_halo_right = grad_output.narrow(shard_dim, -halo_size, halo_size)

    # Allocate receive buffers
    recv_halo_left = torch.zeros_like(send_halo_left)
    recv_halo_right = torch.zeros_like(send_halo_right)

    # Build P2P operations
    ops = []
    if not is_leftmost:
        ops += [
            dist.P2POp(dist.irecv, recv_halo_left, left_rank),
            dist.P2POp(dist.isend, send_halo_left.contiguous(), left_rank),
        ]
    if not is_rightmost:
        ops += [
            dist.P2POp(dist.isend, send_halo_right.contiguous(), right_rank),
            dist.P2POp(dist.irecv, recv_halo_right, right_rank),
        ]

    # Execute P2P communication
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Clone interior to fresh storage (functional output)
    grad_input = grad_output.narrow(
        shard_dim, halo_size, grad_output.size(shard_dim) - 2 * halo_size
    ).clone(memory_format=torch.contiguous_format)

    # Accumulate received halo contributions into fresh grad_input
    grad_input_left = grad_input.narrow(shard_dim, 0, halo_size)
    grad_input_right = grad_input.narrow(shard_dim, -halo_size, halo_size)
    grad_input_left.add_(recv_halo_left)
    grad_input_right.add_(recv_halo_right)

    return grad_input


@halo_exchange_bwd.register_fake
def _(grad_output, shard_dim, halo_size, left_rank, right_rank, is_leftmost, is_rightmost):
    """Fake implementation for shape inference."""
    output_shape = list(grad_output.shape)
    output_shape[shard_dim] -= 2 * halo_size
    return grad_output.new_empty(output_shape)


# NO autograd registration for halo_exchange_bwd
# Higher-order derivatives are explicitly deferred to PR4
