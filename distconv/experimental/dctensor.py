"""
Compiler-compatible DCTensor wrapper (Candidate B).

Preserves ordinary PyTorch model interface - users keep nn.Conv3d unchanged.
Distribution configured at tensor boundary via DCTensorExperimental input.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils._pytree import tree_map
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from typing import Optional, Tuple

from .spec import DistConvSpec
from .resources import register_parallel_strategy, get_parallel_strategy


class _DCTensorSum(Function):
    """
    Custom autograd function for DCTensor sum reduction.

    Gradient flow:
    1. This Function receives scalar grad_output from loss.backward()
    2. Function expands it and wraps in DCTensorExperimental
    3. Compiled model backward receives the DCTensor-compatible result

    This avoids global torch.Tensor patches while maintaining correct
    metadata through the autograd graph.
    """

    @staticmethod
    def forward(ctx, dc_tensor, spec, args, kwargs):
        """
        Forward: unwrap DCTensor, compute sum, save metadata for backward.

        Args:
            dc_tensor: DCTensorExperimental input
            spec: DistConvSpec metadata to reconstruct gradient
            args: Positional arguments for sum()
            kwargs: Keyword arguments for sum()
        """
        # Save metadata for backward
        ctx.spec = spec
        ctx.shape = dc_tensor.shape
        ctx.dtype = dc_tensor.dtype
        ctx.device = dc_tensor.device

        # Unwrap and compute sum on inner tensor
        return dc_tensor._tensor.sum(*args, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: expand scalar gradient to DCTensorExperimental.

        This Function receives the scalar gradient from loss.backward().
        It expands and wraps it so compiled model backward receives
        DCTensor-compatible input.

        Args:
            grad_output: Scalar gradient (typically 1.0 for .sum())

        Returns:
            (grad_dc_tensor, None, None, None) - gradient as DCTensorExperimental
        """
        # Expand scalar gradient to full tensor shape
        grad_tensor = grad_output.expand(ctx.shape).contiguous()

        # Wrap in DCTensorExperimental with saved metadata
        grad_dc_tensor = DCTensorExperimental(grad_tensor, ctx.spec)

        # Return gradient for dc_tensor (first arg), None for spec/args/kwargs
        return grad_dc_tensor, None, None, None


class DCTensorExperimental(torch.Tensor):
    """
    Compile-compatible distributed convolution tensor wrapper.

    Implements flatten/unflatten protocol for torch.compile() compatibility.
    Uses frozen DistConvSpec metadata (no process groups stored).
    """

    @staticmethod
    def __new__(cls, local_tensor: torch.Tensor, spec: DistConvSpec):
        """Create wrapper subclass around local tensor shard."""
        return torch.Tensor._make_wrapper_subclass(
            cls,
            local_tensor.shape,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            requires_grad=local_tensor.requires_grad,
        )

    def __init__(self, local_tensor: torch.Tensor, spec: DistConvSpec):
        """
        Initialize DCTensorExperimental.

        Args:
            local_tensor: Local shard data
            spec: Frozen distributed convolution specification
        """
        self._tensor = local_tensor
        self._spec = spec

    def sum(self, *args, **kwargs):
        """
        Local shard sum reduction with correct gradient flow.

        Computes sum over the local shard only. Does not perform cross-rank
        aggregation - use torch.distributed operations for global reductions.

        Supports only no-argument full reduction: dc_tensor.sum()
        General dim/keepdim reductions are not supported in this prototype.

        Returns:
            Scalar tensor (ordinary torch.Tensor, not DCTensorExperimental)

        Raises:
            NotImplementedError: If dim or keepdim arguments provided
        """
        # Bounded prototype: only support full local sum
        if args or kwargs:
            raise NotImplementedError(
                "DCTensorExperimental.sum() only supports full reduction with no arguments. "
                "Partial reductions (dim/keepdim) are not implemented in this prototype."
            )
        return _DCTensorSum.apply(self, self._spec, args, kwargs)

    def __tensor_flatten__(self):
        """
        Flatten for compilation.

        Returns:
            (inner_tensor_names, metadata) tuple
        """
        return ["_tensor"], self._spec

    @staticmethod
    def __tensor_unflatten__(inner_tensors, spec, outer_size, outer_stride):
        """
        Unflatten from compilation.

        Args:
            inner_tensors: Dict mapping names to tensors
            spec: DistConvSpec metadata
            outer_size: Outer tensor size (unused)
            outer_stride: Outer tensor stride (unused)

        Returns:
            Reconstructed DCTensorExperimental
        """
        return DCTensorExperimental(inner_tensors["_tensor"], spec)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """
        Dispatch operations to either distributed convolution or default handling.

        For convolution: calls halo exchange + adjusted padding convolution.
        For convolution_backward: handles distributed gradient computation.
        For other ops: unwrap → execute → rewrap.
        """
        kwargs = kwargs or {}

        # Handle convolution forward
        if func is torch.ops.aten.convolution.default:
            return cls._dispatch_convolution(args, kwargs)

        # Handle convolution backward
        elif func is torch.ops.aten.convolution_backward.default:
            return cls._dispatch_convolution_backward(args, kwargs)

        # For all other operations: unwrap → execute → maybe rewrap
        def unwrap(x):
            return x._tensor if isinstance(x, DCTensorExperimental) else x

        unwrapped_args = tree_map(unwrap, args)
        unwrapped_kwargs = tree_map(unwrap, kwargs)

        result = func(*unwrapped_args, **unwrapped_kwargs)

        # Only rewrap non-scalar tensors
        # Scalars (from reductions like sum, mean) should be returned unwrapped
        # so they maintain autograd connectivity for backward()
        if args and isinstance(args[0], DCTensorExperimental):
            if isinstance(result, torch.Tensor):
                # Scalar results: return unwrapped (for backward to work)
                if result.ndim == 0:
                    return result
                # Non-scalar tensors: rewrap to maintain DCTensor semantics
                return cls._rewrap(result, args[0]._spec)

        return result

    @classmethod
    def _dispatch_convolution(cls, args, kwargs):
        """
        Dispatch convolution through distributed halo exchange.

        Convolution schema: (input, weight, bias, stride, padding, dilation,
                            transposed, output_padding, groups)
        """
        input_tensor = args[0]
        weight = args[1]
        bias = args[2]
        stride = args[3]
        padding = args[4]
        dilation = args[5]
        transposed = args[6]
        output_padding = args[7]
        groups = args[8]

        if not isinstance(input_tensor, DCTensorExperimental):
            # Not a DCTensor, use default behavior
            return torch.ops.aten.convolution.default(*args, **kwargs)

        # Extract spec and compute halo parameters
        spec = input_tensor._spec
        local_input = input_tensor._tensor

        # Get shard dimension (normalize negative indices)
        shard_dim = spec.shard_dim[0]
        if shard_dim < 0:
            shard_dim = local_input.ndim + shard_dim

        # Compute halo size from kernel size
        # For Conv3d, weight shape is [out_channels, in_channels/groups, kD, kH, kW]
        # shard_dim for input is spatial dim (2, 3, or 4 for Conv3d)
        # We need to map it to weight's spatial dims (2, 3, 4)
        weight_spatial_dim = shard_dim  # Weight has same spatial indexing as input
        kernel_size = weight.size(weight_spatial_dim)

        # Only support odd kernels with halo, or even kernels with zero padding
        if kernel_size % 2 == 1:
            halo_size = kernel_size // 2
        else:
            # Even kernel - must have zero padding on sharded dim
            halo_size = 0

        # Get neighbor information
        left_rank, right_rank = spec.neighbor_ranks(0)  # dim_index=0 for first sharded dim
        is_leftmost = (spec.shard_ind[0] == 0)
        is_rightmost = (spec.shard_ind[0] == spec.num_shards[0] - 1)

        # Determine convolution function based on input dimensions
        # Input shape: [batch, channels, *spatial_dims]
        # Conv1d: 3D input [N, C, L], Conv2d: 4D input [N, C, H, W], Conv3d: 5D input [N, C, D, H, W]
        ndim = local_input.ndim
        if ndim == 3:
            conv_fn = F.conv1d
        elif ndim == 4:
            conv_fn = F.conv2d
        elif ndim == 5:
            conv_fn = F.conv3d
        else:
            raise ValueError(f"Unsupported input dimensions: {ndim} (expected 3, 4, or 5)")

        # Perform distributed convolution
        if halo_size == 0:
            # No halo exchange needed
            output = conv_fn(
                local_input, weight, bias,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )
        else:
            # 1. Halo exchange: N → N + 2*halo_size
            input_with_halo = torch.ops.distconv_experimental.halo_exchange_fwd(
                local_input,
                shard_dim,
                halo_size,
                left_rank,
                right_rank,
                is_leftmost,
                is_rightmost,
            )

            # 2. Adjust padding for sharded dimension (set to 0)
            adjusted_padding = list(padding)
            # padding tuple is for spatial dims only
            # shard_dim 2→first spatial, 3→second spatial, etc.
            padding_index = shard_dim - 2
            adjusted_padding[padding_index] = 0

            # 3. Native convolution with adjusted padding
            output = conv_fn(
                input_with_halo, weight, bias,
                stride=tuple(stride), padding=tuple(adjusted_padding),
                dilation=tuple(dilation), groups=groups
            )

        # Wrap output in DCTensorExperimental
        return DCTensorExperimental(output, spec)

    @classmethod
    def _dispatch_convolution_backward(cls, args, kwargs):
        """
        Dispatch convolution backward through distributed halo exchange.

        Convolution backward schema: (grad_output, input, weight, bias_sizes,
                                     stride, padding, dilation, transposed,
                                     output_padding, groups, output_mask)

        Returns: (grad_input, grad_weight, grad_bias)
        """
        grad_output = args[0]
        input_tensor = args[1]
        weight = args[2]
        # bias_sizes = args[3]  # Not used directly
        stride = args[4]
        padding = args[5]
        dilation = args[6]
        transposed = args[7]
        output_padding = args[8]
        groups = args[9]
        output_mask = args[10]

        # If input is not DCTensor, use default behavior
        if not isinstance(input_tensor, DCTensorExperimental):
            return torch.ops.aten.convolution_backward.default(*args, **kwargs)

        # Extract spec from input
        spec = input_tensor._spec
        local_input = input_tensor._tensor
        local_grad_output = grad_output._tensor if isinstance(grad_output, DCTensorExperimental) else grad_output

        # Get shard dimension and halo parameters (same as forward)
        shard_dim = spec.shard_dim[0]
        if shard_dim < 0:
            shard_dim = local_input.ndim + shard_dim

        weight_spatial_dim = shard_dim
        kernel_size = weight.size(weight_spatial_dim)

        if kernel_size % 2 == 1:
            halo_size = kernel_size // 2
        else:
            halo_size = 0

        # Get neighbor information
        left_rank, right_rank = spec.neighbor_ranks(0)
        is_leftmost = (spec.shard_ind[0] == 0)
        is_rightmost = (spec.shard_ind[0] == spec.num_shards[0] - 1)

        if halo_size == 0:
            # No halo - use native backward directly
            grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward.default(
                local_grad_output, local_input, weight, *args[3:], **kwargs
            )
            # grad_input may be None if output_mask[0] is False
        else:
            # Need to:
            # 1. Recreate forward's halo-extended input
            # 2. Compute backward with adjusted padding
            # 3. Apply halo backward to grad_input

            # Recreate halo-extended input from forward pass
            input_with_halo = torch.ops.distconv_experimental.halo_exchange_fwd(
                local_input,
                shard_dim,
                halo_size,
                left_rank,
                right_rank,
                is_leftmost,
                is_rightmost,
            )

            # Adjust padding for backward (same as forward)
            adjusted_padding = list(padding)
            padding_index = shard_dim - 2
            adjusted_padding[padding_index] = 0

            # Native convolution backward with halo-extended input
            grad_input_halo, grad_weight, grad_bias = torch.ops.aten.convolution_backward.default(
                local_grad_output,
                input_with_halo,
                weight,
                args[3],  # bias_sizes
                tuple(stride),
                tuple(adjusted_padding),
                tuple(dilation),
                transposed,
                tuple(output_padding),
                groups,
                output_mask
            )

            # Apply halo backward only if grad_input was requested
            # (grad_input_halo may be None if output_mask[0] is False)
            if output_mask[0] and grad_input_halo is not None:
                # grad_input_halo has shape N + 2*halo_size
                # Apply halo backward to get local gradient with boundary contributions
                grad_input = torch.ops.distconv_experimental.halo_exchange_bwd(
                    grad_input_halo,
                    shard_dim,
                    halo_size,
                    left_rank,
                    right_rank,
                    is_leftmost,
                    is_rightmost,
                )
            else:
                grad_input = None

        # Wrap grad_input in DCTensorExperimental, keep grad_weight and grad_bias as plain tensors
        if output_mask[0] and grad_input is not None:
            grad_input = DCTensorExperimental(grad_input, spec)
        else:
            grad_input = None

        return grad_input, grad_weight, grad_bias

    @classmethod
    def _rewrap(cls, result, spec: DistConvSpec):
        """Rewrap tensors in result structure."""
        if isinstance(result, torch.Tensor):
            return DCTensorExperimental(result, spec)
        elif isinstance(result, (list, tuple)):
            rewrapped = [cls._rewrap(x, spec) for x in result]
            return type(result)(rewrapped)
        else:
            return result

    def __repr__(self):
        return f"DCTensorExperimental({self._tensor.shape}, spec={self._spec})"

    @classmethod
    def distribute(cls, tensor: torch.Tensor, parallel_strategy: "ParallelStrategy") -> "DCTensorExperimental":
        """
        Shard a tensor according to the given parallel strategy.

        Public API compatible with original DCTensor.distribute().

        Args:
            tensor: The tensor to be sharded
            parallel_strategy: The parallel strategy for sharding

        Returns:
            DCTensorExperimental with tensor sharded according to strategy
        """
        # Extract primitive metadata for DistConvSpec
        spec = DistConvSpec(
            num_shards=parallel_strategy.num_shards,
            shard_dim=parallel_strategy.shard_dim,
            rank=parallel_strategy.rank,
            world_size=parallel_strategy.world_size,
            shard_ind=tuple(parallel_strategy.shard_ind),
            is_periodic=(False,) * len(parallel_strategy.shard_dim),  # TODO: add periodic support
        )

        # Register resources for later access (outside tracing)
        register_parallel_strategy(spec, parallel_strategy)

        # Use device_mesh from parallel_strategy for DTensor distribution
        placements = [Shard(i) for i in parallel_strategy.shard_dim]
        device_mesh = parallel_strategy.device_mesh[parallel_strategy.distconv_dim_names]

        dtensor = distribute_tensor(
            tensor,
            device_mesh=device_mesh,
            placements=placements,
        )
        local_tensor = dtensor.to_local()

        # TODO: Preserve memory format (add infer_contiguous_format utility)
        # For now, assume contiguous

        return cls(local_tensor, spec)

    @classmethod
    def from_shard(cls, tensor: torch.Tensor, parallel_strategy: "ParallelStrategy") -> "DCTensorExperimental":
        """
        Create DCTensorExperimental from a local shard with autograd connectivity.

        Public API compatible with original DCTensor.from_shard().

        Args:
            tensor: Local tensor shard (must require_grad if gradients needed)
            parallel_strategy: The parallel strategy for distributing the tensor

        Returns:
            DCTensorExperimental with autograd connection to input tensor
        """
        # Extract primitive metadata for DistConvSpec
        spec = DistConvSpec(
            num_shards=parallel_strategy.num_shards,
            shard_dim=parallel_strategy.shard_dim,
            rank=parallel_strategy.rank,
            world_size=parallel_strategy.world_size,
            shard_ind=tuple(parallel_strategy.shard_ind),
            is_periodic=(False,) * len(parallel_strategy.shard_dim),  # TODO: add periodic support
        )

        # Register resources for later access
        register_parallel_strategy(spec, parallel_strategy)

        return _FromTensor.apply(tensor, spec)

    def to_ddp(self) -> torch.Tensor:
        """
        Convert the DCTensor to a distributed data parallel tensor.

        Public API compatible with original DCTensor.to_ddp().
        Reshards tensor to batch dimension.

        Returns:
            torch.Tensor: The tensor resharded to the batch dimension
        """
        # Look up registered ParallelStrategy for resources
        parallel_strategy = get_parallel_strategy(self._spec)

        # Get device_mesh for redistribution
        device_mesh = parallel_strategy.device_mesh[parallel_strategy.distconv_dim_names]
        placements = [Shard(i) for i in parallel_strategy.shard_dim]

        # Convert to DTensor with current sharding, then reshard to batch dimension
        dtensor = DTensor.from_local(
            _ToTensor.apply(self),
            device_mesh=device_mesh,
            placements=placements,
        ).redistribute(
            device_mesh=device_mesh,
            placements=[Shard(0)] * device_mesh.ndim
        )

        local_tensor = dtensor.to_local()

        # TODO: Preserve memory format
        return local_tensor

    def to_replicate(self) -> torch.Tensor:
        """
        Convert the DCTensor to a replicated tensor.

        Public API compatible with original DCTensor.to_replicate().
        Gathers full tensor on all ranks.

        Returns:
            torch.Tensor: The full replicated tensor
        """
        # Look up registered ParallelStrategy for resources
        parallel_strategy = get_parallel_strategy(self._spec)

        # Get device_mesh for redistribution
        device_mesh = parallel_strategy.device_mesh[parallel_strategy.distconv_dim_names]
        placements = [Shard(i) for i in parallel_strategy.shard_dim]

        # Convert to DTensor with current sharding, then replicate
        dtensor = DTensor.from_local(
            _ToTensor.apply(self),
            device_mesh=device_mesh,
            placements=placements,
        ).redistribute(
            device_mesh=device_mesh,
            placements=[Replicate()] * device_mesh.ndim
        )

        local_tensor = dtensor.to_local()

        # TODO: Preserve memory format
        return local_tensor


class _FromTensor(Function):
    """
    Autograd Function to convert torch.Tensor -> DCTensorExperimental.

    This establishes the autograd connection between the original tensor
    and the wrapped DCTensor, allowing gradients to flow back properly.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, spec: DistConvSpec):
        """Forward: wrap tensor in DCTensorExperimental."""
        return DCTensorExperimental(tensor, spec)

    @staticmethod
    def backward(ctx, grad):
        """Backward: unwrap gradient back to plain tensor."""
        # grad might be DCTensorExperimental or plain tensor (from autograd engine)
        if isinstance(grad, DCTensorExperimental):
            return _ToTensor.apply(grad), None
        else:
            # Already a plain tensor, return as-is
            return grad, None


class _ToTensor(Function):
    """
    Autograd Function to convert DCTensorExperimental -> torch.Tensor.

    Used to unwrap DCTensor back to plain tensor while maintaining autograd.
    """

    @staticmethod
    def forward(ctx, dc_tensor: DCTensorExperimental):
        """Forward: extract inner tensor."""
        ctx.spec = dc_tensor._spec
        # Alias the tensor to prevent circular references (see PR #20)
        return torch.Tensor(dc_tensor._tensor)

    @staticmethod
    def backward(ctx, grad):
        """Backward: wrap gradient back in DCTensorExperimental."""
        # grad might already be wrapped (from another DCTensor operation)
        if isinstance(grad, DCTensorExperimental):
            return grad
        else:
            return _FromTensor.apply(grad, ctx.spec)
