"""
Compiler-compatible DCTensor wrapper (Candidate B).

Preserves ordinary PyTorch model interface - users keep nn.Conv3d unchanged.
Distribution configured at tensor boundary via DCTensorExperimental input.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils._pytree import tree_map
from typing import Optional, Tuple

from .spec import DistConvSpec


# Register coercion handler for plain Tensor → DCTensorExperimental conversion
def _coerce_tensor_to_dctensor(self, metadata, target_type=None):
    """
    Global handler to convert plain Tensor gradients to DCTensorExperimental.

    Registered on torch.Tensor to support AOTAutograd's tangent coercion during
    compiled backward. When backward() receives a plain gradient tensor but
    expects a DCTensorExperimental, this wraps it with the correct metadata.
    """
    # When target_type is provided, use it to construct the wrapper
    # When None (backward compatibility), check the metadata type
    if target_type is not None:
        # Reconstruct using the provided target class
        return target_type(self, metadata)

    # Fallback: shouldn't happen in normal flow
    return self


# Monkey-patch torch.Tensor to add coercion handler
# Required for AOTAutograd to convert plain Tensor gradients back to DCTensorExperimental
if not hasattr(torch.Tensor, '__coerce_same_metadata_as_tangent__'):
    torch.Tensor.__coerce_same_metadata_as_tangent__ = _coerce_tensor_to_dctensor


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

    def __coerce_same_metadata_as_tangent__(self, metadata, target_type=None):
        """
        Coerce gradient tensor to expected metadata during AOTAutograd backward.

        Called by AOTAutograd when it expects a DCTensorExperimental but receives
        a plain Tensor gradient. This method wraps the plain tensor into
        DCTensorExperimental with the correct metadata.

        Args:
            metadata: Expected DistConvSpec
            target_type: Expected subclass type (DCTensorExperimental or None for backward compat)

        Returns:
            DCTensorExperimental wrapping self with metadata
        """
        # Backward compatibility: if target_type not provided, assume DCTensorExperimental
        if target_type is None or target_type is DCTensorExperimental:
            # Wrap the gradient tensor (self) in DCTensorExperimental
            return DCTensorExperimental(self, metadata)
        return self

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

        # Perform distributed convolution
        if halo_size == 0:
            # No halo exchange needed
            output = F.conv3d(
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
            # padding tuple is for spatial dims only: (D, H, W) for Conv3d
            # shard_dim 2→D, 3→H, 4→W
            padding_index = shard_dim - 2
            adjusted_padding[padding_index] = 0

            # 3. Native convolution with adjusted padding
            output = F.conv3d(
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
    def from_shard(cls, tensor: torch.Tensor, spec: DistConvSpec) -> "DCTensorExperimental":
        """
        Create DCTensorExperimental from a local shard with autograd connectivity.

        This is the differentiable entry point that maintains gradient flow.

        Args:
            tensor: Local tensor shard (must require_grad if gradients needed)
            spec: Distribution specification

        Returns:
            DCTensorExperimental with autograd connection to input tensor
        """
        return _FromTensor.apply(tensor, spec)


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
