import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


def all_gather_vlen(tensor: torch.Tensor, group=None, dim=0) -> list[torch.Tensor]:
    """Gather tensors with the same number of dimensions but different lengths.

    Credit: https://stackoverflow.com/a/78934638
    """
    world_size = dist.get_world_size(group=group)
    # Gather lengths first
    shape = torch.as_tensor(tensor.shape, device=tensor.device)
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=group)
    # Gather data
    inputs = [tensor] * world_size
    outputs = [
        torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device)
        for _shape in shapes
    ]
    dist.all_to_all(outputs, inputs, group=group)
    return torch.cat(outputs, dim=dim)


@pytest.fixture(scope="module")
def parallel_strategy(device: torch.device):
    ps = ParallelStrategy(num_shards=4, device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


def find_padding(kernel_size):
    if kernel_size % 2 != 0:
        return kernel_size // 2
    else:
        return 0


def generate_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            for kernel_size in [1, 3, 5]:
                for stride in [1, 2, 4]:
                    configs.append((ndims, shard_dim, kernel_size, stride))

    return "ndims,shard_dim,kernel_size,stride", configs


@pytest.mark.parametrize(*generate_configs())
def test_transposeconv_zerospadding(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    padding: int,
    stride: int,
    device: torch.device,
):
    """
    Test distributed convolution with different number of dimensions, kernel sizes, and strides.
    Checks the output and gradients of the distributed convolution against the non-distributed
    convolution.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim
    padding = find_padding(kernel_size)

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"ConvTranspose{ndims}d")
    conv = conv_class(4, 8, kernel_size=kernel_size, padding=padding, stride=stride).to(
        device
    )

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.sum().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = dist_conv(dcx)
    dcy_merge = all_gather_vlen(dcy, dim=(parallel_strategy.shard_dim))
    dc_loss = dcy.sum()
    dist.all_reduce(dc_loss)
    dc_loss.backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    assert fp32_allclose(ref_y, dcy_merge)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)


@pytest.mark.parametrize(*generate_configs())
def test_transposeconv_circularpadding(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    stride: int,
    device: torch.device,
):
    """
    Test distributed convolution with different number of dimensions, kernel sizes, and strides.
    Checks the output and gradients of the distributed convolution against the non-distributed
    convolution.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim
    padding = find_padding(kernel_size)

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)

    conv_kwargs = dict(kernel_size=kernel_size, stride=stride)

    # set periodic padding case for reference
    new_padding = [padding, padding] * ndims
    x_periodic = torch.nn.functional.pad(input=x, pad=new_padding, mode="circular")
    ref_padding = kernel_size - 1

    conv_class = getattr(nn, f"ConvTranspose{ndims}d")
    conv = (
        conv_class(4, 8, padding=ref_padding, **conv_kwargs)
        .to(device)
        .requires_grad_(False)
    )
    conv.requires_grad_(True)

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x_periodic)
    for i in range(0, ndims):
        crop_amount = (kernel_size - 1 - padding) * (stride - 1)
        ref_y = ref_y.narrow(i + 2, crop_amount, ref_y.shape[i + 2] - 2 * crop_amount)
    ref_y.sum().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcx_periodic = torch.nn.functional.pad(input=dcx, pad=new_padding, mode="circular")
    dcy = dist_conv(dcx_periodic)
    for i in range(0, ndims):
        if i != shard_dim:
            crop_amount = (kernel_size - 1 - padding) * (stride - 1)
            dcy = dcy.narrow(i + 2, crop_amount, dcy.shape[i + 2] - 2 * crop_amount)
    dcy_merge = all_gather_vlen(dcy.contiguous(), dim=(parallel_strategy.shard_dim))
    dc_loss = dcy.sum()
    dist.all_reduce(dc_loss)
    dc_loss.backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate the results
    assert fp32_allclose(ref_y, dcy_merge)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
