import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.functional import pad
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
                padding = find_padding(kernel_size)
                for stride in [1, 2, 4]:
                    configs.append((ndims, shard_dim, kernel_size, padding, stride))

    return "ndims,shard_dim,kernel_size,padding,stride", configs


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
        kernel_size (int): Size of the convolution kernel.
        padding (int): Amount of padding to apply to the input tensor.
        stride (int): Stride of the convolution.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim
    parallel_strategy.space_ndim = ndims
    parallel_strategy.set_shard_inds()

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True, dtype=torch.double)
    conv_class = getattr(nn, f"ConvTranspose{ndims}d")
    conv = (
        conv_class(
            4, 8, kernel_size=kernel_size, padding=padding, stride=stride, bias=True
        )
        .to(device)
        .double()
    )
    torch.nn.init.uniform_(conv.weight)
    torch.nn.init.uniform_(conv.bias)

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
    x_grad = dcx.grad.to_replicate(shape=ref_x_grad.shape, stride=ref_x_grad.stride())
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
        kernel_size (int): Size of the convolution kernel.
        padding (int): Amount of padding to apply to the input tensor.
        stride (int): Stride of the convolution.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim
    parallel_strategy.space_ndim = ndims
    parallel_strategy.set_shard_inds()
    parallel_strategy.is_periodic = True
    parallel_strategy.padding_mode = "circular"

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True, dtype=torch.double)

    conv_kwargs = dict(kernel_size=kernel_size, stride=stride, bias=True)

    # set periodic padding case for reference
    new_padding = [padding, padding] * ndims
    x_periodic = pad(input=x, pad=new_padding, mode="circular")
    ref_padding = kernel_size - 1

    conv_class = getattr(nn, f"ConvTranspose{ndims}d")
    conv = (
        conv_class(4, 8, padding=padding, **conv_kwargs)
        .to(device)
        .requires_grad_(False)
        .double()
    )
    ref_conv = (
        conv_class(4, 8, padding=ref_padding, **conv_kwargs)
        .to(device)
        .requires_grad_(False)
        .double()
    )
    torch.nn.init.uniform_(conv.weight)
    torch.nn.init.uniform_(conv.bias)
    conv.weight.copy_(ref_conv.weight)
    conv.bias.copy_(ref_conv.bias)
    conv.requires_grad_(True)
    ref_conv.requires_grad_(True)

    # Perform forward and backward pass for reference (non-distributed) convolution
    ref_conv.zero_grad()
    ref_y = ref_conv(x_periodic)
    crop_amount = (kernel_size - 1 - padding) * (stride - 1)
    ref_y = pad(input=ref_y, pad=(-crop_amount, -crop_amount) * ndims, mode="circular")
    ref_y.sum().backward()
    ref_x_grad = x.grad
    ref_conv_grad = ref_conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = dist_conv(dcx)
    dcy_merge = all_gather_vlen(dcy, dim=(parallel_strategy.shard_dim))
    dc_loss = dcy.sum()
    dist.all_reduce(dc_loss)
    dc_loss.backward()
    x_grad = dcx.grad.to_replicate(shape=ref_x_grad.shape, stride=ref_x_grad.stride())
    dc_conv_grad = conv.weight.grad

    # Validate the results
    assert fp32_allclose(ref_y, dcy_merge)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
