import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


@pytest.fixture(scope="module")
def parallel_strategy(device: torch.device):
    ps = ParallelStrategy(num_shards=4, device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


def generate_channels_last_configs():
    """Generate test configurations for channels_last testing."""
    configs = []
    # Test 2D convolutions with channels_last (ndims=2 -> Conv2d)
    for shard_dim in range(2):  # H or W dimension
        for kernel_size in [1, 3, 5]:
            configs.append((2, shard_dim, kernel_size, torch.channels_last))

    # Test 3D convolutions with channels_last_3d (ndims=3 -> Conv3d)
    for shard_dim in range(3):  # D, H, or W dimension
        for kernel_size in [1, 3, 5]:
            configs.append((3, shard_dim, kernel_size, torch.channels_last_3d))

    return "ndims,shard_dim,kernel_size,memory_format", configs


@pytest.mark.parametrize(*generate_channels_last_configs())
def test_channels_last_forward_backward(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    memory_format: torch.memory_format,
    device: torch.device,
):
    """
    Test distributed convolution with channels_last memory format.
    Verifies correctness and that memory format is preserved.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (2 or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        memory_format (torch.memory_format): Memory format to use.
        device (torch.device): Torch device to run test with.
    """
    parallel_strategy.shard_dim = 2 + shard_dim

    # Create input tensor with channels_last format
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device).to(memory_format=memory_format).requires_grad_(True)

    # Verify input is in channels_last format
    assert x.is_contiguous(memory_format=memory_format)

    # Create convolution layer with channels_last format
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, kernel_size=kernel_size, padding=kernel_size // 2).to(
        device
    )
    conv = conv.to(memory_format=memory_format)

    # Reference forward/backward
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad.clone()
    ref_conv_grad = conv.weight.grad.clone()

    # Distributed forward/backward
    conv.zero_grad()
    x.grad = None
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)

    # Verify DCTensor preserves channels_last format
    assert dcx._tensor.is_contiguous(memory_format=memory_format), (
        "DCTensor did not preserve channels_last format"
    )

    dcy = ddp_conv(dcx)

    # Verify output preserves channels_last format
    assert dcy._tensor.is_contiguous(memory_format=memory_format), (
        "Output did not preserve channels_last format"
    )

    ddpy = dcy.to_ddp()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_ddp()
    dc_conv_grad = conv.weight.grad

    # Validate numerical correctness
    if dist.get_rank() == 0:
        assert fp32_allclose(ref_y, ddpy)
    else:
        assert ddpy.numel() == 0
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)


def generate_periodic_channels_last_configs():
    """Generate test configurations for periodic padding with channels_last."""
    configs = []
    # Test 2D convolutions with channels_last
    for shard_dim in range(2):
        for kernel_size in [3, 5]:
            configs.append((2, shard_dim, kernel_size, torch.channels_last))

    # Test 3D convolutions with channels_last_3d
    for shard_dim in range(3):
        for kernel_size in [3, 5]:
            configs.append((3, shard_dim, kernel_size, torch.channels_last_3d))

    return "ndims,shard_dim,kernel_size,memory_format", configs


@pytest.mark.parametrize(*generate_periodic_channels_last_configs())
def test_channels_last_periodic_padding(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    memory_format: torch.memory_format,
    device: torch.device,
):
    """
    Test periodic padding with channels_last format.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (2 or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        memory_format (torch.memory_format): Memory format to use.
        device (torch.device): Torch device to run test with.
    """
    parallel_strategy.shard_dim = 2 + shard_dim

    # Create input tensor with channels_last format
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device).to(memory_format=memory_format).requires_grad_(True)

    # Create convolution layer with circular padding
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(
        4, 8, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="circular"
    ).to(device)
    conv = conv.to(memory_format=memory_format)

    # Reference forward/backward
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad.clone()
    ref_conv_grad = conv.weight.grad.clone()

    # Distributed forward/backward
    conv.zero_grad()
    x.grad = None
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)

    dcy = ddp_conv(dcx)

    # Verify output preserves channels_last format
    assert dcy._tensor.is_contiguous(memory_format=memory_format), (
        "Output did not preserve channels_last format with periodic padding"
    )

    ddpy = dcy.to_ddp()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_ddp()
    dc_conv_grad = conv.weight.grad

    # Validate numerical correctness
    if dist.get_rank() == 0:
        assert fp32_allclose(ref_y, ddpy)
    else:
        assert ddpy.numel() == 0
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
