import os
import pytest
import torch
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


@pytest.fixture(scope="module")
def parallel_strategy_2d(device: torch.device):
    ps = ParallelStrategy(num_shards=(2, 2), shard_dim=(2, 3), device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


@pytest.fixture(scope="module")
def parallel_strategy_3d(device: torch.device):
    ps = ParallelStrategy(
        num_shards=(2, 2, 2), shard_dim=(2, 3, 4), device_type=device.type
    )
    yield ps
    cleanup_parallel_strategy(ps)


def generate_2d_configs():
    configs = []
    for ndims in [2, 3]:
        for shard_dim_i in range(ndims):
            for shard_dim_j in range(ndims):
                if shard_dim_i == shard_dim_j:
                    continue
                for kernel_size in [1, 3, 5]:
                    for stride in [1, 2, 4]:
                        for padding_mode in ["zeros", "circular"]:
                            configs.append(
                                (
                                    ndims,
                                    shard_dim_i,
                                    shard_dim_j,
                                    kernel_size,
                                    stride,
                                    padding_mode,
                                )
                            )

    return "ndims,shard_dim_i,shard_dim_j,kernel_size,stride,padding_mode", configs


def generate_3d_configs():
    configs = []
    for kernel_size in [1, 3, 5]:
        for stride in [1, 2, 4]:
            configs.append((kernel_size, stride))

    return "kernel_size,stride", configs


@pytest.mark.parametrize(*generate_2d_configs())
def test_2d_splitting(
    parallel_strategy_2d: ParallelStrategy,
    ndims: int,
    shard_dim_i: int,
    shard_dim_j: int,
    kernel_size: int,
    stride: int,
    padding_mode: str,
    device: torch.device,
):
    """
    Test distributed convolution with different number of dimensions and shard dimensions.
    Checks the output and gradients of the distributed convolution against the non-distributed
    convolution.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (2 or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy = parallel_strategy_2d
    parallel_strategy.shard_dim = [shard_dim_i + 2, shard_dim_j + 2]

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(
        4,
        8,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=stride,
        padding_mode=padding_mode,
    ).to(device)

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate the results
    assert fp32_allclose(ref_y, ddpy)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)


@pytest.mark.parametrize(*generate_3d_configs())
@pytest.mark.skipif(int(os.environ["WORLD_SIZE"]) < 8, reason="requires 8 ranks")
def test_3d_splitting(
    parallel_strategy_3d: ParallelStrategy,
    kernel_size: int,
    stride: int,
    device: torch.device,
):
    """
    Test distributed convolution with different number of dimensions and shard dimensions.
    Checks the output and gradients of the distributed convolution against the non-distributed
    convolution.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (2 or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    ndims = 3
    parallel_strategy = parallel_strategy_3d

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(
        4, 8, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride
    ).to(device)

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate the results
    assert fp32_allclose(ref_y, ddpy)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)


@pytest.mark.parametrize(
    "kernel_sizes,padding",
    [
        ((3, 5), (1, 2)),  # Different halo sizes: 1 vs 2
        ((5, 3), (2, 1)),  # Reversed order
        ((1, 3), (0, 1)),  # One with no halo, one with halo
        ((3, 1), (1, 0)),  # Reversed
    ],
)
def test_2d_splitting_nonuniform_kernel(
    parallel_strategy_2d: ParallelStrategy,
    kernel_sizes: tuple,
    padding: tuple,
    device: torch.device,
):
    """
    Test 2D splitting with non-uniform kernel sizes.

    Verifies that halo exchange correctly handles different halo sizes per dimension
    when kernel sizes differ (e.g., 3x5 kernel requires halo_size=1 for dim 0 and
    halo_size=2 for dim 1).
    """
    parallel_strategy = parallel_strategy_2d
    parallel_strategy.shard_dim = (2, 3)

    # Initialize input tensor - large enough to be divisible by stride and shards
    shape = [1, 4, 64, 64]
    x = torch.randn(*shape, device=device, requires_grad=True)

    # Create conv with non-uniform kernel sizes
    conv = nn.Conv2d(
        4,
        8,
        kernel_size=kernel_sizes,
        padding=padding,
        stride=1,
        padding_mode="zeros",
    ).to(device)

    # Reference (non-distributed) forward and backward
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad.clone()
    ref_conv_grad = conv.weight.grad.clone()

    # Distributed forward and backward
    x.grad = None
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate results
    assert fp32_allclose(ref_y, ddpy), "Forward pass mismatch with non-uniform kernel"
    assert fp32_allclose(ref_x_grad, x_grad), (
        "Input gradient mismatch with non-uniform kernel"
    )
    assert fp32_allclose(ref_conv_grad, dc_conv_grad), (
        "Weight gradient mismatch with non-uniform kernel"
    )


@pytest.mark.parametrize(
    "kernel_sizes",
    [
        (3, 5),
        (5, 3),
    ],
)
def test_2d_splitting_nonuniform_kernel_circular(
    parallel_strategy_2d: ParallelStrategy,
    kernel_sizes: tuple,
    device: torch.device,
):
    """
    Test 2D splitting with non-uniform kernels and circular padding.

    Verifies that non-uniform kernel sizes work correctly with periodic boundary
    conditions, where each dimension has its own halo size and periodicity flag.
    """
    parallel_strategy = parallel_strategy_2d
    parallel_strategy.shard_dim = (2, 3)

    # Padding for "same" convolution with odd kernels
    padding = (kernel_sizes[0] // 2, kernel_sizes[1] // 2)

    # Initialize input tensor
    shape = [1, 4, 64, 64]
    x = torch.randn(*shape, device=device, requires_grad=True)

    # Create conv with non-uniform kernel sizes and circular padding
    conv = nn.Conv2d(
        4,
        8,
        kernel_size=kernel_sizes,
        padding=padding,
        stride=1,
        padding_mode="circular",
    ).to(device)

    # Reference (non-distributed) forward and backward
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad.clone()
    ref_conv_grad = conv.weight.grad.clone()

    # Distributed forward and backward
    x.grad = None
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate results
    assert fp32_allclose(ref_y, ddpy), (
        "Forward pass mismatch with non-uniform kernel + circular padding"
    )
    assert fp32_allclose(ref_x_grad, x_grad), (
        "Input gradient mismatch with non-uniform kernel + circular padding"
    )
    assert fp32_allclose(ref_conv_grad, dc_conv_grad), (
        "Weight gradient mismatch with non-uniform kernel + circular padding"
    )


@pytest.mark.skipif(int(os.environ["WORLD_SIZE"]) < 8, reason="requires 8 ranks")
@pytest.mark.parametrize(
    "kernel_sizes,padding",
    [
        ((3, 5, 7), (1, 2, 3)),  # All different halo sizes
        ((1, 3, 5), (0, 1, 2)),  # Progressive sizes
        ((7, 3, 1), (3, 1, 0)),  # Decreasing sizes
    ],
)
def test_3d_splitting_nonuniform_kernel(
    parallel_strategy_3d: ParallelStrategy,
    kernel_sizes: tuple,
    padding: tuple,
    device: torch.device,
):
    """
    Test 3D splitting with non-uniform kernel sizes.

    Verifies that halo exchange correctly handles three different halo sizes
    when splitting across all three spatial dimensions with varying kernel sizes.
    """
    parallel_strategy = parallel_strategy_3d

    # Initialize input tensor - 3D spatial dimensions
    shape = [1, 4, 32, 32, 32]
    x = torch.randn(*shape, device=device, requires_grad=True)

    # Create Conv3d with non-uniform kernel sizes
    conv = nn.Conv3d(
        4,
        8,
        kernel_size=kernel_sizes,
        padding=padding,
        stride=1,
        padding_mode="zeros",
    ).to(device)

    # Reference (non-distributed) forward and backward
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad.clone()
    ref_conv_grad = conv.weight.grad.clone()

    # Distributed forward and backward
    x.grad = None
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate results
    assert fp32_allclose(ref_y, ddpy), (
        "Forward pass mismatch with 3D non-uniform kernel"
    )
    assert fp32_allclose(ref_x_grad, x_grad), (
        "Input gradient mismatch with 3D non-uniform kernel"
    )
    assert fp32_allclose(ref_conv_grad, dc_conv_grad), (
        "Weight gradient mismatch with 3D non-uniform kernel"
    )


@pytest.mark.parametrize("stride", [1, 2])
def test_2d_splitting_nonuniform_kernel_with_stride(
    parallel_strategy_2d: ParallelStrategy,
    stride: int,
    device: torch.device,
):
    """
    Test 2D splitting with non-uniform kernels and strided convolution.

    This tests the interaction between non-uniform kernels and stride, which
    affects how the output tensor size is computed and how halo data is used.
    """
    parallel_strategy = parallel_strategy_2d
    parallel_strategy.shard_dim = (2, 3)

    # Non-uniform kernel with different halo sizes
    kernel_sizes = (3, 5)
    padding = (1, 2)

    # Initialize input tensor - must be divisible by stride and shards
    shape = [1, 4, 64, 64]
    x = torch.randn(*shape, device=device, requires_grad=True)

    # Create conv with non-uniform kernel sizes
    conv = nn.Conv2d(
        4,
        8,
        kernel_size=kernel_sizes,
        padding=padding,
        stride=stride,
        padding_mode="zeros",
    ).to(device)

    # Reference (non-distributed) forward and backward
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad.clone()
    ref_conv_grad = conv.weight.grad.clone()

    # Distributed forward and backward
    x.grad = None
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate results
    assert fp32_allclose(ref_y, ddpy), (
        "Forward pass mismatch with non-uniform kernel + stride"
    )
    assert fp32_allclose(ref_x_grad, x_grad), (
        "Input gradient mismatch with non-uniform kernel + stride"
    )
    assert fp32_allclose(ref_conv_grad, dc_conv_grad), (
        "Weight gradient mismatch with non-uniform kernel + stride"
    )
