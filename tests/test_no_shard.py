import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


@pytest.fixture(scope="module")
def parallel_strategy(device: torch.device):
    ps = ParallelStrategy(num_shards=1, device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


def generate_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            for kernel_size in [1, 3, 5]:
                configs.append((ndims, shard_dim, kernel_size))

    return "ndims,shard_dim,kernel_size", configs


@pytest.mark.parametrize(*generate_configs())
def test_no_shard(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    device: torch.device,
):
    """
    Test distributed convolution with num_shards=1 (no actual sharding).
    This should behave as a no-op for the distributed convolution logic.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy with num_shards=1.
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor would be sharded.
        kernel_size (int): Size of the convolution kernel.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, kernel_size=kernel_size, padding=kernel_size // 2).to(
        device
    )

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
    ddpy = dcy.to_ddp()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_ddp()
    dc_conv_grad = conv.weight.grad

    # Validate the results
    # With num_shards=1, every rank has the full tensor (no spatial sharding)
    assert fp32_allclose(ref_y, ddpy)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
