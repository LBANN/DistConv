import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


def generate_2d_configs():
    configs = []
    for ndims in [2, 3]:
        for shard_dim_i in range(ndims):
            for shard_dim_j in range(ndims):
                if shard_dim_i == shard_dim_j:
                    continue
                for kernel_size in [1, 3, 5]:
                    configs.append((ndims, shard_dim_i, shard_dim_j, kernel_size))

    return "ndims,shard_dim_i,shard_dim_j,kernel_size", configs


@pytest.mark.parametrize(*generate_2d_configs())
def test_2d_splitting(
    ndims: int,
    shard_dim_i: int,
    shard_dim_j: int,
    kernel_size: int,
    device: torch.device,
):
    """
    Test distributed convolution with different number of dimensions and shard dimensions.
    Checks the output and gradients of the distributed convolution against the non-distributed
    convolution.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    # parallel_strategy.shard_dim = 2 + shard_dim
    parallel_strategy = ParallelStrategy(
        num_shards=(2, 2),
        shard_dim=[shard_dim_i + 2, shard_dim_j + 2],
        device_type=device.type,
    )

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
    if dist.get_rank() == 0:
        assert fp32_allclose(ref_y, ddpy)
    else:
        assert ddpy.numel() == 0
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
