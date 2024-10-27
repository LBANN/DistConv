import pytest
from utils import fp32_allclose
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
from distconv import DCTensor, ParallelStrategy, DistConvDDP


def generate_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            for kernel_size in [1, 3, 5]:
                configs.append((ndims, shard_dim, kernel_size))

    return "ndims,shard_dim,kernel_size", configs


@pytest.mark.parametrize(*generate_configs())
def test_basic(ndims: int, shard_dim: int, kernel_size: int):
    """
    Test distributed convolution with different number of dimensions and shard dimensions.
    Checks the output and gradients of the distributed convolution against the non-distributed
    convolution.

    Args:
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
    """
    # Define the parallel strategy for distributed convolution
    ps = ParallelStrategy(num_shards=4, shard_dim=2 + shard_dim)

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device="cuda", requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, kernel_size=kernel_size, padding=kernel_size // 2).cuda()

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=ps)
    dcx = DCTensor.distribute(x, ps)
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
