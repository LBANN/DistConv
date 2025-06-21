import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


@pytest.fixture(scope="module")
def parallel_strategy(device: torch.device):
    ps = ParallelStrategy(num_shards=2, device_type=device.type)
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
def test_ddp_with_distconv(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    device: torch.device,
):
    """
    Test DDP with distributed convolution using different number of dimensions and shard dimensions.
    Checks the output and gradients of the DDP with distributed convolution against the DDP-only
    convolution.

    Args:
        parallel_strategy (ParallelStrategy): Parallel strategy for the distributed convolution.
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim

    # Initialize the input tensor (and distribute it) and convolution layer
    shape = [2, 4] + [64] * ndims
    x = torch.randn(*shape, device=device)
    x = (
        distribute_tensor(x, parallel_strategy.device_mesh, [Shard(0), Replicate()])
        .to_local()
        .requires_grad_()
    )
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, kernel_size=kernel_size, padding=kernel_size // 2).to(
        device
    )
    ddp_conv = DDP(conv)

    # Perform forward and backward pass for reference DDP-only convolution
    conv.zero_grad()
    ref_y = ddp_conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    dc_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = dc_conv(dcx)
    ddpy = dcy.to_ddp()
    ddpy.square().mean().backward()
    x_grad = dcx.grad.to_ddp()
    dc_conv_grad = conv.weight.grad

    # Validate the results
    if dist.get_rank() % 2 == 0:
        assert fp32_allclose(ref_y, ddpy)
    else:
        assert ddpy.numel() == 0
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
