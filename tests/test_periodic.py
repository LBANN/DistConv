import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


def generate_configs():
    configs = []
    # NOTE: Not working for 3D yet
    for ndims in [1, 2]:
        for shard_dim in range(ndims):
            for kernel_size in [1, 3, 5]:
                for stride in [1, 2]:
                    configs.append((ndims, shard_dim, kernel_size, stride))

    return "ndims,shard_dim,kernel_size,stride", configs


@pytest.mark.parametrize(*generate_configs())
def test_periodic(
    ndims: int,
    shard_dim: int,
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
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy = ParallelStrategy(
        num_shards=4,
        shard_dim=shard_dim + 2,
        device_type=device.type,
        is_periodic=True,
        space_ndim=ndims,
    )

    conv_kwargs = dict(
        kernel_size=kernel_size, padding=kernel_size // 2, bias=False, stride=stride
    )

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    # x = torch.ones(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, **conv_kwargs).to(device).requires_grad_(False)
    ref_conv = (
        conv_class(4, 8, padding_mode="circular", **conv_kwargs)
        .to(device)
        .requires_grad_(False)
    )
    conv.weight.copy_(ref_conv.weight)
    conv.requires_grad_(True)
    ref_conv.requires_grad_(True)

    # Perform forward and backward pass for reference (non-distributed) convolution
    ref_conv.zero_grad()
    ref_y = ref_conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = ref_conv.weight.grad

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
    assert fp32_allclose(
        ref_x_grad, x_grad
    ), f"max error is {torch.abs(ref_x_grad-x_grad).max()}"
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
