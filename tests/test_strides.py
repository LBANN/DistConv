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


def generate_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for kernel_size in [1, 2, 3]:
            padding = 1 if kernel_size == 3 else 0
            for stride in [2, 4]:
                configs.append((ndims, kernel_size, padding, stride))

    return "ndims,kernel_size,padding,stride", configs


@pytest.mark.parametrize(*generate_configs())
def test_strides(
    parallel_strategy: ParallelStrategy,
    ndims: int,
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
    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, kernel_size=kernel_size, padding=padding, stride=stride).to(
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
