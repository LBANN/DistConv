import pytest
import torch
import torch.nn as nn
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
                for num_shards in [2]:
                    configs.append((ndims, shard_dim, kernel_size, num_shards))
    return "ndims,shard_dim,kernel_size,num_shards", configs


@pytest.mark.parametrize(*generate_configs())
def test_double_backprop_gradientloss(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    num_shards: int,
    device: torch.device,
):
    """
    Test distributed convolution with different number of dimensions and shard dimensions.
    Also consider hybrid spatial-data parallelism.
    Checks the output and gradients of the distributed convolution against the reference DDP
    convolution.

    Args:
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        num_shards (int): Number of spatial partitions for data
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = shard_dim + 2

    conv_kwargs = dict(
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=False,
        stride=1,
        padding_mode="circular",
    )

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [16] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, **conv_kwargs).to(device).requires_grad_(False)
    conv.requires_grad_(True)

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x)
    # find gradient wrt input
    ref_grads = torch.autograd.grad(
        outputs=[ref_y.sum()], inputs=[x], create_graph=True
    )[0]
    # find all losses
    ref_loss_grad = ref_grads.mean()
    ref_loss = ref_loss_grad
    ref_loss.backward()
    ref_conv_grad = conv.weight.grad.clone()

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    # find gradient wrt input
    dc_grads = torch.autograd.grad(
        outputs=[ddpy.sum()], inputs=[dcx], create_graph=True
    )[0]
    dc_grads_rep = dc_grads.to_replicate()
    # find all losses
    dc_loss_grad = dc_grads_rep.mean()
    dc_loss = dc_loss_grad
    dc_loss.backward()
    dc_conv_grad = ddp_conv.module.weight.grad

    # Validate the results
    assert fp32_allclose(ref_loss, dc_loss)
    assert fp32_allclose(ref_y, ddpy)
    assert fp32_allclose(ref_grads, dc_grads_rep)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)


@pytest.mark.parametrize(*generate_configs())
def test_double_backprop_combinedloss(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
    num_shards: int,
    device: torch.device,
):
    """
    Test distributed convolution with different number of dimensions and shard dimensions.
    Also consider hybrid spatial-data parallelism.
    Checks the output and gradients of the distributed convolution against the reference DDP
    convolution.

    Args:
        ndims (int): Number of dimensions for the convolution (1, 2, or 3).
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        num_shards (int): Number of spatial partitions for data
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = shard_dim + 2

    conv_kwargs = dict(
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        bias=False,
        stride=1,
        padding_mode="circular",
    )

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [16] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(4, 8, **conv_kwargs).to(device).requires_grad_(False)
    conv.requires_grad_(True)

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x)
    # find gradient wrt input
    ref_grads = torch.autograd.grad(
        outputs=[ref_y.sum()], inputs=[x], create_graph=True
    )[0]
    # find all losses
    ref_loss_y = ref_y.square().norm()
    ref_loss_grad = ref_grads.mean()
    ref_loss = ref_loss_y + ref_loss_grad
    ref_loss.backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad.clone()

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    ddp_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = ddp_conv(dcx)
    ddpy = dcy.to_replicate()
    # find gradient wrt input
    dc_grads = torch.autograd.grad(
        outputs=[ddpy.sum()], inputs=[dcx], create_graph=True
    )[0]
    dc_grads_rep = dc_grads.to_replicate()
    # find all losses
    dc_loss_y = ddpy.square().norm()
    dc_loss_grad = dc_grads_rep.mean()
    dc_loss = dc_loss_y + dc_loss_grad
    dc_loss.backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = ddp_conv.module.weight.grad

    # Validate the results
    assert fp32_allclose(ref_loss, dc_loss)
    assert fp32_allclose(ref_y, ddpy)
    assert fp32_allclose(ref_grads, dc_grads_rep)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)
