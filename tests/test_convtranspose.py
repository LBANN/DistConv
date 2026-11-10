import pytest
import torch
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


@pytest.fixture(scope="module")
def parallel_strategy(device: torch.device):
    ps = ParallelStrategy(num_shards=4, device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


def find_padding(kernel_size, stride=1, explicit_padding=False):
    ep = kernel_size // 2 if explicit_padding else 0
    pad = (kernel_size + 2 * ep * stride - 1) // 2
    out_pad = stride - 1
    if explicit_padding:
        return pad, out_pad, ep
    return pad, out_pad


def generate_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            for kernel_size in [1, 3, 5]:
                for stride in [1, 2, 4]:
                    configs.append((ndims, shard_dim, kernel_size, stride))

    return "ndims,shard_dim,kernel_size,stride", configs


@pytest.mark.parametrize(*generate_configs())
def test_transposeconv_zerospadding(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    kernel_size: int,
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
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim
    padding, output_padding = find_padding(kernel_size, stride)

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)
    conv_class = getattr(nn, f"ConvTranspose{ndims}d")
    conv = conv_class(
        4,
        8,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        output_padding=output_padding,
    ).to(device)

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = dist_conv(dcx)
    dcy_merge = dcy.to_replicate()
    dc_loss = dcy.to_ddp().square().mean()
    dc_loss.backward()
    x_grad = dcx.grad.to_replicate()
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
        shard_dim (int): Dimension along which the tensor is sharded.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        device (torch.device): Torch device to run test with.
    """
    # Set the shard dimension for the parallel strategy
    parallel_strategy.shard_dim = 2 + shard_dim
    padding, output_padding, explicit_padding = find_padding(
        kernel_size, stride, explicit_padding=True
    )

    # Initialize the input tensor and convolution layer
    shape = [1, 4] + [64] * ndims
    x = torch.randn(*shape, device=device, requires_grad=True)

    conv_kwargs = dict(
        kernel_size=kernel_size, stride=stride, output_padding=output_padding
    )

    # set periodic padding case for reference
    explicit_padding = [explicit_padding, explicit_padding] * ndims
    x_periodic = torch.nn.functional.pad(input=x, pad=explicit_padding, mode="circular")

    conv_class = getattr(nn, f"ConvTranspose{ndims}d")
    conv = (
        conv_class(4, 8, padding=padding, **conv_kwargs)
        .to(device)
        .requires_grad_(False)
    )
    conv.requires_grad_(True)

    # Perform forward and backward pass for reference (non-distributed) convolution
    conv.zero_grad()
    ref_y = conv(x_periodic)
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    # Perform forward and backward pass for distributed convolution
    conv.zero_grad()
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcx_periodic = torch.nn.functional.pad(
        input=dcx, pad=explicit_padding, mode="circular"
    )
    dcy = dist_conv(dcx_periodic)
    dcy_merge = dcy.to_replicate()
    dc_loss = dcy.to_ddp().square().mean()
    dc_loss.backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    # Validate the results
    assert fp32_allclose(ref_y, dcy_merge)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)


def run_transposeconv_1d(
    parallel_strategy: ParallelStrategy,
    device: torch.device,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    periodic_padding: int | None,
):
    """
    Run a 1D transposed convolution distributed and non-distributed and compare the
    output and gradients. If periodic_padding is not None, the input is first padded
    circularly by that amount.
    """
    parallel_strategy.shard_dim = 2
    x = torch.randn(1, 4, 64, device=device, requires_grad=True)
    conv = nn.ConvTranspose1d(
        4,
        8,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    ).to(device)

    def pad(t):
        if periodic_padding is None:
            return t
        return nn.functional.pad(t, (periodic_padding, periodic_padding), "circular")

    conv.zero_grad()
    ref_y = conv(pad(x))
    ref_y.square().mean().backward()
    ref_x_grad = x.grad
    ref_conv_grad = conv.weight.grad

    conv.zero_grad()
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy)
    dcy = dist_conv(pad(dcx))
    dcy_merge = dcy.to_replicate()
    dcy.to_ddp().square().mean().backward()
    x_grad = dcx.grad.to_replicate()
    dc_conv_grad = conv.weight.grad

    assert fp32_allclose(ref_y, dcy_merge)
    assert fp32_allclose(ref_x_grad, x_grad)
    assert fp32_allclose(ref_conv_grad, dc_conv_grad)


@pytest.mark.parametrize(
    "kernel_size,stride,padding,output_padding,periodic_padding",
    [
        # Even kernels no larger than the stride need no halo
        (2, 2, 0, 0, None),
        (4, 4, 0, 0, None),
        (2, 4, 0, 2, None),
        (2, 2, 2, 0, 1),
        # Odd kernels with padding other than (kernel_size - 1) / 2
        (5, 4, 1, 1, None),
        (1, 2, 0, 1, None),
        # Periodic padding larger than the halo needed
        (3, 1, 3, 0, 2),
        (5, 2, 4, 1, 1),
    ],
)
def test_transposeconv_supported_configs(
    parallel_strategy: ParallelStrategy,
    device: torch.device,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    periodic_padding: int | None,
):
    """
    Test transposed convolution configurations outside the standard
    "same"-style setup that DistConv computes correctly.
    """
    run_transposeconv_1d(
        parallel_strategy,
        device,
        kernel_size,
        stride,
        padding,
        output_padding,
        periodic_padding,
    )


@pytest.mark.parametrize(
    "kernel_size,stride,padding,output_padding,periodic_padding",
    [
        # Even kernels larger than the stride need a halo that isn't exchanged
        (4, 2, 1, 0, None),
        (6, 2, 2, 0, None),
        (4, 4, 1, 2, None),
        (4, 2, 3, 0, 1),
        # Periodic padding smaller than the halo needed
        (5, 1, 3, 0, 1),
        (7, 1, 4, 0, 1),
        # Periodic padding leaves a negative local padding
        (1, 3, 2, 0, 1),
    ],
)
def test_transposeconv_unsupported_configs(
    parallel_strategy: ParallelStrategy,
    device: torch.device,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    periodic_padding: int | None,
):
    """
    Test that transposed convolution configurations DistConv cannot compute
    correctly raise instead of returning wrong values.
    """
    with pytest.raises(Exception, match="DistConv"):
        run_transposeconv_1d(
            parallel_strategy,
            device,
            kernel_size,
            stride,
            padding,
            output_padding,
            periodic_padding,
        )


@pytest.mark.parametrize(
    "kernel_size,stride,padding,padding_mode",
    [
        # Padding smaller than the halo
        (3, 4, 0, "zeros"),
        (5, 4, 1, "zeros"),
        (5, 4, 1, "circular"),
        # Even kernels with nonzero padding
        (4, 4, 1, "zeros"),
        (4, 2, 1, "zeros"),
    ],
)
def test_conv_unsupported_configs(
    parallel_strategy: ParallelStrategy,
    device: torch.device,
    kernel_size: int,
    stride: int,
    padding: int,
    padding_mode: str,
):
    """
    Test that regular convolution configurations that produce the right local
    output size but misaligned halos raise instead of returning wrong values.
    """
    parallel_strategy.shard_dim = 2
    conv = nn.Conv1d(
        4,
        8,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
    ).to(device)
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    x = torch.randn(1, 4, 64, device=device)
    dcx = DCTensor.distribute(x, parallel_strategy)
    with pytest.raises(Exception, match="DistConv"):
        dist_conv(dcx)


@pytest.mark.parametrize(
    "conv_cls,kwargs",
    [
        (
            nn.ConvTranspose1d,
            dict(kernel_size=5, stride=2, padding=2, output_padding=1),
        ),
        (nn.Conv1d, dict(kernel_size=5, padding=2)),
    ],
)
def test_halo_larger_than_local_input(
    parallel_strategy: ParallelStrategy,
    device: torch.device,
    conv_cls: type,
    kwargs: dict,
):
    """
    Test that a local input smaller than the halo raises a DistConv error
    instead of failing inside the halo exchange.
    """
    parallel_strategy.shard_dim = 2
    conv = conv_cls(4, 8, **kwargs).to(device)
    dist_conv = DistConvDDP(conv, parallel_strategy=parallel_strategy)
    # One element per shard, but kernel size 5 needs a halo of 2
    x = torch.randn(1, 4, 4, device=device)
    dcx = DCTensor.distribute(x, parallel_strategy)
    with pytest.raises(Exception, match="DistConv: the local input size"):
        dist_conv(dcx)
