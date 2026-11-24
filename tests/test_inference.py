"""Convolution inference must exchange the same halos as eager training."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from utils import cleanup_parallel_strategy

from distconv import DCTensor, ParallelStrategy


@pytest.fixture
def parallel_strategy(device):
    if dist.get_world_size() < 2:
        pytest.skip("Inference halo regression requires at least two ranks")
    strategy = ParallelStrategy(dist.get_world_size(), device_type=device.type)
    yield strategy
    cleanup_parallel_strategy(strategy)


def _input(ndims, device):
    torch.manual_seed(123)
    shape = [2, 4, 8 * dist.get_world_size()] + [8] * (ndims - 1)
    value = torch.randn(shape, device=device)
    if ndims > 1:
        value = value.contiguous(
            memory_format=torch.channels_last if ndims == 2 else torch.channels_last_3d
        )
    return value.requires_grad_()


def _mode(name):
    return {
        "train": nullcontext,
        "no_grad": torch.no_grad,
        "inference": torch.inference_mode,
    }[name]()


def _assert_global_reference(operation, value, strategy, mode):
    with _mode(mode):
        expected = operation(value)
        local = value.chunk(dist.get_world_size(), dim=2)[dist.get_rank()]
        shard = DCTensor.from_shard(local.detach().clone().requires_grad_(), strategy)
        actual = operation(shard)
        assert isinstance(actual, DCTensor)
        assert actual._parallel_strategy is strategy
        assert actual.requires_grad == (mode == "train")
        assert actual.is_inference() == (mode == "inference")
        torch.testing.assert_close(
            actual._tensor,
            expected.chunk(dist.get_world_size(), dim=2)[dist.get_rank()],
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("mode", ["train", "no_grad", "inference"])
@pytest.mark.parametrize("ndims", [1, 2, 3])
@pytest.mark.parametrize(
    "case",
    [
        "conv_default",
        "conv_grouped",
        "transpose_default",
        "transpose_halo",
        "transpose_even",
    ],
)
def test_convolution_modes_match_global_native(
    parallel_strategy, device, mode, ndims, case
):
    """Cover schema defaults, keyword binding, and both transpose halo cases."""
    value = _input(ndims, device)
    transpose = case.startswith("transpose")
    kernel = 1 if case.endswith("default") else 2 if case.endswith("even") else 3
    groups = 2 if case == "conv_grouped" else 1
    weight_shape = [4, 6 // groups] if transpose else [6, 4 // groups]
    weight = torch.randn(
        *weight_shape, *([kernel] * ndims), device=device
    ).requires_grad_()
    bias = torch.randn(6, device=device, requires_grad=True)
    if transpose:
        operation = getattr(torch.ops.aten, f"conv_transpose{ndims}d")
        operation = operation.default if ndims == 1 else operation.input
    else:
        operation = getattr(torch.ops.aten, f"conv{ndims}d").default

    def run(tensor):
        if case.endswith("default"):
            # Omit every optional schema argument, including the bias.
            return operation(tensor, weight)
        if transpose:
            return operation(
                input=tensor,
                weight=weight,
                bias=bias,
                stride=[2] * ndims,
                padding=[1 if kernel == 3 else 0] * ndims,
                output_padding=[1 if kernel == 3 else 0] * ndims,
                groups=groups,
                dilation=[1] * ndims,
            )
        return operation(
            tensor,
            weight,
            bias=bias,
            padding=[1] * ndims,
            stride=[2] * ndims,
            groups=groups,
        )

    _assert_global_reference(run, value, parallel_strategy, mode)


@pytest.mark.parametrize("mode", ["train", "no_grad", "inference"])
@pytest.mark.parametrize("ndims", [1, 2, 3])
@pytest.mark.parametrize("padding,kernel", [("same", 3), ("valid", 1)])
def test_string_padding_matches_global_native(
    parallel_strategy, device, mode, ndims, padding, kernel
):
    value = _input(ndims, device)
    weight = torch.randn(6, 4, *([kernel] * ndims), device=device).requires_grad_()
    operation = getattr(torch.ops.aten, f"conv{ndims}d").padding
    _assert_global_reference(
        lambda tensor: operation(tensor, weight, padding=padding),
        value,
        parallel_strategy,
        mode,
    )


@pytest.mark.parametrize("mode", ["train", "no_grad", "inference"])
@pytest.mark.parametrize("ndims", [2, 3])
def test_asymmetric_same_padding_on_unsharded_axis(
    parallel_strategy, device, mode, ndims
):
    """Even kernels on an unlisted axis must retain native asymmetric padding."""
    value = _input(ndims, device)
    kernel_shape = [3] * (ndims - 1) + [2]
    weight = torch.randn(6, 4, *kernel_shape, device=device).requires_grad_()
    operation = getattr(torch.nn.functional, f"conv{ndims}d")
    _assert_global_reference(
        lambda tensor: operation(tensor, weight, padding="same"),
        value,
        parallel_strategy,
        mode,
    )


@pytest.mark.parametrize("ndims", [1, 2, 3])
@pytest.mark.parametrize(
    "padding,kernel,message",
    [
        ("same", 2, "DistConv: asymmetric padding='same'"),
        (
            "valid",
            3,
            'DistConv: when kernel size is odd, padding must be equivalent to "same"',
        ),
    ],
)
def test_unsupported_inference_padding_raises(
    parallel_strategy, device, ndims, padding, kernel, message
):
    value = _input(ndims, device)
    weight = torch.randn(6, 4, *([kernel] * ndims), device=device)
    operation = getattr(torch.ops.aten, f"conv{ndims}d").padding
    with torch.inference_mode():
        local = value.chunk(dist.get_world_size(), dim=2)[dist.get_rank()]
        shard = DCTensor.from_shard(local, parallel_strategy)
        with pytest.raises(Exception, match=message):
            operation(shard, weight, padding=padding)
