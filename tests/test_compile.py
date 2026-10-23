"""Eager and full-graph Inductor parity checks for DCTensor."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy
from distconv import distconv as implementation


@pytest.fixture(autouse=True)
def isolate_compiler_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def parallel_strategy(device: torch.device):
    ps = ParallelStrategy(
        num_shards=dist.get_world_size(), device_type=device.type
    )
    yield ps
    cleanup_parallel_strategy(ps)


def test_fake_halos_do_not_communicate(device: torch.device):
    tensor = torch.randn(1, 4, 16, device=device)
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(tensor)
        with patch.object(
            dist,
            "batch_isend_irecv",
            side_effect=AssertionError("fake communication"),
        ):
            halo_minus, halo_plus = implementation._exchange_halos(
                fake, fake, 0, 2, 1, 1, True
            )
        for halo in (halo_minus, halo_plus):
            assert halo.shape == fake.shape
            assert halo.stride() == fake.stride()


def test_flatten_roundtrip(
    parallel_strategy: ParallelStrategy, device: torch.device
):
    local = torch.randn(2, 4, 16, device=device).transpose(0, 1)
    value = DCTensor(local, parallel_strategy)
    value._is_periodic = (True,)
    value._periodic_shard_padding = (1,)
    for halo in (None, torch.randn(4, 2, 18, device=device)):
        value._tensor_with_halo = halo
        names, metadata = value.__tensor_flatten__()
        with patch(
            "distconv.distconv.init_device_mesh",
            side_effect=AssertionError("mesh recreated"),
        ):
            restored = DCTensor.__tensor_unflatten__(
                {name: getattr(value, name) for name in names},
                deepcopy(metadata),
                value.shape,
                value.stride(),
            )
        assert restored._parallel_strategy is parallel_strategy
        assert restored._parallel_strategy.device_mesh is parallel_strategy.device_mesh
        assert restored._tensor is local
        assert restored._tensor_with_halo is halo
        assert restored._is_periodic == (True,)
        assert restored._periodic_shard_padding == (1,)
        assert restored.shape == value.shape
        assert restored.stride() == value.stride()
    restored = DCTensor.__tensor_unflatten__(
        {"_tensor": local}, metadata, (4, 2, 8), (16, 64, 2)
    )
    assert restored.shape == (4, 2, 8)
    assert restored.stride() == (16, 64, 2)


def test_compile_arithmetic(
    parallel_strategy: ParallelStrategy, device: torch.device
):
    compiled = torch.compile(
        lambda t: t.sin() + 1, backend="inductor", fullgraph=True
    )
    for _ in range(2):
        value = DCTensor(torch.randn(1, 4, 32, device=device), parallel_strategy)
        actual = compiled(value)
        assert actual._parallel_strategy is parallel_strategy
        assert fp32_allclose(actual._tensor, value._tensor.sin() + 1)


@pytest.mark.parametrize("ndims", [1, 2, 3])
@pytest.mark.parametrize("periodic", [False, True])
def test_compile_convolution(
    parallel_strategy: ParallelStrategy,
    device: torch.device,
    periodic: bool,
    ndims: int,
):
    """Compare outputs and gradients using DistConvDDP for synchronization."""
    torch.manual_seed(123)
    conv_class = getattr(nn, f"Conv{ndims}d")
    conv = conv_class(
        4,
        8,
        kernel_size=3,
        padding=1,
        padding_mode="circular" if periodic else "zeros",
    ).to(device)
    eager = DistConvDDP(deepcopy(conv), parallel_strategy=parallel_strategy)
    # Compile the convolution in full, keeping DDP's reducer outside the graph.
    compiled = DistConvDDP(
        torch.compile(deepcopy(conv), backend="inductor", fullgraph=True),
        parallel_strategy=parallel_strategy,
    )

    for iteration in range(2):
        torch.manual_seed(200 + iteration)
        shape = [1, 4] + [16] * ndims
        x = torch.randn(*shape, device=device)
        if ndims > 1:
            memory_format = (
                torch.channels_last if ndims == 2 else torch.channels_last_3d
            )
            x = x.contiguous(memory_format=memory_format)
        x.requires_grad_()

        # Compute the unsharded reference output and gradients.
        conv.zero_grad(set_to_none=True)
        ref_y = conv(x)
        ref_y.square().sum().backward()
        rank = dist.get_rank()
        ref_y_shard = ref_y.chunk(dist.get_world_size(), dim=2)[rank]
        ref_x_grad_shard = x.grad.chunk(dist.get_world_size(), dim=2)[rank]

        # DistConvDDP reduces and scales parameter gradients for both models.
        for model in (eager, compiled):
            model.zero_grad(set_to_none=True)
            dcx = DCTensor.distribute(
                x.detach().requires_grad_(), parallel_strategy
            )
            dcy = model(dcx)
            dcy.square().sum().backward()

            # assert dcx._tensor_with_halo is None
            assert fp32_allclose(dcy._tensor, ref_y_shard)
            assert fp32_allclose(dcx.grad._tensor, ref_x_grad_shard)
            for param, ref_param in zip(model.parameters(), conv.parameters()):
                assert fp32_allclose(param.grad, ref_param.grad)
