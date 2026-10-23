"""Two-rank eager/Inductor parity checks; run sequentially via bash run.sh."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from distconv import DCTensor, ParallelStrategy
from distconv import distconv as implementation
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture(autouse=True)
def isolate_compiler_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def strategy(device):
    with patch('distconv.distconv.init_device_mesh', wraps=implementation.init_device_mesh) as initialize:
        strategy = ParallelStrategy(num_shards=dist.get_world_size(), device_type=device.type)
        assert initialize.call_count == 1
        yield strategy
        assert initialize.call_count == 1


def test_fake_halos_do_not_communicate(device):
    tensor = torch.randn(1, 4, 16, device=device)
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(tensor)
        with patch.object(dist, 'batch_isend_irecv', side_effect=AssertionError('fake communication')):
            extended = implementation._forward_halo(fake, 1, 2, 2, 0, 1, 1, True)
            restored = implementation._backward_halo(extended, 1, 2, 2, 0, 1, 1, True)
        assert extended.shape == (1, 4, 18)
        assert restored.shape == fake.shape
        assert restored.stride() == fake.stride()


def test_flatten_roundtrip(strategy, device):
    local = torch.randn(2, 4, 16, device=device).transpose(0, 1)
    value = DCTensor(local, strategy)
    value._is_periodic = (True,)
    value._periodic_shard_padding = (1,)
    for halo in (None, torch.randn(4, 2, 18, device=device)):
        value._tensor_with_halo = halo
        names, metadata = value.__tensor_flatten__()
        with patch('distconv.distconv.init_device_mesh', side_effect=AssertionError('mesh recreated')):
            restored = DCTensor.__tensor_unflatten__(
                {name: getattr(value, name) for name in names}, deepcopy(metadata),
                value.shape, value.stride(),
            )
        assert restored._parallel_strategy is strategy
        assert restored._parallel_strategy.device_mesh is strategy.device_mesh
        assert restored._tensor is local
        assert restored._tensor_with_halo is halo
        assert restored._is_periodic == (True,)
        assert restored._periodic_shard_padding == (1,)
        assert restored.shape == value.shape
        assert restored.stride() == value.stride()
    restored = DCTensor.__tensor_unflatten__(
        {'_tensor': local}, metadata, (4, 2, 8), (16, 64, 2)
    )
    assert restored.shape == (4, 2, 8)
    assert restored.stride() == (16, 64, 2)


def test_compile_arithmetic(strategy, device):
    compiled = torch.compile(lambda t: t.sin() + 1, backend='inductor', fullgraph=True)
    with patch('distconv.distconv.init_device_mesh', side_effect=AssertionError('mesh recreated')):
        for _ in range(2):
            value = DCTensor(torch.randn(1, 4, 32, device=device), strategy)
            actual = compiled(value)
            assert actual._parallel_strategy is strategy
            torch.testing.assert_close(actual._tensor, value._tensor.sin() + 1)


@pytest.mark.parametrize('ndims', [1, 2, 3])
@pytest.mark.parametrize('periodic', [False, True])
def test_compile_convolution(strategy, device, periodic, ndims):
    torch.manual_seed(123)
    conv_class = getattr(torch.nn, f'Conv{ndims}d')
    conv = conv_class(4, 8, 3, padding=1,
                           padding_mode='circular' if periodic else 'zeros').to(device)
    eager = deepcopy(conv)
    optimized = deepcopy(conv)
    compiled = torch.compile(optimized, backend='inductor', fullgraph=True)
    for iteration in range(2):
        torch.manual_seed(200 + iteration)
        x = torch.randn([1, 4] + [16] * ndims, device=device)
        if ndims > 1:
            memory_format = torch.channels_last if ndims == 2 else torch.channels_last_3d
            x = x.contiguous(memory_format=memory_format)
        x.requires_grad_()
        conv.zero_grad(set_to_none=True)
        reference = conv(x)
        reference.square().sum().backward()
        for model in (eager, compiled):
            model.zero_grad(set_to_none=True)
            value = DCTensor.distribute(x.detach().requires_grad_(), strategy)
            original = value._tensor
            original_data = original.clone()
            with patch('distconv.distconv.init_device_mesh', side_effect=AssertionError('mesh recreated')):
                actual = model(value)
                actual.square().sum().backward()
            assert value._tensor is original
            torch.testing.assert_close(value._tensor, original_data)
            assert value._tensor_with_halo is None
            rank = dist.get_rank()
            torch.testing.assert_close(actual._tensor, reference.chunk(dist.get_world_size(), dim=2)[rank])
            torch.testing.assert_close(value.grad._tensor, x.grad.chunk(dist.get_world_size(), dim=2)[rank], rtol=1e-3, atol=1e-4)
            for param, ref_param in zip(model.parameters(), conv.parameters()):
                dist.all_reduce(param.grad)
                torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-4, atol=1e-4)
