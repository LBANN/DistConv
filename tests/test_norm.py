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


@pytest.fixture(scope="module")
def parallel_strategy_2d(device: torch.device):
    ps = ParallelStrategy(num_shards=(2, 2), shard_dim=(2, 3), device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


class Permuted(nn.Module):
    """Apply a module in channels-last order (e.g., ConvNeXt-style LayerNorm)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        ndims = x.dim()
        x = x.permute(0, *range(2, ndims), 1)
        x = self.module(x)
        return x.permute(0, ndims - 1, *range(1, ndims - 1))


def make_input(shape, device, memory_format=torch.contiguous_format):
    return (
        torch.randn(*shape, device=device)
        .to(memory_format=memory_format)
        .requires_grad_(True)
    )


def local_shard(tensor: torch.Tensor, parallel_strategy: ParallelStrategy):
    return DCTensor.distribute(tensor.detach(), parallel_strategy)._tensor


def check_against_reference(
    module: nn.Module, x: torch.Tensor, parallel_strategy: ParallelStrategy
):
    """
    Check the output and gradients of a module applied to a DCTensor against the
    non-distributed module. Outputs and input gradients are compared shard-wise, so this
    supports uneven sharding.
    """
    params = [p for p in module.parameters() if p.requires_grad]

    # Reference (non-distributed) forward and backward pass
    module.zero_grad()
    ref_y = module(x)
    grad_y = torch.randn_like(ref_y)
    ref_y.backward(grad_y)
    ref_x_grad = x.grad
    ref_param_grads = [p.grad.clone() for p in params]

    # Distributed forward and backward pass
    module.zero_grad()
    if params:
        module = DistConvDDP(module, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x.detach(), parallel_strategy).requires_grad_(True)
    dcy = module(dcx)
    dcy.backward(DCTensor.distribute(grad_y, parallel_strategy))

    assert isinstance(dcy, DCTensor)
    assert fp32_allclose(dcy._tensor, local_shard(ref_y, parallel_strategy))
    assert fp32_allclose(dcx.grad._tensor, local_shard(ref_x_grad, parallel_strategy))
    for p, ref_grad in zip(params, ref_param_grads):
        assert not isinstance(p.grad, DCTensor)
        assert fp32_allclose(p.grad, ref_grad)

    return dcy


def generate_group_norm_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            for num_groups in [1, 2, 8]:
                for affine in [True, False]:
                    configs.append((ndims, shard_dim, num_groups, affine))
    return "ndims,shard_dim,num_groups,affine", configs


@pytest.mark.parametrize(*generate_group_norm_configs())
def test_group_norm(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    num_groups: int,
    affine: bool,
    device: torch.device,
):
    parallel_strategy.shard_dim = 2 + shard_dim
    x = make_input([2, 8] + [16] * ndims, device)
    norm = nn.GroupNorm(num_groups, 8, affine=affine).to(device)
    check_against_reference(norm, x, parallel_strategy)


@pytest.mark.parametrize("ndims", [2, 3])
def test_group_norm_2d_splitting(
    parallel_strategy_2d: ParallelStrategy, ndims: int, device: torch.device
):
    parallel_strategy_2d.shard_dim = (ndims, ndims + 1)
    x = make_input([2, 8] + [16] * ndims, device)
    norm = nn.GroupNorm(2, 8).to(device)
    check_against_reference(norm, x, parallel_strategy_2d)


def test_group_norm_uneven(parallel_strategy: ParallelStrategy, device: torch.device):
    # 10 splits into (3, 3, 3, 1)
    parallel_strategy.shard_dim = 2
    x = make_input([2, 8, 10, 16], device)
    norm = nn.GroupNorm(2, 8).to(device)
    check_against_reference(norm, x, parallel_strategy)


@pytest.mark.parametrize(
    "ndims,memory_format", [(2, torch.channels_last), (3, torch.channels_last_3d)]
)
@pytest.mark.parametrize("norm_type", ["group", "instance", "layer"])
def test_norm_channels_last(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    memory_format: torch.memory_format,
    norm_type: str,
    device: torch.device,
):
    parallel_strategy.shard_dim = 3
    shape = [2, 8] + [16] * ndims
    x = make_input(shape, device, memory_format)
    if norm_type == "group":
        norm = nn.GroupNorm(2, 8)
    elif norm_type == "instance":
        norm = getattr(nn, f"InstanceNorm{ndims}d")(8, affine=True)
    else:
        norm = nn.LayerNorm(shape[1:])
    dcy = check_against_reference(norm.to(device), x, parallel_strategy)
    assert dcy._tensor.is_contiguous(memory_format=memory_format)


def generate_instance_norm_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            for affine in [True, False]:
                for track_running_stats in [True, False]:
                    configs.append((ndims, shard_dim, affine, track_running_stats))
    return "ndims,shard_dim,affine,track_running_stats", configs


@pytest.mark.parametrize(*generate_instance_norm_configs())
def test_instance_norm(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    affine: bool,
    track_running_stats: bool,
    device: torch.device,
):
    parallel_strategy.shard_dim = 2 + shard_dim
    x = make_input([2, 8] + [16] * ndims, device)
    norm_class = getattr(nn, f"InstanceNorm{ndims}d")
    ref_norm = norm_class(
        8, affine=affine, track_running_stats=track_running_stats
    ).to(device)
    norm = norm_class(8, affine=affine, track_running_stats=track_running_stats).to(
        device
    )
    norm.load_state_dict(ref_norm.state_dict())

    # Compare running statistics against the reference, which is updated twice to account for
    # both the reference and distributed passes in check_against_reference
    ref_norm(x)
    ref_norm(x)
    check_against_reference(norm, x, parallel_strategy)
    if track_running_stats:
        assert fp32_allclose(norm.running_mean, ref_norm.running_mean)
        assert fp32_allclose(norm.running_var, ref_norm.running_var)

        # Evaluation mode normalizes with the running statistics
        norm.eval()
        ref_norm.eval()
        with torch.no_grad():
            dcy = norm(DCTensor.distribute(x.detach(), parallel_strategy))
            assert fp32_allclose(dcy._tensor, local_shard(ref_norm(x), parallel_strategy))


def generate_layer_norm_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            # Number of trailing spatial dims normalized over (plus channels if "all")
            for normalized in ["all", "spatial", "last"]:
                for elementwise_affine in [True, False]:
                    configs.append((ndims, shard_dim, normalized, elementwise_affine))
    return "ndims,shard_dim,normalized,elementwise_affine", configs


def normalized_shape_for(shape, normalized):
    if normalized == "all":
        return shape[1:]
    elif normalized == "spatial":
        return shape[2:]
    return shape[-1:]


@pytest.mark.parametrize(*generate_layer_norm_configs())
def test_layer_norm(
    parallel_strategy: ParallelStrategy,
    ndims: int,
    shard_dim: int,
    normalized: str,
    elementwise_affine: bool,
    device: torch.device,
):
    parallel_strategy.shard_dim = 2 + shard_dim
    shape = [2, 8] + [16] * ndims
    x = make_input(shape, device)
    norm = nn.LayerNorm(
        normalized_shape_for(shape, normalized), elementwise_affine=elementwise_affine
    ).to(device)
    check_against_reference(norm, x, parallel_strategy)


@pytest.mark.parametrize("normalized", ["all", "spatial", "last"])
def test_layer_norm_2d_splitting(
    parallel_strategy_2d: ParallelStrategy, normalized: str, device: torch.device
):
    # With "last", only the shards along the last dimension may be combined
    parallel_strategy_2d.shard_dim = (2, 3)
    shape = [2, 8, 16, 16]
    x = make_input(shape, device)
    norm = nn.LayerNorm(normalized_shape_for(shape, normalized)).to(device)
    check_against_reference(norm, x, parallel_strategy_2d)


def test_layer_norm_uneven(parallel_strategy: ParallelStrategy, device: torch.device):
    parallel_strategy.shard_dim = 3
    shape = [2, 8, 16, 10]
    x = make_input(shape, device)
    norm = nn.LayerNorm(shape[2:]).to(device)
    check_against_reference(norm, x, parallel_strategy)


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_layer_norm_permuted(
    parallel_strategy: ParallelStrategy, shard_dim: int, device: torch.device
):
    # Normalizing over channels in channels-last order needs no reduction across shards
    parallel_strategy.shard_dim = 2 + shard_dim
    x = make_input([2, 8, 16, 16], device)
    norm = Permuted(nn.LayerNorm(8)).to(device)
    check_against_reference(norm, x, parallel_strategy)


def test_layer_norm_shape_mismatch(
    parallel_strategy: ParallelStrategy, device: torch.device
):
    parallel_strategy.shard_dim = 3
    norm = nn.LayerNorm([16, 12]).to(device)
    dcx = DCTensor.distribute(torch.randn(2, 8, 16, 16, device=device), parallel_strategy)
    with pytest.raises(ValueError, match="normalized_shape"):
        norm(dcx)


def test_conv_group_norm_model(
    parallel_strategy_2d: ParallelStrategy, device: torch.device
):
    parallel_strategy_2d.shard_dim = (2, 3)
    x = make_input([2, 4, 32, 32], device)
    model = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.GroupNorm(4, 8),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.InstanceNorm2d(8, affine=True),
        nn.Conv2d(8, 4, kernel_size=1),
        nn.LayerNorm([4, 32, 32]),
    ).to(device)
    check_against_reference(model, x, parallel_strategy_2d)
