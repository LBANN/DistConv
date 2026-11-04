from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy
from distconv.norm import group_norm_forward


@pytest.fixture(scope="module")
def parallel_strategy(device: torch.device):
    ps = ParallelStrategy(num_shards=4, device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


@pytest.fixture(scope="module")
def parallel_strategy_unsharded(device: torch.device):
    ps = ParallelStrategy(num_shards=1, device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


@pytest.fixture(scope="module")
def parallel_strategy_2d(device: torch.device):
    ps = ParallelStrategy(num_shards=(2, 2), shard_dim=(2, 3), device_type=device.type)
    yield ps
    cleanup_parallel_strategy(ps)


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_group_norm_mixed_precision(
    parallel_strategy: ParallelStrategy, dtype: torch.dtype, device: torch.device
):
    # Reduced-precision input with float32 affine parameters. The input has a small variance
    # relative to its mean (large rstd), so rounding the saved statistics to the input dtype
    # would produce badly wrong input gradients. Native mixed-type group norm is only supported on
    # CPU, so the reference is computed there.
    parallel_strategy.shard_dim = 2
    x = 1 + torch.randint(-2, 3, (2, 8, 16)) / 128
    x = x.to(dtype).requires_grad_(True)
    norm = nn.GroupNorm(2, 8)
    nn.init.uniform_(norm.weight, 0.5, 1.5)
    nn.init.uniform_(norm.bias, -0.5, 0.5)

    ref_y = norm(x)
    grad_y = torch.randn_like(ref_y)
    ref_y.backward(grad_y)
    ref_y, ref_x_grad, grad_y = (t.to(device) for t in (ref_y, x.grad, grad_y))
    ref_param_grads = [norm.weight.grad.to(device), norm.bias.grad.to(device)]
    x = x.detach().to(device)
    norm = norm.to(device)
    norm.zero_grad()

    ddp_norm = DistConvDDP(norm, parallel_strategy=parallel_strategy)
    dcx = DCTensor.distribute(x, parallel_strategy).requires_grad_(True)
    dcy = ddp_norm(dcx)
    dcy.backward(DCTensor.distribute(grad_y, parallel_strategy))

    def close(a, b):
        return torch.allclose(a.float(), b.float(), rtol=2e-2, atol=2e-2)

    assert dcy.dtype == dtype and dcx.grad.dtype == dtype
    assert close(dcy._tensor, local_shard(ref_y, parallel_strategy))
    assert close(dcx.grad._tensor, local_shard(ref_x_grad, parallel_strategy))
    for p, ref_grad in zip([norm.weight, norm.bias], ref_param_grads):
        assert p.grad.dtype == torch.float32
        assert close(p.grad, ref_grad)


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
        norm = nn.LayerNorm(shape[1:], elementwise_affine=False)
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
    ref_norm = norm_class(8, affine=affine, track_running_stats=track_running_stats).to(
        device
    )
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
            assert fp32_allclose(
                dcy._tensor, local_shard(ref_norm(x), parallel_strategy)
            )


# Globally empty spatial inputs (as opposed to empty local shards of a nonempty input) have NaN
# statistics, which must not be folded into the running statistics. The sharded spatial dimension
# (2) or an unsharded one is empty. An empty batch does update them (to NaN) in the native op.
@pytest.mark.parametrize("shape", [(2, 8, 0), (2, 8, 16, 0), (0, 8, 16)])
def test_instance_norm_empty(
    parallel_strategy: ParallelStrategy, shape: tuple, device: torch.device
):
    parallel_strategy.shard_dim = 2
    norm_class = getattr(nn, f"InstanceNorm{len(shape) - 2}d")
    ref_norm = norm_class(8, track_running_stats=True).to(device)
    norm = norm_class(8, track_running_stats=True).to(device)
    x = torch.randn(*shape, device=device)

    ref_y = ref_norm(x)
    dcy = norm(DCTensor.distribute(x, parallel_strategy))

    assert dcy._tensor.shape == local_shard(ref_y, parallel_strategy).shape
    for buf, ref_buf in [
        (norm.running_mean, ref_norm.running_mean),
        (norm.running_var, ref_norm.running_var),
    ]:
        torch.testing.assert_close(buf, ref_buf, rtol=0, atol=0, equal_nan=True)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).abs().max() / expected.abs().max()).item()


# Maximum relative error against a float64 reference for correctly computed results
REDUCED_PRECISION_TOLERANCE = {torch.bfloat16: 1e-2, torch.float16: 1e-3}


# Reduced-precision inputs with a small variance relative to their mean (large rstd), for which
# reduced-precision statistics badly corrupt the input gradient and running mean. This happened
# without affine parameters, since instance norm followed group norm's statistics dtype rule.
# Native CPU instance norm is itself imprecise here in some torch versions (2.14 casts float32
# running statistics to the input dtype), so results are compared against a float64 reference.
@pytest.mark.parametrize("track_running_stats", [True, False])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_instance_norm_mixed_precision(
    parallel_strategy: ParallelStrategy,
    dtype: torch.dtype,
    affine: bool,
    track_running_stats: bool,
    device: torch.device,
):
    parallel_strategy.shard_dim = 2
    norm = nn.InstanceNorm1d(4, affine=affine, track_running_stats=track_running_stats)
    if affine:
        nn.init.uniform_(norm.weight, 0.5, 1.5)
        nn.init.uniform_(norm.bias, -0.5, 0.5)
    ref_norm = deepcopy(norm).double()
    x = (1 + torch.randint(-2, 3, (2, 4, 16)) / 128).to(dtype)
    grad_y = torch.randn(2, 4, 16).to(dtype)

    ref_x = x.double().requires_grad_(True)
    ref_y = ref_norm(ref_x)
    ref_y.backward(grad_y.double())

    norm = norm.to(device)
    module = DistConvDDP(norm, parallel_strategy=parallel_strategy) if affine else norm
    dcx = DCTensor.distribute(x.to(device), parallel_strategy).requires_grad_(True)
    dcy = module(dcx)
    dcy.backward(DCTensor.distribute(grad_y.to(device), parallel_strategy))

    def check(actual, expected, tolerance):
        assert relative_error(actual.cpu(), expected.detach().cpu()) < tolerance

    def check_shard(actual, expected, tolerance):
        check(actual, local_shard(expected.to(device), parallel_strategy), tolerance)

    tolerance = REDUCED_PRECISION_TOLERANCE[dtype]
    assert dcy.dtype == dtype and dcx.grad.dtype == dtype
    check_shard(dcy._tensor, ref_y, tolerance)
    check_shard(dcx.grad._tensor, ref_x.grad, tolerance)
    if affine:
        check(norm.weight.grad, ref_norm.weight.grad, 1e-4)
        check(norm.bias.grad, ref_norm.bias.grad, 1e-4)
    if track_running_stats:
        check(norm.running_mean, ref_norm.running_mean, 1e-5)
        check(norm.running_var, ref_norm.running_var, 1e-5)


# The case reported in review, unsharded and sharded (one element per shard): a bfloat16 input
# with float32 running statistics and no affine parameters. With reduced-precision statistics,
# the mean of 1.0039 rounds to 1.0, giving input gradients of [-298, -700, -502, 298] and a
# running mean of 0.1.
@pytest.mark.parametrize(
    "strategy_name", ["parallel_strategy_unsharded", "parallel_strategy"]
)
def test_instance_norm_mixed_precision_review_case(
    request: pytest.FixtureRequest, strategy_name: str, device: torch.device
):
    parallel_strategy = request.getfixturevalue(strategy_name)
    parallel_strategy.shard_dim = 2
    norm = nn.InstanceNorm1d(1, track_running_stats=True).to(device)
    x = torch.tensor(
        [[[1.0, 1.0 + 1 / 128, 1.0 + 1 / 128, 1.0]]],
        dtype=torch.bfloat16,
        device=device,
    )
    grad_y = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=x.dtype, device=device)

    dcx = DCTensor.distribute(x, parallel_strategy).requires_grad_(True)
    norm(dcx).backward(DCTensor.distribute(grad_y, parallel_strategy))

    expected_grad = torch.tensor([[[-298.0, -99.5, 99.5, 298.0]]], device=device)
    assert torch.equal(
        dcx.grad._tensor.float(), local_shard(expected_grad, parallel_strategy)
    )
    expected_mean = torch.tensor([0.1 * (1 + 1 / 256)], device=device)
    torch.testing.assert_close(norm.running_mean, expected_mean)


# Saved statistics follow native group norm's dtype rule by default, and are kept in full
# precision for instance norm
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("full_precision_stats", [True, False])
def test_group_norm_forward_stats_dtype(
    affine: bool, full_precision_stats: bool, device: torch.device
):
    x = torch.randn(2, 4, 8, device=device, dtype=torch.bfloat16)
    weight = torch.ones(4, device=device) if affine else None
    bias = torch.zeros(4, device=device) if affine else None
    _, mean, rstd = group_norm_forward(
        x, weight, bias, 2, 1e-5, [], full_precision_stats=full_precision_stats
    )
    if full_precision_stats or affine:
        expected = torch.float32
    else:
        _, native_mean, _ = torch.native_group_norm(x, None, None, 2, 4, 8, 2, 1e-5)
        expected = native_mean.dtype
    assert mean.dtype == rstd.dtype == expected


class TorchInstanceNorm(nn.Module):
    """Apply an InstanceNorm module's parameters by calling torch.instance_norm directly."""

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        n = self.norm
        return torch.instance_norm(
            x,
            n.weight,
            n.bias,
            n.running_mean,
            n.running_var,
            True,
            n.momentum,
            n.eps,
            torch.backends.cudnn.enabled,
        )


@pytest.mark.parametrize("track_running_stats", [True, False])
def test_torch_instance_norm(
    parallel_strategy: ParallelStrategy,
    track_running_stats: bool,
    device: torch.device,
):
    parallel_strategy.shard_dim = 2
    x = make_input([2, 8, 16, 16], device)
    ref_norm = nn.InstanceNorm2d(
        8, affine=True, track_running_stats=track_running_stats
    ).to(device)
    norm = nn.InstanceNorm2d(
        8, affine=True, track_running_stats=track_running_stats
    ).to(device)
    norm.load_state_dict(ref_norm.state_dict())

    # Updated twice to match the reference and distributed passes in check_against_reference
    ref_norm(x)
    ref_norm(x)
    check_against_reference(TorchInstanceNorm(norm), x, parallel_strategy)
    if track_running_stats:
        assert fp32_allclose(norm.running_mean, ref_norm.running_mean)
        assert fp32_allclose(norm.running_var, ref_norm.running_var)


def generate_layer_norm_configs():
    configs = []
    for ndims in [1, 2, 3]:
        for shard_dim in range(ndims):
            # Number of trailing spatial dims normalized over (plus channels if "all")
            for normalized in ["all", "spatial", "last"]:
                configs.append((ndims, shard_dim, normalized))
    return "ndims,shard_dim,normalized", configs


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
    device: torch.device,
):
    parallel_strategy.shard_dim = 2 + shard_dim
    shape = [2, 8] + [16] * ndims
    x = make_input(shape, device)
    norm = nn.LayerNorm(
        normalized_shape_for(shape, normalized), elementwise_affine=False
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
    norm = nn.LayerNorm(
        normalized_shape_for(shape, normalized), elementwise_affine=False
    ).to(device)
    check_against_reference(norm, x, parallel_strategy_2d)


def test_layer_norm_uneven(parallel_strategy: ParallelStrategy, device: torch.device):
    parallel_strategy.shard_dim = 3
    shape = [2, 8, 16, 10]
    x = make_input(shape, device)
    norm = nn.LayerNorm(shape[2:], elementwise_affine=False).to(device)
    check_against_reference(norm, x, parallel_strategy)


@pytest.mark.parametrize("normalized", ["all", "last"])
@pytest.mark.parametrize("bias", [True, False])
def test_layer_norm_affine_not_supported(
    parallel_strategy: ParallelStrategy,
    normalized: str,
    bias: bool,
    device: torch.device,
):
    # Affine parameters are rejected whether or not the normalized dimensions are sharded
    parallel_strategy.shard_dim = 2
    shape = [2, 8, 16, 16]
    norm = nn.LayerNorm(normalized_shape_for(shape, normalized), bias=bias).to(device)
    dcx = DCTensor.distribute(torch.randn(*shape, device=device), parallel_strategy)
    with pytest.raises(NotImplementedError, match="GroupNorm"):
        norm(dcx)


# The input is (2, 8, 16, 16) with the last dimension sharded 4 ways. For the sharded dimension,
# 12 matches neither the global (16) nor local (4) size, and 4 matches the local size, which is
# not accepted since DCTensor cannot tell whether the dimension was permuted. The unsharded
# dimension must also be validated, as must the number of normalized dimensions.
@pytest.mark.parametrize(
    "normalized_shape", [[16, 12], [16, 4], [12, 16], [3, 8, 16, 16, 16]]
)
def test_layer_norm_shape_mismatch(
    parallel_strategy: ParallelStrategy, normalized_shape: list, device: torch.device
):
    parallel_strategy.shard_dim = 3
    norm = nn.LayerNorm(normalized_shape, elementwise_affine=False).to(device)
    dcx = DCTensor.distribute(
        torch.randn(2, 8, 16, 16, device=device), parallel_strategy
    )
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
        nn.LayerNorm([4, 32, 32], elementwise_affine=False),
    ).to(device)
    check_against_reference(model, x, parallel_strategy_2d)
