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


@pytest.mark.parametrize(
    "ndims,memory_format", [(2, torch.channels_last), (3, torch.channels_last_3d)]
)
@pytest.mark.parametrize("norm_type", ["group", "instance"])
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
    else:
        norm = getattr(nn, f"InstanceNorm{ndims}d")(8, affine=True)
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


# Precision modes: (parameter dtype, input dtype, autocast)
PRECISION_MODES = {
    # A single dtype
    "float32": (torch.float32, torch.float32, False),
    "bfloat16": (torch.bfloat16, torch.bfloat16, False),
    "float16": (torch.float16, torch.float16, False),
    # Automatic mixed precision
    "amp_bfloat16": (torch.float32, torch.bfloat16, True),
    # Mixed dtypes outside of autocast, with higher and lower precision parameters
    "mixed_bfloat16": (torch.float32, torch.bfloat16, False),
    "float32_bfloat16_params": (torch.bfloat16, torch.float32, False),
}

# Maximum relative error against a float64 reference for correctly computed results
PRECISION_TOLERANCE = {
    torch.float32: 1e-4,
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
}

PRECISION_NORMS = {
    "group": lambda: nn.GroupNorm(2, 4),
    "instance": lambda: nn.InstanceNorm1d(4),
    "instance_affine_tracked": lambda: nn.InstanceNorm1d(
        4, affine=True, track_running_stats=True
    ),
}


# Inputs with a small variance relative to their mean (large rstd), where rounding the saved
# statistics to a reduced-precision dtype corrupts the gradients and running statistics. Results
# are compared against a float64 reference computed from the same rounded inputs and parameters,
# since native PyTorch rounds the statistics in some of these modes.
@pytest.mark.parametrize("mode", PRECISION_MODES)
@pytest.mark.parametrize("norm_name", PRECISION_NORMS)
def test_norm_precision(
    parallel_strategy: ParallelStrategy,
    norm_name: str,
    mode: str,
    device: torch.device,
):
    parallel_strategy.shard_dim = 2
    param_dtype, dtype, use_autocast = PRECISION_MODES[mode]
    norm = PRECISION_NORMS[norm_name]()
    for p in norm.parameters():
        nn.init.uniform_(p, 0.5, 1.5)
    norm = norm.to(device, param_dtype)
    ref_norm = deepcopy(norm).double()
    x = (1 + torch.randint(-2, 3, (2, 4, 16)) / 128).to(device, dtype)
    grad_y = torch.randn(2, 4, 16).to(device, dtype)

    ref_x = x.double().requires_grad_(True)
    ref_y = ref_norm(ref_x)
    ref_y.backward(grad_y.double())

    def autocast():
        return torch.autocast(device.type, dtype=dtype, enabled=use_autocast)

    # The output dtype matches native PyTorch's: the input dtype, except where autocast casts the
    # input (native PyTorch rejects some of the mixed dtypes outside of autocast)
    if use_autocast:
        with autocast(), torch.no_grad():
            native_dtype = deepcopy(norm)(x).dtype
    else:
        native_dtype = dtype

    module = (
        DistConvDDP(norm, parallel_strategy=parallel_strategy) if norm.affine else norm
    )
    dcx = DCTensor.distribute(x, parallel_strategy).requires_grad_(True)
    with autocast():
        dcy = module(dcx)
    dcy.backward(DCTensor.distribute(grad_y.to(dcy.dtype), parallel_strategy))

    def check(actual, expected, tolerance_dtype, scale=1):
        error = relative_error(actual.cpu(), expected.detach().cpu())
        assert error < PRECISION_TOLERANCE[tolerance_dtype] * scale

    def check_shard(actual, expected):
        check(actual, local_shard(expected, parallel_strategy), dtype)

    assert dcy.dtype == native_dtype and dcx.grad.dtype == dtype
    check_shard(dcy._tensor, ref_y)
    check_shard(dcx.grad._tensor, ref_x.grad)
    # Each shard's partial parameter gradient is rounded to the parameter dtype before the
    # cross-shard sum, so the tolerance scales with the number of shards
    for p, ref_p in zip(norm.parameters(), ref_norm.parameters()):
        assert p.grad.dtype == param_dtype
        check(p.grad, ref_p.grad, param_dtype, parallel_strategy.total_num_shards)
    for buf, ref_buf in zip(norm.buffers(), ref_norm.buffers()):
        if buf.is_floating_point():
            check(buf, ref_buf, param_dtype)


# A bfloat16 input whose mean is not representable in bfloat16, with float32 running statistics
# and no affine parameters, unsharded and sharded down to one element per shard. Rounding the
# statistics to bfloat16 would corrupt the output, the gradients and the running mean.
@pytest.mark.parametrize("use_autocast", [False, True])
@pytest.mark.parametrize(
    "strategy_name", ["parallel_strategy_unsharded", "parallel_strategy"]
)
def test_instance_norm_unrepresentable_mean(
    request: pytest.FixtureRequest,
    strategy_name: str,
    use_autocast: bool,
    device: torch.device,
):
    parallel_strategy = request.getfixturevalue(strategy_name)
    parallel_strategy.shard_dim = 2
    norm = nn.InstanceNorm1d(1, track_running_stats=True).to(device)
    ref_norm = deepcopy(norm).double()
    x = torch.tensor(
        [[[1.0, 1.0 + 1 / 128, 1.0 + 1 / 128, 1.0]]],
        dtype=torch.bfloat16,
        device=device,
    )
    grad_y = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=x.dtype, device=device)

    ref_x = x.double().requires_grad_(True)
    ref_y = ref_norm(ref_x)
    ref_y.backward(grad_y.double())

    dcx = DCTensor.distribute(x, parallel_strategy).requires_grad_(True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
        dcy = norm(dcx)
    dcy.backward(DCTensor.distribute(grad_y, parallel_strategy))

    torch.testing.assert_close(
        dcy._tensor, local_shard(ref_y.to(x.dtype), parallel_strategy)
    )
    torch.testing.assert_close(
        dcx.grad._tensor, local_shard(ref_x.grad.to(x.dtype), parallel_strategy)
    )
    torch.testing.assert_close(
        norm.running_mean, ref_norm.running_mean.to(norm.running_mean.dtype)
    )


# The saved statistics are float32 for every input dtype
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_group_norm_forward_stats_dtype(
    affine: bool, dtype: torch.dtype, device: torch.device
):
    x = torch.randn(2, 4, 8, device=device, dtype=dtype)
    weight = torch.ones(4, device=device, dtype=dtype) if affine else None
    bias = torch.zeros(4, device=device, dtype=dtype) if affine else None
    _, mean, rstd = group_norm_forward(x, weight, bias, 2, 1e-5, [])
    assert mean.dtype == rstd.dtype == torch.float32


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


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("normalized", ["channels", "spatial"])
def test_layer_norm_unsharded(
    parallel_strategy: ParallelStrategy,
    normalized: str,
    affine: bool,
    device: torch.device,
):
    # Layer norm over unsharded dimensions is local to each shard. Shard along the first
    # spatial dimension, and normalize over either the trailing spatial dimension or the channels
    # of a channels-last (permuted) tensor.
    parallel_strategy.shard_dim = 2
    shape = [2, 8, 16, 12]
    x = make_input(shape, device)
    if normalized == "channels":
        norm = PermutedLayerNorm(shape[1], affine)
    else:
        norm = nn.LayerNorm(shape[-1:], elementwise_affine=affine)
    check_against_reference(norm.to(device), x, parallel_strategy)


class PermutedLayerNorm(nn.Module):
    """Layer norm over channels of an NCHW tensor, as in ConvNeXt."""

    def __init__(self, num_channels: int, affine: bool):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, elementwise_affine=affine)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


@pytest.mark.parametrize("normalized", ["all", "spatial", "last"])
@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_sharded_not_supported(
    parallel_strategy: ParallelStrategy,
    normalized: str,
    affine: bool,
    device: torch.device,
):
    parallel_strategy.shard_dim = 3
    shape = [2, 8, 16, 16]
    if normalized == "all":
        normalized_shape = shape[1:]
    elif normalized == "spatial":
        normalized_shape = shape[2:]
    else:
        normalized_shape = shape[-1:]
    norm = nn.LayerNorm(normalized_shape, elementwise_affine=affine).to(device)
    dcx = DCTensor.distribute(torch.randn(*shape, device=device), parallel_strategy)
    with pytest.raises(NotImplementedError, match="GroupNorm.*open an issue"):
        norm(dcx)


def test_layer_norm_stale_shard_dim(
    parallel_strategy: ParallelStrategy, device: torch.device
):
    # After a permutation, the sharded dimension (global size 16, local size 4) is the last
    # dimension, but DCTensor still records the shard dimension as 2. Layer norm over it must not
    # silently normalize the local shard; the native op's shape validation rejects it.
    parallel_strategy.shard_dim = 2
    norm = nn.LayerNorm(16, elementwise_affine=False).to(device)
    dcx = DCTensor.distribute(
        torch.randn(2, 8, 16, 12, device=device), parallel_strategy
    )
    with pytest.raises(RuntimeError, match="normalized_shape"):
        norm(dcx.permute(0, 3, 1, 2))


def test_conv_group_norm_model(
    parallel_strategy_2d: ParallelStrategy, device: torch.device
):
    parallel_strategy_2d.shard_dim = (2, 3)
    x = make_input([2, 4, 32, 32], device)
    model = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.GroupNorm(4, 8),
        nn.ReLU(),
        # Instance norm cancels a per-channel bias, whose gradient is then pure rounding error
        nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(8, affine=True),
        nn.Conv2d(8, 4, kernel_size=1),
    ).to(device)
    check_against_reference(model, x, parallel_strategy_2d)
