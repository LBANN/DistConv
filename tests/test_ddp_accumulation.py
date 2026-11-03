"""Check accumulated gradients and optimizer updates against a scalar reference."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from torch import nn

from distconv import DistConvDDP, ParallelStrategy


@pytest.fixture(scope="module", params=[1, 2])
def parallel_strategy(request, device):
    if dist.get_world_size() % request.param:
        pytest.skip("The number of ranks must be divisible by the spatial shards.")
    strategy = ParallelStrategy(num_shards=request.param, device_type=device.type)
    yield strategy
    # A mesh dimension can reuse WORLD. The session fixture owns that group.
    for group in strategy.device_mesh.get_all_groups():
        if group is not dist.group.WORLD:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("use_no_sync", [False, True])
@pytest.mark.parametrize("set_to_none", [False, True])
@pytest.mark.parametrize("gradient_as_bucket_view", [False, True])
def test_ddp_accumulation(
    parallel_strategy, device, use_no_sync, set_to_none, gradient_as_bucket_view
):
    model = nn.Linear(1, 1, bias=False, device=device, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ddp = DistConvDDP(
        model,
        parallel_strategy=parallel_strategy,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Three distinct microbatches per rank: [1, 2, 3], [4, 5, 6], ...
    values = torch.arange(
        1, 3 * dist.get_world_size() + 1, device=device, dtype=torch.float64
    ).reshape(-1, 3)
    local_inputs = values[dist.get_rank()]

    # For y=w*x and loss=y**2, the gradient is 2*w*x**2. Sum over
    # all ranks and microbatches, then average over data replicas only.
    coefficient = 2 * values.square().sum() / parallel_strategy.ddp_ranks
    expected_weight = torch.ones_like(model.weight)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=set_to_none)
        for microbatch, value in enumerate(local_inputs):
            context = ddp.no_sync() if use_no_sync and microbatch < 2 else nullcontext()
            with context:
                ddp(value.reshape(1, 1)).square().sum().backward()

        expected_grad = expected_weight * coefficient
        torch.testing.assert_close(model.weight.grad, expected_grad)
        optimizer.step()
        expected_weight = expected_weight - learning_rate * expected_grad
        torch.testing.assert_close(model.weight, expected_weight)
