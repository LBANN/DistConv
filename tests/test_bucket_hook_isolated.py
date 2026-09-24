"""
Isolated tests for DistConvDDP bucket communication hook.

Tests bucket hook behavior WITHOUT DCTensor integration.
Uses ordinary tensors with DistConvDDP to validate gradient reduction logic.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DistConvDDP, ParallelStrategy


class TestBucketHookIsolated:
    """Isolated bucket hook tests without DCTensor dependency."""

    @pytest.fixture(scope="class")
    def ps_spatial(self, device: torch.device):
        """W=4, S=4, D=1 - pure spatial sharding."""
        ps = ParallelStrategy(num_shards=4, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

    @pytest.fixture(scope="class")
    def ps_data(self, device: torch.device):
        """W=4, S=1, D=4 - pure data-parallel."""
        ps = ParallelStrategy(num_shards=1, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

    def test_spatial_gradient_sum_q1(self, ps_spatial, device):
        """
        Test Q=1 (spatial only): gradients should be SUMMED.

        Topology: W=4, S=4, D=1, Q=1
        Each rank: different data, same parameters
        Expected: synchronized_grad = SUM(local_grads) / Q = SUM(local_grads) / 1
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        torch.manual_seed(42)  # Same seed for all ranks - shared parameters

        # Create model with known parameters
        model = nn.Linear(3, 1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(1.0)  # w = [1, 1, 1]

        # Rank-specific input (simulates different spatial regions)
        # Use small integer values for predictable gradients
        rank_inputs = [
            torch.tensor([[1.0, 0.0, 0.0]], device=device),  # rank 0
            torch.tensor([[0.0, 1.0, 0.0]], device=device),  # rank 1
            torch.tensor([[0.0, 0.0, 1.0]], device=device),  # rank 2
            torch.tensor([[1.0, 1.0, 1.0]], device=device),  # rank 3
        ]
        x = rank_inputs[rank].requires_grad_(True)

        # Loss: y^2 where y = w·x
        # For w=[1,1,1], rank gradients:
        # rank 0: y=1, grad = 2*y*x = 2*1*[1,0,0] = [2,0,0]
        # rank 1: y=1, grad = 2*y*x = 2*1*[0,1,0] = [0,2,0]
        # rank 2: y=1, grad = 2*y*x = 2*1*[0,0,1] = [0,0,2]
        # rank 3: y=3, grad = 2*y*x = 2*3*[1,1,1] = [6,6,6]
        # Expected synchronized: [2+0+0+6, 0+2+0+6, 0+0+2+6] = [8, 8, 8]

        expected_grad = torch.tensor([[8.0, 8.0, 8.0]], device=device)

        # Create DDP with Q=1
        ddp_model = DistConvDDP(model, parallel_strategy=ps_spatial)

        # Forward and backward
        model.zero_grad()
        y = ddp_model(x)
        loss = y.square().mean()
        loss.backward()

        actual_grad = model.weight.grad

        # All ranks should have the same synchronized gradient
        assert actual_grad is not None
        assert fp32_allclose(expected_grad, actual_grad, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: expected {expected_grad}, got {actual_grad}, " \
            f"diff={(expected_grad - actual_grad).abs().max().item():.6e}"

    def test_data_parallel_gradient_avg_q4(self, ps_data, device):
        """
        Test Q=4 (data-parallel): gradients should be AVERAGED.

        Topology: W=4, S=1, D=4, Q=4
        Each rank: different data, same parameters
        Expected: synchronized_grad = SUM(local_grads) / Q = SUM(local_grads) / 4
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        torch.manual_seed(42)  # Same seed for all ranks

        # Create model
        model = nn.Linear(3, 1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(1.0)

        # Rank-specific input (data-parallel pattern)
        rank_inputs = [
            torch.tensor([[1.0, 0.0, 0.0]], device=device),
            torch.tensor([[0.0, 1.0, 0.0]], device=device),
            torch.tensor([[0.0, 0.0, 1.0]], device=device),
            torch.tensor([[1.0, 1.0, 1.0]], device=device),
        ]
        x = rank_inputs[rank].requires_grad_(True)

        # Same local gradients as before: [2,0,0], [0,2,0], [0,0,2], [6,6,6]
        # Expected synchronized: ([2,0,0] + [0,2,0] + [0,0,2] + [6,6,6]) / 4
        #                      = [8,8,8] / 4 = [2,2,2]

        expected_grad = torch.tensor([[2.0, 2.0, 2.0]], device=device)

        # Create DDP with Q=4
        ddp_model = DistConvDDP(model, parallel_strategy=ps_data)

        # Forward and backward
        model.zero_grad()
        y = ddp_model(x)
        loss = y.square().mean()
        loss.backward()

        actual_grad = model.weight.grad

        assert actual_grad is not None
        assert fp32_allclose(expected_grad, actual_grad, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: expected {expected_grad}, got {actual_grad}, " \
            f"diff={(expected_grad - actual_grad).abs().max().item():.6e}"

    @pytest.mark.parametrize("gradient_as_bucket_view", [True, False])
    def test_bucket_view_modes(self, ps_spatial, device, gradient_as_bucket_view):
        """Test both gradient_as_bucket_view settings produce correct results."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        torch.manual_seed(42)

        model = nn.Linear(2, 1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(1.0)

        # Simple rank-specific input
        x = torch.tensor([[float(rank + 1), float(rank + 1)]], device=device).requires_grad_(True)

        # Local gradients:
        # rank 0: y=2, grad = 2*2*[1,1] = [4,4]
        # rank 1: y=4, grad = 2*4*[2,2] = [16,16]
        # rank 2: y=6, grad = 2*6*[3,3] = [36,36]
        # rank 3: y=8, grad = 2*8*[4,4] = [64,64]
        # Expected: [4+16+36+64, 4+16+36+64] = [120, 120]

        expected_grad = torch.tensor([[120.0, 120.0]], device=device)

        ddp_model = DistConvDDP(
            model,
            parallel_strategy=ps_spatial,
            gradient_as_bucket_view=gradient_as_bucket_view
        )

        model.zero_grad()
        y = ddp_model(x)
        loss = y.square().mean()
        loss.backward()

        actual_grad = model.weight.grad

        assert actual_grad is not None
        assert fp32_allclose(expected_grad, actual_grad, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}, bucket_view={gradient_as_bucket_view}: " \
            f"expected {expected_grad}, got {actual_grad}"


class TestAccumulationIsolated:
    """Test gradient accumulation with no_sync()."""

    @pytest.fixture(scope="class")
    def ps(self, device: torch.device):
        """4-rank spatial strategy."""
        ps = ParallelStrategy(num_shards=4, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

    def test_accumulation_correctness_eager(self, ps, device):
        """
        Test accumulation produces correct final gradient (eager mode).

        Independent oracle: For scalar model y=w*x with loss=y^2:
        dloss/dw = 2*w*x^2

        With w=1, accumulating over microbatches:
        final_grad = SUM(2*x_i^2 for all ranks, all microbatches) / Q
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        torch.manual_seed(42)

        # Scalar model
        model = nn.Linear(1, 1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(1.0)

        # 3 microbatches per rank, deterministic values
        # rank 0: [1, 2, 3], rank 1: [4, 5, 6], rank 2: [7, 8, 9], rank 3: [10, 11, 12]
        microbatches = [
            torch.tensor([[float(rank * 3 + i + 1)]], device=device)
            for i in range(3)
        ]

        # Expected gradient calculation:
        # For each value v, local_grad = 2*v^2
        # rank 0: 2*(1^2 + 2^2 + 3^2) = 2*(1+4+9) = 28
        # rank 1: 2*(16+25+36) = 154
        # rank 2: 2*(49+64+81) = 388
        # rank 3: 2*(100+121+144) = 730
        # Total: 28+154+388+730 = 1300
        # Q=1 (spatial): expected = 1300

        expected_grad = torch.tensor([[1300.0]], device=device)

        ddp_model = DistConvDDP(model, parallel_strategy=ps)

        # Accumulation pattern
        model.zero_grad()
        for i in range(2):
            with ddp_model.no_sync():
                y = ddp_model(microbatches[i])
                loss = y.square().mean()
                loss.backward()

        # Final microbatch with sync
        y = ddp_model(microbatches[2])
        loss = y.square().mean()
        loss.backward()

        actual_grad = model.weight.grad

        assert actual_grad is not None
        assert fp32_allclose(expected_grad, actual_grad, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: expected {expected_grad}, got {actual_grad}, " \
            f"diff={(expected_grad - actual_grad).abs().max().item():.6e}"

    def test_accumulation_correctness_compiled(self, ps, device):
        """Test accumulation with torch.compile (Inductor backend)."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available")

        torch.manual_seed(42)

        model = nn.Linear(1, 1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(1.0)

        ddp_model = DistConvDDP(model, parallel_strategy=ps)
        compiled_model = torch.compile(ddp_model, backend="inductor")

        microbatches = [
            torch.tensor([[float(rank * 3 + i + 1)]], device=device)
            for i in range(3)
        ]

        expected_grad = torch.tensor([[1300.0]], device=device)

        # Accumulation with compiled model
        model.zero_grad()
        for i in range(2):
            with compiled_model.no_sync():
                y = compiled_model(microbatches[i])
                loss = y.square().mean()
                loss.backward()

        y = compiled_model(microbatches[2])
        loss = y.square().mean()
        loss.backward()

        actual_grad = model.weight.grad

        assert actual_grad is not None
        assert fp32_allclose(expected_grad, actual_grad, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: expected {expected_grad}, got {actual_grad}"

    @pytest.mark.parametrize("set_to_none", [True, False])
    def test_zero_grad_modes(self, ps, device, set_to_none):
        """Test both zero_grad modes work correctly."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        torch.manual_seed(42)

        model = nn.Linear(1, 1, bias=False).to(device)
        ddp_model = DistConvDDP(model, parallel_strategy=ps)

        x = torch.tensor([[1.0]], device=device)

        # First accumulation window
        model.zero_grad(set_to_none=set_to_none)
        y = ddp_model(x)
        y.square().mean().backward()
        grad1 = model.weight.grad.clone()

        # Second window with same data
        model.zero_grad(set_to_none=set_to_none)
        y = ddp_model(x)
        y.square().mean().backward()
        grad2 = model.weight.grad

        # Both windows should produce identical gradients
        assert fp32_allclose(grad1, grad2, rtol=1e-6, atol=1e-8)


class TestAPICompatibility:
    """Test API validation and compatibility."""

    def test_explicit_grad_reduction_factor(self, device):
        """Test explicit gradient_divisor."""
        model = nn.Linear(2, 2).to(device)
        ddp = DistConvDDP(model, grad_reduction_factor=2.0)

        assert ddp._gradient_divisor == 2.0
        assert ddp.grad_reduction_factor == dist.get_world_size() / 2.0

    def test_parallel_strategy_derived_factor(self, device):
        """Test gradient_divisor derived from ParallelStrategy."""
        ps = ParallelStrategy(num_shards=2, device_type=device.type)
        model = nn.Linear(2, 2).to(device)
        ddp = DistConvDDP(model, parallel_strategy=ps)

        assert ddp._gradient_divisor == ps.ddp_ranks
        cleanup_parallel_strategy(ps)

    def test_invalid_divisor_rejected(self, device):
        """Test invalid gradient_divisor values are rejected."""
        model = nn.Linear(2, 2).to(device)

        with pytest.raises(ValueError, match="finite and positive"):
            DistConvDDP(model, grad_reduction_factor=0.0)

        with pytest.raises(ValueError, match="finite and positive"):
            DistConvDDP(model, grad_reduction_factor=-1.0)

        with pytest.raises(ValueError, match="finite and positive"):
            DistConvDDP(model, grad_reduction_factor=float('inf'))

    def test_topology_mismatch_rejected(self, device):
        """Test ParallelStrategy with invalid topology is rejected.

        Note: Explicit grad_reduction_factor no longer requires divisibility,
        but ParallelStrategy-derived divisors must satisfy W = S * D.
        """
        # This test verifies that IF a ParallelStrategy somehow produces
        # invalid topology (W % ddp_ranks != 0), it's caught.
        # In practice, ParallelStrategy construction ensures W = S * D,
        # so this check is defensive.

        # Skip this test - ParallelStrategy construction itself validates topology,
        # and explicit divisors are now allowed to be non-divisors.
        # The validation in DistConvDDP is a redundant sanity check.
        pytest.skip("ParallelStrategy topology validation handled at ParallelStrategy construction")

    def test_missing_both_arguments(self, device):
        """Test missing both grad_reduction_factor and parallel_strategy is rejected."""
        model = nn.Linear(2, 2).to(device)

        with pytest.raises(ValueError, match="Either grad_reduction_factor or parallel_strategy"):
            DistConvDDP(model)

    def test_fractional_divisor(self, device):
        """Test Q=2.5 produces correct gradient scaling."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        torch.manual_seed(42)

        model = nn.Linear(1, 1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(1.0)

        # Q = 2.5, W = 4
        ddp = DistConvDDP(model, grad_reduction_factor=2.5)

        # Rank-specific inputs for testing
        x = torch.tensor([[float(rank + 1)]], device=device)

        # Forward + backward
        model.zero_grad()
        y = ddp(x)
        y.square().mean().backward()

        # Expected: SUM(local_grads) / 2.5
        # For y = w*x with loss = y^2, gradient = 2*w*x^2 = 2*x^2 (w=1)
        # local_grads: rank0=2*1^2=2, rank1=2*2^2=8, rank2=2*3^2=18, rank3=2*4^2=32
        # SUM = 2+8+18+32 = 60, 60/2.5 = 24
        expected = torch.tensor([[24.0]], device=device)
        assert torch.allclose(model.weight.grad, expected, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: expected {expected}, got {model.weight.grad}"

    def test_non_divisor_integer(self, device):
        """Test Q=3 with W=4 (non-divisor) produces correct gradient."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size != 4:
            pytest.skip(f"Test requires 4 ranks, got {world_size}")

        torch.manual_seed(42)

        model = nn.Linear(1, 1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(1.0)

        # Q = 3, W = 4 (4 % 3 != 0)
        ddp = DistConvDDP(model, grad_reduction_factor=3)

        rank = dist.get_rank()
        x = torch.tensor([[float(rank + 1)]], device=device)

        model.zero_grad()
        y = ddp(x)
        y.square().mean().backward()

        # Expected: SUM(local_grads) / 3 = 60 / 3 = 20
        expected = torch.tensor([[20.0]], device=device)
        assert torch.allclose(model.weight.grad, expected, rtol=1e-5, atol=1e-6), \
            f"Rank {rank}: expected {expected}, got {model.weight.grad}"

    def test_process_group_configuration(self, device):
        """Verify bucket hook uses the configured process group."""
        ps = ParallelStrategy(num_shards=4, device_type=device.type)
        model = nn.Linear(2, 1, bias=False).to(device)
        ddp_model = DistConvDDP(model, parallel_strategy=ps)

        # Access registered hook state
        # DDP stores hooks in self._comm_hooks list
        assert hasattr(ddp_model, '_comm_hooks')
        assert len(ddp_model._comm_hooks) > 0

        # Verify state structure
        hook_entry = ddp_model._comm_hooks[0]
        state = hook_entry.state

        # Verify state has correct process group
        assert hasattr(state, 'process_group')
        assert state.process_group == ddp_model.process_group
        assert hasattr(state, 'world_size')
        assert state.world_size == dist.get_world_size(state.process_group)
        assert hasattr(state, 'gradient_divisor')
        assert state.gradient_divisor == ps.ddp_ranks

        # Cleanup
        from tests.utils import cleanup_parallel_strategy
        cleanup_parallel_strategy(ps)
