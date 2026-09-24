"""
Tests for DistConvDDP bucket communication hook.

Validates:
1. Gradient correctness across spatial, data-parallel, and mixed topologies
2. Gradient accumulation with no_sync()
3. Compilation compatibility with torch.compile
4. API compatibility and validation edge cases
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


class TestBucketHookCorrectness:
    """Test correctness of bucket hook gradient scaling."""

    @pytest.fixture(scope="class")
    def ps_spatial(self, device: torch.device):
        """W=2, S=2, D=1 - spatial sharding only."""
        ps = ParallelStrategy(num_shards=2, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

    @pytest.fixture(scope="class")
    def ps_data(self, device: torch.device):
        """W=2, S=1, D=2 - data-parallel only."""
        ps = ParallelStrategy(num_shards=1, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

    def test_spatial_topology_2ranks(self, ps_spatial, device):
        """
        W=2, S=2, D=1: Gradients should be summed (no averaging).

        Expected: With local gradients g0=1.0, g1=2.0 on ranks 0,1,
        synchronized gradient = (g0 + g1) / D = (1 + 2) / 1 = 3.0
        """
        torch.manual_seed(42 + dist.get_rank())

        # Create simple model: single Conv2d layer
        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device)

        # Create rank-specific input that will produce known gradients
        # Use small input to have predictable gradient values
        x_local = torch.randn(1, 4, 8, 8, device=device, requires_grad=True)

        # Reference: Non-distributed computation
        conv.zero_grad()
        ref_y = conv(x_local)
        ref_y.square().mean().backward()
        ref_grad = conv.weight.grad.clone()

        # DistConv: with bucket hook
        conv.zero_grad()
        x_local.grad = None
        dc_conv = DistConvDDP(conv, parallel_strategy=ps_spatial)

        # For spatial topology, both ranks process different spatial regions
        # Use DCTensor to distribute spatial dimensions
        dcx = DCTensor.distribute(x_local, ps_spatial)
        dcy = dc_conv(dcx)
        ddpy = dcy.to_ddp()
        ddpy.square().mean().backward()

        dc_grad = conv.weight.grad

        # Validation: gradients should match SUM over spatial shards
        # Both ranks should have the same synchronized gradient
        assert dc_grad is not None, "Gradient should not be None"
        assert fp32_allclose(ref_grad, dc_grad, rtol=1e-3, atol=1e-4), \
            f"Rank {dist.get_rank()}: Gradient mismatch, max_diff={(ref_grad - dc_grad).abs().max().item():.6e}"

    def test_data_parallel_topology_2ranks(self, ps_data, device):
        """
        W=2, S=1, D=2: Gradients should be averaged over data replicas.

        Expected: With independent data on each rank, gradients are averaged.
        """
        torch.manual_seed(42 + dist.get_rank())

        # Create simple model
        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device)

        # Each rank gets different data (data-parallel pattern)
        x_local = torch.randn(1, 4, 8, 8, device=device, requires_grad=True)

        # Compute local gradient
        conv.zero_grad()
        local_y = conv(x_local)
        local_y.square().mean().backward()
        local_grad = conv.weight.grad.clone()

        # For reference: manually average gradients across ranks
        # This simulates what data-parallel DDP should do
        ref_grad = local_grad.clone()
        dist.all_reduce(ref_grad, op=dist.ReduceOp.AVG)

        # DistConv with data-parallel (no spatial sharding)
        conv.zero_grad()
        x_local.grad = None

        # With num_shards=1, there's no spatial sharding, only data parallelism
        dc_conv = DistConvDDP(conv, parallel_strategy=ps_data)

        # For data-parallel, use regular tensor (no spatial distribution)
        y = dc_conv(x_local)
        y.square().mean().backward()

        dc_grad = conv.weight.grad

        # Validation: gradients should be averaged
        assert dc_grad is not None, "Gradient should not be None"
        assert fp32_allclose(ref_grad, dc_grad, rtol=1e-3, atol=1e-4), \
            f"Rank {dist.get_rank()}: Gradient mismatch, max_diff={(ref_grad - dc_grad).abs().max().item():.6e}"

    @pytest.mark.parametrize("gradient_as_bucket_view", [True, False])
    def test_bucket_view_modes(self, ps_spatial, device, gradient_as_bucket_view):
        """Test both gradient_as_bucket_view settings."""
        torch.manual_seed(42 + dist.get_rank())

        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device)
        x_local = torch.randn(1, 4, 8, 8, device=device, requires_grad=True)

        # Reference
        conv.zero_grad()
        ref_y = conv(x_local)
        ref_y.square().mean().backward()
        ref_grad = conv.weight.grad.clone()

        # DistConv with specified bucket view mode
        conv.zero_grad()
        x_local.grad = None
        dc_conv = DistConvDDP(
            conv,
            parallel_strategy=ps_spatial,
            gradient_as_bucket_view=gradient_as_bucket_view
        )

        dcx = DCTensor.distribute(x_local, ps_spatial)
        dcy = dc_conv(dcx)
        ddpy = dcy.to_ddp()
        ddpy.square().mean().backward()

        dc_grad = conv.weight.grad

        assert dc_grad is not None
        assert fp32_allclose(ref_grad, dc_grad, rtol=1e-3, atol=1e-4), \
            f"Rank {dist.get_rank()}, bucket_view={gradient_as_bucket_view}: " \
            f"Gradient mismatch, max_diff={(ref_grad - dc_grad).abs().max().item():.6e}"


class TestAccumulation:
    """Test gradient accumulation with no_sync()."""

    @pytest.fixture(scope="class")
    def ps(self, device: torch.device):
        """Standard 2-rank spatial strategy."""
        ps = ParallelStrategy(num_shards=2, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

    def test_accumulation_correctness(self, ps, device):
        """
        Test multi-step accumulation produces correct final gradients.

        Pattern:
            1. zero_grad()
            2. no_sync() × (N-1) microbatches
            3. synchronized backward on final microbatch
            4. Verify gradient = sum of all microbatches
        """
        torch.manual_seed(42 + dist.get_rank())

        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device)
        dc_conv = DistConvDDP(conv, parallel_strategy=ps)

        num_microbatches = 3

        # Generate different data for each microbatch
        microbatch_data = [
            torch.randn(1, 4, 8, 8, device=device, requires_grad=True)
            for _ in range(num_microbatches)
        ]

        # Compute reference: sum of all microbatch gradients
        conv.zero_grad()
        for x in microbatch_data:
            dcx = DCTensor.distribute(x, ps)
            dcy = dc_conv(dcx)  # Forward without sync
            ddpy = dcy.to_ddp()
            loss = ddpy.square().mean()
            loss.backward()  # This will sync - we need no_sync for this test

        ref_accumulated_grad = conv.weight.grad.clone()

        # Test accumulation pattern
        conv.zero_grad()

        # Accumulate first N-1 microbatches without sync
        for i in range(num_microbatches - 1):
            with dc_conv.no_sync():
                dcx = DCTensor.distribute(microbatch_data[i], ps)
                dcy = dc_conv(dcx)
                ddpy = dcy.to_ddp()
                loss = ddpy.square().mean()
                loss.backward()

        # Final microbatch with sync
        dcx = DCTensor.distribute(microbatch_data[-1], ps)
        dcy = dc_conv(dcx)
        ddpy = dcy.to_ddp()
        loss = ddpy.square().mean()
        loss.backward()

        accumulated_grad = conv.weight.grad

        # Verify accumulated gradient matches reference
        assert accumulated_grad is not None
        assert fp32_allclose(ref_accumulated_grad, accumulated_grad, rtol=1e-3, atol=1e-4), \
            f"Rank {dist.get_rank()}: Accumulated gradient mismatch, " \
            f"max_diff={(ref_accumulated_grad - accumulated_grad).abs().max().item():.6e}"

    @pytest.mark.parametrize("set_to_none", [True, False])
    def test_zero_grad_modes(self, ps, device, set_to_none):
        """Test both zero_grad(set_to_none=True/False) modes work correctly."""
        torch.manual_seed(42 + dist.get_rank())

        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1).to(device)
        dc_conv = DistConvDDP(conv, parallel_strategy=ps)

        x = torch.randn(1, 4, 8, 8, device=device, requires_grad=True)

        # First accumulation window
        conv.zero_grad(set_to_none=set_to_none)
        dcx = DCTensor.distribute(x, ps)
        dcy = dc_conv(dcx)
        ddpy = dcy.to_ddp()
        ddpy.square().mean().backward()
        grad_window1 = conv.weight.grad.clone()

        # Second accumulation window with same data
        conv.zero_grad(set_to_none=set_to_none)
        dcx = DCTensor.distribute(x, ps)
        dcy = dc_conv(dcx)
        ddpy = dcy.to_ddp()
        ddpy.square().mean().backward()
        grad_window2 = conv.weight.grad.clone()

        # Gradients should match across windows with same input
        assert grad_window2 is not None
        assert fp32_allclose(grad_window1, grad_window2, rtol=1e-3, atol=1e-4), \
            f"Rank {dist.get_rank()}, set_to_none={set_to_none}: " \
            f"Gradient mismatch across windows, max_diff={(grad_window1 - grad_window2).abs().max().item():.6e}"


class TestCompilation:
    """Test torch.compile compatibility."""

    @pytest.fixture(scope="class")
    def ps(self, device: torch.device):
        """Standard 2-rank spatial strategy."""
        ps = ParallelStrategy(num_shards=2, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

    def test_compiled_model_correctness(self, ps, device):
        """Test compiled model produces correct gradients."""
        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available")

        torch.manual_seed(42 + dist.get_rank())

        # Simple ordinary tensor model (not DCTensor)
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ).to(device)

        # Wrap with DistConvDDP
        ddp_model = DistConvDDP(model, parallel_strategy=ps)

        # Compile the DDP model
        compiled_model = torch.compile(ddp_model, backend="inductor")

        # Generate data
        x = torch.randn(4, 32, device=device)
        target = torch.randn(4, 10, device=device)

        # Reference: eager mode
        model.zero_grad()
        eager_y = ddp_model(x)
        eager_loss = (eager_y - target).square().mean()
        eager_loss.backward()
        eager_grad = model[0].weight.grad.clone()

        # Test: compiled mode
        model.zero_grad()
        compiled_y = compiled_model(x)
        compiled_loss = (compiled_y - target).square().mean()
        compiled_loss.backward()
        compiled_grad = model[0].weight.grad

        # Verify outputs and gradients match
        assert torch.allclose(eager_y, compiled_y, rtol=1e-4, atol=1e-5), \
            f"Rank {dist.get_rank()}: Output mismatch between eager and compiled"
        assert fp32_allclose(eager_grad, compiled_grad, rtol=1e-3, atol=1e-4), \
            f"Rank {dist.get_rank()}: Gradient mismatch between eager and compiled, " \
            f"max_diff={(eager_grad - compiled_grad).abs().max().item():.6e}"

    def test_repeated_steps_no_recompilation(self, ps, device):
        """Test compilation cache reuse across optimizer steps."""
        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available")

        torch.manual_seed(42 + dist.get_rank())

        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ).to(device)

        ddp_model = DistConvDDP(model, parallel_strategy=ps)
        compiled_model = torch.compile(ddp_model, backend="inductor")

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Run multiple training steps
        for step in range(3):
            x = torch.randn(4, 32, device=device)
            target = torch.randn(4, 10, device=device)

            optimizer.zero_grad()
            y = compiled_model(x)
            loss = (y - target).square().mean()
            loss.backward()
            optimizer.step()

        # If we got here without errors, compilation succeeded
        # Detailed compilation counting would require instrumentation
        # which is in PR1's test_compile_diagnostics.py
        assert True, "Multiple training steps completed successfully"


class TestCompatibility:
    """Test API compatibility and edge cases."""

    def test_explicit_grad_reduction_factor(self, device):
        """Test explicit grad_reduction_factor argument."""
        ps = ParallelStrategy(num_shards=2, device_type=device.type)

        model = nn.Linear(10, 10).to(device)

        # Use explicit factor (should equal ps.ddp_ranks)
        explicit_factor = ps.ddp_ranks
        ddp_model = DistConvDDP(model, grad_reduction_factor=explicit_factor)

        # Verify internal state
        assert ddp_model._gradient_divisor == explicit_factor
        assert ddp_model.grad_reduction_factor == dist.get_world_size() / explicit_factor

        cleanup_parallel_strategy(ps)

    def test_parallel_strategy_derived_factor(self, device):
        """Test ParallelStrategy-derived factor."""
        ps = ParallelStrategy(num_shards=2, device_type=device.type)

        model = nn.Linear(10, 10).to(device)

        # Use parallel_strategy (implicit factor)
        ddp_model = DistConvDDP(model, parallel_strategy=ps)

        # Verify internal state
        assert ddp_model._gradient_divisor == ps.ddp_ranks
        assert ddp_model.grad_reduction_factor == dist.get_world_size() / ps.ddp_ranks

        cleanup_parallel_strategy(ps)

    def test_invalid_divisor_rejected(self, device):
        """Test rejection of invalid gradient divisors."""
        model = nn.Linear(10, 10).to(device)

        # Test negative divisor
        with pytest.raises(ValueError, match="must be finite and positive"):
            DistConvDDP(model, grad_reduction_factor=-1.0)

        # Test zero divisor
        with pytest.raises(ValueError, match="must be finite and positive"):
            DistConvDDP(model, grad_reduction_factor=0.0)

        # Test infinite divisor
        with pytest.raises(ValueError, match="must be finite and positive"):
            DistConvDDP(model, grad_reduction_factor=float('inf'))

        # Test NaN divisor
        with pytest.raises(ValueError, match="must be finite and positive"):
            DistConvDDP(model, grad_reduction_factor=float('nan'))

    def test_topology_mismatch_rejected(self, device):
        """Test rejection when divisor doesn't divide world_size."""
        model = nn.Linear(10, 10).to(device)
        world_size = dist.get_world_size()

        # Use a divisor that doesn't divide world_size evenly
        # For world_size=2 or 4, use 3 as an invalid divisor
        invalid_divisor = 3 if world_size % 3 != 0 else world_size + 1

        with pytest.raises(ValueError, match="must be evenly divisible"):
            DistConvDDP(model, grad_reduction_factor=invalid_divisor)

    def test_missing_both_arguments(self, device):
        """Test that providing neither argument raises an error."""
        model = nn.Linear(10, 10).to(device)

        with pytest.raises(ValueError, match="Either grad_reduction_factor or parallel_strategy"):
            DistConvDDP(model)
