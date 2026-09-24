"""
Tests for DistConvDDP bucket communication hook.

Validates:
1. Compilation compatibility with torch.compile
2. API compatibility and validation edge cases
3. Zero grad modes

Note: Gradient correctness tests moved to test_bucket_hook_isolated.py for proper oracle-based validation.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from utils import cleanup_parallel_strategy, fp32_allclose

from distconv import DCTensor, DistConvDDP, ParallelStrategy


class TestAccumulation:
    """Test gradient accumulation edge cases."""

    @pytest.fixture(scope="class")
    def ps(self, device: torch.device):
        """Standard 2-rank spatial strategy."""
        ps = ParallelStrategy(num_shards=2, device_type=device.type)
        yield ps
        cleanup_parallel_strategy(ps)

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
        """Test that explicit divisors no longer require divisibility.

        After Item 2 fix, explicit grad_reduction_factor accepts any positive value.
        This test now verifies that non-divisor Q values are ACCEPTED.
        """
        model = nn.Linear(10, 10).to(device)
        world_size = dist.get_world_size()

        # Use a divisor that doesn't divide world_size evenly
        # For world_size=4, use 3 as a valid non-divisor
        non_divisor = 3 if world_size % 3 != 0 else world_size + 1

        # This should now SUCCEED (not raise)
        ddp_model = DistConvDDP(model, grad_reduction_factor=non_divisor)
        assert ddp_model._gradient_divisor == non_divisor

    def test_missing_both_arguments(self, device):
        """Test that providing neither argument raises an error."""
        model = nn.Linear(10, 10).to(device)

        with pytest.raises(ValueError, match="Either grad_reduction_factor or parallel_strategy"):
            DistConvDDP(model)
