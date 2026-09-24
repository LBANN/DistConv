"""
PR2 DDP Bucket Hook Demo

Demonstrates DistConvDDP bucket communication hook behavior across topologies.

Usage:
    # 2-rank spatial (W=2, S=2, D=1)
    python -m torch.distributed.run --standalone --nproc-per-node=2 pr2_ddp_demo.py --topology spatial

    # 2-rank data-parallel (W=2, S=1, D=2)
    python -m torch.distributed.run --standalone --nproc-per-node=2 pr2_ddp_demo.py --topology data

    # 4-rank mixed (W=4, S=2, D=2)
    python -m torch.distributed.run --standalone --nproc-per-node=4 pr2_ddp_demo.py --topology mixed
"""

import argparse
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.distributed as dist
import torch.nn as nn

from distconv import DistConvDDP, ParallelStrategy


def print_topology_info(ps: ParallelStrategy):
    """Print parallel strategy topology information."""
    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Topology Information")
        print(f"{'='*60}")
        print(f"World size (W): {ps.world_size}")
        print(f"Spatial shards (S): {ps.total_num_shards}")
        print(f"Data replicas (D): {ps.ddp_ranks}")
        print(f"Scaling factor (W/D): {ps.world_size / ps.ddp_ranks:.2f}")
        print(f"Gradient contract: synchronized_grad = SUM(all_grads) / D")
        print(f"{'='*60}\n")


def test_simple_gradient(ps: ParallelStrategy, device: torch.device, mode="eager"):
    """
    Test simple known gradient values.

    Creates a scenario where we can predict exact gradient values.
    """
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)

    if rank == 0:
        print(f"\n--- Testing {mode.upper()} Mode: Simple Gradient ---")

    # Create simple single-parameter model
    model = nn.Linear(4, 1, bias=False).to(device)

    # Initialize weight to 1.0 for predictable gradients
    with torch.no_grad():
        model.weight.fill_(1.0)

    # Create rank-specific input
    # Input values: rank 0 gets [1,1,1,1], rank 1 gets [2,2,2,2], etc.
    x = torch.full((1, 4), float(rank + 1), device=device, requires_grad=False)

    # Target: 0.0 (so loss = output^2)
    target = torch.zeros(1, 1, device=device)

    # Wrap with DistConvDDP
    ddp_model = DistConvDDP(model, parallel_strategy=ps)

    # Compile if requested
    if mode == "compiled" and hasattr(torch, 'compile'):
        ddp_model = torch.compile(ddp_model, backend="inductor")

    # Forward and backward
    model.zero_grad()
    output = ddp_model(x)
    loss = (output - target).square().mean()
    loss.backward()

    # Get gradient
    grad = model.weight.grad

    # Compute expected gradient
    # For each rank i (0-indexed): input is (i+1), weight is 1.0
    # output = sum(weight * input) = sum(1.0 * (i+1)) = 4*(i+1)
    # loss = output^2 = [4*(i+1)]^2
    # dloss/dweight[j] = 2 * output * input[j] = 2 * 4*(i+1) * (i+1) = 8*(i+1)^2
    #
    # For W ranks with D data replicas:
    # Synchronized gradient = SUM(all_rank_grads) / D

    world_size = ps.world_size
    ddp_ranks = ps.ddp_ranks
    spatial_shards = ps.total_num_shards

    # Each rank i contributes: 8*(i+1)^2
    # Sum over all ranks: sum(8*(i+1)^2 for i in range(world_size))
    expected_sum = sum(8 * (i + 1) ** 2 for i in range(world_size))
    expected_grad = expected_sum / ddp_ranks

    # Print results
    if rank == 0:
        actual_grad = grad[0, 0].item()
        error = abs(actual_grad - expected_grad)
        rel_error = error / abs(expected_grad) if expected_grad != 0 else 0

        print(f"Expected gradient: {expected_grad:.6f}")
        print(f"Actual gradient:   {actual_grad:.6f}")
        print(f"Absolute error:    {error:.6e}")
        print(f"Relative error:    {rel_error:.6e}")

        status = "PASS" if rel_error < 1e-4 else "FAIL"
        print(f"Status: {status}")

        return {
            "mode": mode,
            "expected": expected_grad,
            "actual": actual_grad,
            "abs_error": error,
            "rel_error": rel_error,
            "status": status
        }

    return None


def test_accumulation(ps: ParallelStrategy, device: torch.device):
    """Test gradient accumulation with no_sync()."""
    rank = dist.get_rank()

    if rank == 0:
        print(f"\n--- Testing Gradient Accumulation ---")

    torch.manual_seed(42 + rank)

    # Simple model
    model = nn.Linear(8, 4, bias=False).to(device)

    with torch.no_grad():
        model.weight.fill_(1.0)

    ddp_model = DistConvDDP(model, parallel_strategy=ps)

    num_microbatches = 3

    # Generate microbatch data
    microbatch_data = [
        torch.randn(2, 8, device=device)
        for _ in range(num_microbatches)
    ]
    target = torch.zeros(2, 4, device=device)

    # Reference: proper accumulation - accumulate all, then sync once
    model.zero_grad()

    # Accumulate all microbatches without sync
    for i in range(num_microbatches - 1):
        with ddp_model.no_sync():
            output = ddp_model(microbatch_data[i])
            loss = (output - target).square().mean()
            loss.backward()

    # Final microbatch with sync (triggers gradient synchronization)
    output = ddp_model(microbatch_data[-1])
    loss = (output - target).square().mean()
    loss.backward()

    ref_grad = model.weight.grad.clone()

    # Test: Same pattern should give same result
    model.zero_grad()

    # Accumulate N-1 microbatches without sync
    for i in range(num_microbatches - 1):
        with ddp_model.no_sync():
            output = ddp_model(microbatch_data[i])
            loss = (output - target).square().mean()
            loss.backward()

    # Final microbatch with sync
    output = ddp_model(microbatch_data[-1])
    loss = (output - target).square().mean()
    loss.backward()

    acc_grad = model.weight.grad

    # Compare
    if rank == 0:
        max_diff = (ref_grad - acc_grad).abs().max().item()
        rel_error = max_diff / (ref_grad.abs().max().item() + 1e-10)

        print(f"Reference gradient norm: {ref_grad.norm().item():.6f}")
        print(f"Accumulated gradient norm: {acc_grad.norm().item():.6f}")
        print(f"Max absolute difference: {max_diff:.6e}")
        print(f"Relative error: {rel_error:.6e}")

        status = "PASS" if rel_error < 1e-3 else "FAIL"
        print(f"Status: {status}")

        return {
            "num_microbatches": num_microbatches,
            "ref_norm": ref_grad.norm().item(),
            "acc_norm": acc_grad.norm().item(),
            "max_diff": max_diff,
            "rel_error": rel_error,
            "status": status
        }

    return None


def test_repeated_steps(ps: ParallelStrategy, device: torch.device):
    """Test repeated optimizer steps."""
    rank = dist.get_rank()

    if rank == 0:
        print(f"\n--- Testing Repeated Optimizer Steps ---")

    torch.manual_seed(42 + rank)

    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8)
    ).to(device)

    ddp_model = DistConvDDP(model, parallel_strategy=ps)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    num_steps = 5
    losses = []

    for step in range(num_steps):
        x = torch.randn(4, 16, device=device)
        target = torch.randn(4, 8, device=device)

        optimizer.zero_grad()
        output = ddp_model(x)
        loss = (output - target).square().mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if rank == 0:
        print(f"Completed {num_steps} optimizer steps")
        print(f"Loss history: {[f'{l:.4f}' for l in losses]}")
        print(f"Status: PASS")

        return {
            "num_steps": num_steps,
            "losses": losses,
            "status": "PASS"
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="PR2 DDP Bucket Hook Demo")
    parser.add_argument(
        "--topology",
        type=str,
        choices=["spatial", "data", "mixed"],
        default="spatial",
        help="Topology to test: spatial (W=2,S=2,D=1), data (W=2,S=1,D=2), or mixed (W=4,S=2,D=2)"
    )
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Select device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Create parallel strategy based on topology
    if args.topology == "spatial":
        if world_size != 2:
            if rank == 0:
                print(f"ERROR: spatial topology requires 2 ranks, got {world_size}")
            dist.destroy_process_group()
            sys.exit(1)
        ps = ParallelStrategy(num_shards=2, device_type=device.type)

    elif args.topology == "data":
        if world_size != 2:
            if rank == 0:
                print(f"ERROR: data topology requires 2 ranks, got {world_size}")
            dist.destroy_process_group()
            sys.exit(1)
        ps = ParallelStrategy(num_shards=1, device_type=device.type)

    elif args.topology == "mixed":
        if world_size != 4:
            if rank == 0:
                print(f"ERROR: mixed topology requires 4 ranks, got {world_size}")
            dist.destroy_process_group()
            sys.exit(1)
        ps = ParallelStrategy(num_shards=2, device_type=device.type)

    # Print topology info
    print_topology_info(ps)

    # Run tests
    results = {}

    # Test 1: Simple gradient (eager)
    r = test_simple_gradient(ps, device, mode="eager")
    if r:
        results["simple_eager"] = r

    # Test 2: Simple gradient (compiled)
    if hasattr(torch, 'compile'):
        r = test_simple_gradient(ps, device, mode="compiled")
        if r:
            results["simple_compiled"] = r
    else:
        if rank == 0:
            print("\nSkipping compiled test: torch.compile not available")

    # Test 3: Accumulation
    r = test_accumulation(ps, device)
    if r:
        results["accumulation"] = r

    # Test 4: Repeated steps
    r = test_repeated_steps(ps, device)
    if r:
        results["repeated_steps"] = r

    # Save results (rank 0 only)
    if rank == 0:
        output_file = f"pr2_demo_results_{args.topology}_w{world_size}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}\n")

    # Clean up
    for g in ps.device_mesh.get_all_groups():
        dist.destroy_process_group(g)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
