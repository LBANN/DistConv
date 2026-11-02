"""Reproduce incorrect gradient accumulation with two spatial shards.

Run from the repository root (CPU only; no torch.compile):
    torchrun --nproc-per-node=2 --master-addr=127.0.0.1 reproduce_ddp_accumulation.py

Both cases should give a weight gradient of 60. The existing DistConvDDP hook
gives 60 for a single backward, but 80 for two accumulated microbatches, causing
the final assertion to fail.
"""

import torch
import torch.distributed as dist
from torch import nn

from distconv import DistConvDDP, ParallelStrategy


def main():
    dist.init_process_group("gloo")
    try:
        if dist.get_world_size() != 2:
            raise ValueError("Run this reproducer with exactly two processes.")

        rank = dist.get_rank()
        strategy = ParallelStrategy(num_shards=2, device_type="cpu")
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ddp = DistConvDDP(model, parallel_strategy=strategy)

        # Each rank supplies one spatial shard's contributions. Ordinary tensors
        # suffice here: the issue is in DistConvDDP's gradient scaling.
        # Rank 0 gets [1, 2]; rank 1 gets [3, 4].
        inputs = torch.tensor([[2.0 * rank + 1], [2.0 * rank + 2]])

        # y = w*x, loss = y**2, so dloss/dw = 2*w*x**2.
        # At w=1, summing both shards and both microbatches gives:
        expected = torch.tensor([[2.0 * (1**2 + 2**2 + 3**2 + 4**2)]])

        # Control: process both local inputs in one forward/backward.
        ddp.zero_grad(set_to_none=True)
        ddp(inputs).square().sum().backward()
        single_backward = model.weight.grad.clone()

        # Same inputs and unchanged weight, now accumulated over two backwards.
        ddp.zero_grad(set_to_none=True)
        with ddp.no_sync():
            ddp(inputs[:1]).square().sum().backward()
        ddp(inputs[1:]).square().sum().backward()
        accumulated = model.weight.grad.clone()

        if rank == 0:
            print(f"PyTorch: {torch.__version__}", flush=True)
            print(f"Expected gradient: {expected.item():g}", flush=True)
            print(f"Single backward:   {single_backward.item():g}", flush=True)
            print(f"Accumulated:       {accumulated.item():g}", flush=True)

        torch.testing.assert_close(single_backward, expected, rtol=0, atol=0)
        torch.testing.assert_close(accumulated, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
