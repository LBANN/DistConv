import torch
import torch.distributed as dist
from distconv import ParallelStrategy


def fp32_allclose(a, b, rtol=1e-3, atol=1e-4):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def cleanup_parallel_strategy(ps: ParallelStrategy):
    for g in ps.device_mesh.get_all_groups():
        dist.destroy_process_group(g)
