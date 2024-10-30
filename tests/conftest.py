import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="session", autouse=True)
def pytorch_init():
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
    torch.manual_seed(0)
    yield
    dist.destroy_process_group()


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)
        return device
    else:
        device = torch.device("cpu")
        return device
