"""Halo values and layouts across the plain-contiguous P2P transport."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode

from distconv import distconv as implementation


def make_halos(rank, layout, device):
    shape = (2, 3, 4, 5) if layout == "channels_last" else (2, 3, 4, 5, 6)
    count = 1
    for size in shape:
        count *= size
    minus = torch.arange(count, device=device, dtype=torch.float32).reshape(shape)
    minus = minus + rank * 10000
    if layout == "channels_last":
        minus = minus.contiguous(memory_format=torch.channels_last)
    elif layout == "channels_last_3d":
        minus = minus.contiguous(memory_format=torch.channels_last_3d)
    elif layout == "transposed":
        minus = minus.transpose(2, 3)
    return minus, minus + 1000


@pytest.mark.parametrize(
    "layout", ["contiguous", "channels_last", "channels_last_3d", "transposed"]
)
@pytest.mark.parametrize("periodic", [False, True])
def test_halo_transport_values_and_layout(layout, periodic, device):
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip("Requires at least two ranks")
    rank = dist.get_rank()
    minus_rank, plus_rank = (rank - 1) % world_size, (rank + 1) % world_size
    send_minus, send_plus = make_halos(rank, layout, device)
    batch_isend_irecv = dist.batch_isend_irecv

    def checked_exchange(ops):
        assert all(op.tensor.is_contiguous() for op in ops)
        if layout != "transposed":
            # Dense supported layouts need only a view, including on GPUs.
            send_ptrs = {send_minus.data_ptr(), send_plus.data_ptr()}
            assert all(
                op.tensor.data_ptr() in send_ptrs
                for op in ops
                if op.op is dist.isend
            )
        return batch_isend_irecv(ops)

    with patch.object(dist, "batch_isend_irecv", side_effect=checked_exchange):
        actual = implementation._exchange_halos(
            send_minus, send_plus, rank, world_size, minus_rank, plus_rank, periodic
        )
    expected_minus = (
        make_halos(minus_rank, layout, device)[1]
        if rank > 0 or periodic else torch.zeros_like(send_minus)
    )
    expected_plus = (
        make_halos(plus_rank, layout, device)[0]
        if rank < world_size - 1 or periodic else torch.zeros_like(send_plus)
    )
    for result, expected, source in zip(
        actual, (expected_minus, expected_plus), (send_minus, send_plus)
    ):
        torch.testing.assert_close(result, expected)
        assert result.stride() == torch.empty_like(source).stride()


@pytest.mark.parametrize(
    "layout", ["contiguous", "channels_last", "channels_last_3d", "transposed"]
)
@pytest.mark.parametrize("periodic", [False, True])
def test_single_shard_halos_do_not_communicate(layout, periodic, device):
    send_minus, send_plus = make_halos(dist.get_rank(), layout, device)
    with patch.object(
        dist, "batch_isend_irecv", side_effect=AssertionError("self communication")
    ):
        actual = implementation._exchange_halos(
            send_minus, send_plus, 0, 1, dist.get_rank(), dist.get_rank(), periodic
        )
    with FakeTensorMode() as mode:
        fake_outputs = implementation._exchange_halos(
            mode.from_tensor(send_minus), mode.from_tensor(send_plus),
            0, 1, dist.get_rank(), dist.get_rank(), periodic,
        )
    expected = (send_plus, send_minus) if periodic else (
        torch.zeros_like(send_minus), torch.zeros_like(send_plus)
    )
    input_ptrs = {send_minus.data_ptr(), send_plus.data_ptr()}
    for result, reference, fake in zip(actual, expected, fake_outputs):
        torch.testing.assert_close(result, reference)
        assert result.stride() == fake.stride()
        assert result.data_ptr() not in input_ptrs
    assert actual[0].data_ptr() != actual[1].data_ptr()
