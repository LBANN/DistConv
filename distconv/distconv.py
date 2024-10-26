import torch
from torch.autograd import Function
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
from torch.utils._pytree import tree_map
from typing import List


class ParallelStrategy:
    def __init__(self, num_shards: int, shard_dim: int = 2):
        self.num_shards = num_shards
        self.shard_dim = shard_dim

        world_size = dist.get_world_size()
        self.ddp_ranks = world_size // num_shards

        self.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(self.ddp_ranks, num_shards),
            mesh_dim_names=("ddp", "dc"),
        )

    def get_distconv_index(self):
        return dist.get_rank() % self.num_shards


def check_is_distconv_supported(
    tensor_shard_dim: int,
    tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
):
    shard_dim = tensor_shard_dim - 2
    kernel_size = weight.size(tensor_shard_dim)
    if dilation[shard_dim] != 1:
        raise Exception("DistConv: dilation must be 1")
    if tensor.size(tensor_shard_dim) % stride[shard_dim] != 0:
        raise Exception("DistConv: input size must be divisible by stride")
    if kernel_size % 2 == 1:
        if (kernel_size // 2) != padding[shard_dim]:
            raise Exception(
                'DistConv: when kernel size is odd, padding must be equivalent to "same"'
            )
    else:
        if padding[shard_dim] != 0:
            raise Exception("DistConv: when kernel size is even, padding must be zero")
        if stride[shard_dim] % kernel_size != 0:
            raise Exception(
                "DistConv: when kernel size is even, stride must be divisble by kernel size"
            )


def forward_halo_exchange(
    tensor: torch.Tensor, halo_size: int, parallel_strategy: ParallelStrategy
):
    # check if halo exchange needed
    if halo_size == 0:
        return tensor

    # do halo exchange
    shard_dim = parallel_strategy.shard_dim
    num_shards = parallel_strategy.num_shards
    shard_ind = parallel_strategy.get_distconv_index()
    rank = dist.get_rank()

    inner_halo_minus = tensor.narrow(shard_dim, 0, halo_size)
    inner_halo_plus = tensor.narrow(shard_dim, -halo_size, halo_size)
    halo_minus = torch.zeros_like(inner_halo_minus)
    halo_plus = torch.zeros_like(inner_halo_plus)

    ops = []
    if shard_ind > 0:
        ops += [
            dist.P2POp(dist.irecv, halo_minus, rank - 1),
            dist.P2POp(dist.isend, inner_halo_minus.contiguous(), rank - 1),
        ]
    if shard_ind < (num_shards - 1):
        ops += [
            dist.P2POp(dist.isend, inner_halo_plus.contiguous(), rank + 1),
            dist.P2POp(dist.irecv, halo_plus, rank + 1),
        ]

    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # return tensor with halo
    tensor_with_halo = torch.cat([halo_minus, tensor, halo_plus], dim=shard_dim)

    return tensor_with_halo


def backward_halo_exchange(
    tensor: torch.Tensor, halo_size: int, parallel_strategy: ParallelStrategy
):
    # check if halo exchange needed
    if halo_size == 0:
        return tensor

    # do halo exchange
    shard_dim = parallel_strategy.shard_dim
    num_shards = parallel_strategy.num_shards
    shard_ind = parallel_strategy.get_distconv_index()
    rank = dist.get_rank()

    send_halo_minus = tensor.narrow(shard_dim, 0, halo_size)
    send_halo_plus = tensor.narrow(shard_dim, -halo_size, halo_size)
    recv_halo_minus = torch.zeros_like(send_halo_minus)
    recv_halo_plus = torch.zeros_like(send_halo_plus)

    ops = []
    if shard_ind > 0:
        ops += [
            dist.P2POp(dist.irecv, recv_halo_minus, rank - 1),
            dist.P2POp(dist.isend, send_halo_minus.contiguous(), rank - 1),
        ]
    if shard_ind < (num_shards - 1):
        ops += [
            dist.P2POp(dist.isend, send_halo_plus.contiguous(), rank + 1),
            dist.P2POp(dist.irecv, recv_halo_plus, rank + 1),
        ]

    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # accumulate halos
    inner_tensor = tensor.narrow(
        shard_dim, halo_size, tensor.size(shard_dim) - 2 * halo_size
    )
    inner_halo_minus = inner_tensor.narrow(shard_dim, 0, halo_size)
    inner_halo_plus = inner_tensor.narrow(shard_dim, -halo_size, halo_size)
    inner_halo_minus.add_(recv_halo_minus)
    inner_halo_plus.add_(recv_halo_plus)

    return inner_tensor


def distconv_forward(func, args, kwargs):
    # get args
    args = list(args)
    tensor, weight, bias, stride, padding, dilation = args[:6]
    parallel_strategy = tensor._parallel_strategy
    shard_dim = parallel_strategy.shard_dim
    tensor = tensor._tensor

    # check supported
    check_is_distconv_supported(shard_dim, tensor, weight, stride, padding, dilation)

    # do halo exchange
    kernel_size = weight.size(shard_dim)
    halo_size = kernel_size // 2 if (kernel_size % 2 == 1) else 0
    tensor_with_halo = forward_halo_exchange(tensor, halo_size, parallel_strategy)

    # update args
    args[0] = tensor_with_halo
    padding[shard_dim - 2] = 0
    args[4] = padding

    # do conv
    out_tensor = func(*args, **kwargs)

    # wrap outputs
    return DCTensor(out_tensor, parallel_strategy)


def distconv_backward(func, args, kwargs):
    # get args
    args = list(args)
    grad_out_tensor, input_tensor, weight, bias_size, stride, padding, dilation = args[
        :7
    ]
    parallel_strategy = grad_out_tensor._parallel_strategy
    shard_dim = parallel_strategy.shard_dim
    grad_out_tensor = grad_out_tensor._tensor
    input_tensor = input_tensor._tensor

    # check supported
    check_is_distconv_supported(
        shard_dim, input_tensor, weight, stride, padding, dilation
    )

    # do halo exchange
    halo_size = weight.size(shard_dim) // 2
    input_tensor_with_halo = forward_halo_exchange(
        input_tensor, halo_size, parallel_strategy
    )

    # update args
    args[0] = grad_out_tensor
    args[1] = input_tensor_with_halo
    padding[shard_dim - 2] = 0
    args[5] = padding

    # do conv
    grad_in_tensor, grad_weight, grad_bias = func(*args, **kwargs)
    grad_in_tensor = backward_halo_exchange(
        grad_in_tensor, halo_size, parallel_strategy
    )
    grad_in_tensor = DCTensor(grad_in_tensor, parallel_strategy)

    # wrap outputs
    return grad_in_tensor, grad_weight, grad_bias


class DCTensor(torch.Tensor):
    _tensor: torch.Tensor
    _parallel_strategy: ParallelStrategy

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, parallel_strategy: ParallelStrategy):
        dc_tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )
        dc_tensor._tensor = tensor
        dc_tensor._parallel_strategy = parallel_strategy

        return dc_tensor

    @classmethod
    def from_shard(cls, tensor: torch.Tensor, parallel_strategy: ParallelStrategy):
        return _FromTensor.apply(tensor, parallel_strategy)

    @classmethod
    def distribute(cls, tensor: torch.Tensor, parallel_strategy: ParallelStrategy):
        dtensor = distribute_tensor(
            tensor,
            device_mesh=parallel_strategy.device_mesh["dc"],
            placements=[Shard(parallel_strategy.shard_dim)],
        )
        return cls(dtensor.to_local(), parallel_strategy)

    def to_ddp(self):
        device_mesh = self._parallel_strategy.device_mesh["dc"]
        shard_dim = self._parallel_strategy.shard_dim
        dtensor = DTensor.from_local(
            _ToTensor.apply(self),
            device_mesh=device_mesh,
            placements=[Shard(shard_dim)],
        ).redistribute(device_mesh=device_mesh, placements=[Shard(0)])
        return dtensor.to_local()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, DCTensor):
                return t._tensor
            else:
                return t

        def wrap(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, DCTensor):
                return DCTensor(t, self._parallel_strategy)
            else:
                return t

        if func is torch.ops.aten.convolution.default:
            return distconv_forward(func, args, kwargs)
        elif func is torch.ops.aten.convolution_backward.default:
            return distconv_backward(func, args, kwargs)

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

    def __repr__(self):
        return super().__repr__(tensor_contents=f"{self._tensor}")


class _FromTensor(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, parallel_strategy: ParallelStrategy):
        return DCTensor(tensor, parallel_strategy)

    @staticmethod
    def backward(ctx, grad: DCTensor):
        return _ToTensor.apply(grad), None


class _ToTensor(Function):
    @staticmethod
    def forward(ctx, dc_tensor: DCTensor):
        ctx.parallel_strategy = dc_tensor._parallel_strategy
        return dc_tensor._tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _FromTensor.apply(grad, ctx.parallel_strategy)
