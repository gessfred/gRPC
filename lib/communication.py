import torch
import torch.distributed as dist
import quantizy import quantizy
from math import ceil

_pack, _unpack = quantizy('gpu')

def send(tensor, dst):
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	dist.broadcast(tensor, rank, group=private)

def send_quantized(tensor, dst, bits):
	assert(bits in [1,2,4,8])
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	quantized, padding = _pack(tensor, bits)
	dist.broadcast(quantized, rank, group=private)
	return padding

def recv(tensor, src):
	private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(tensor, src, group=private)

def recv_quantized(tensor, src, bits, padding):
	assert(bits in [1,2,4,8])
	quantized_size = ceil(tensor.view(-1).shape[0]/(32/bits))
	quantized = torch.zeros(quantized_size, device=tensor.device)
	private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(quantized, src, group=private)
	tensor.copy_(_unpack(quantized, padding, bits))


def isend(tensor, dst):
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	return dist.broadcast(tensor, rank, group=private, async_op=True)

def isend_quantized(tensor, dst, bits):
	assert(bits in [1,2,4,8])
	tensor_size = tensor.view(-1).shape[0]
	padding = (32 - pad_size) % 32
	quantized_size = ceil(tensor_size/(32/bits))
	quantized = torch.zeros(quantized_size, device=tensor.device)
	private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(quantized, rank, group=private, async_op=True)
	return padding

def irecv(tensor, src):
	private = dist.new_group([src, dist.get_rank()])
	return dist.broadcast(tensor, src, group=private, async_op=True)

def irecv_quantized(tensor, src, bits):
	assert(bits in [1,2,4,8])
	tensor_size = tensor.view(-1).shape[0]
	padding = (32 - pad_size) % 32
	quantized_size = ceil(tensor_size/(32/bits))
	quantized = torch.zeros(quantized_size, device=tensor.device)
	private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(quantized, src, group=private, async_op=True)
	tensor.copy_(_unpack(quantized, padding, bits))

def all_gather(tensor_list, tensor):
	dist.all_gather(tensor_list, tensor)

def all_reduce(tensor, group):
    rank = dist.get_rank()
    chunks = list(tensor.view(dist.get_world_size(), -1))
    for i, chunk in enumerate(chunks):
        dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    chunk = chunks[rank]
    dist.all_gather(chunks, chunk, group=group)
