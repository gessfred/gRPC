import torch
import torch.distributed as dist
from torch.distributed import ReduceOp, group
import asyncio
import concurrent.futures
import threading
import time
from quantizy import quantizy
from math import ceil
_pack, _unpack = quantizy('gpu')

def send_quantized(timer, tensor, dst, bits):
	assert(bits in [1,2,4,8])
	with timer('pack'):
		quantized, padding = _pack(tensor, bits)
	with timer('send'):
		rank = dist.get_rank()
		private = dist.new_group([rank, dst])
		dist.broadcast(quantized, rank, group=private)
	return padding

def recv_quantized(timer, tensor, src, bits):
	assert(bits in [1,2,4,8])
	with timer('pack'):
		tensor_size = tensor.view(-1).shape[0]
		padding = (32 - tensor_size) % 32
		quantized_size = ceil(tensor.view(-1).shape[0]/(32/bits))
		quantized = torch.zeros(quantized_size, dtype=torch.int32, device=tensor.device)
	with timer('recv'):
		private = dist.new_group([src, dist.get_rank()])
		dist.broadcast(quantized, src, group=private)
	with timer('unpack'):
		tensor.copy_(_unpack(quantized, padding, bits))

def all_gather_quantized(timer, tensor_list, tensor, bits=1, group=group.WORLD):
	with timer('pack'):
		quantized, padding = _pack(tensor, bits)
		tensor_sizes = [t.view(-1).shape[0] for t in tensor_list]
		padding_list = [(32 - s) % 32 for s in tensor_sizes]
		quantized_sizes = [ceil(s/(32/bits)) for s in tensor_sizes]
		quantized_list = [torch.empty(s, dtype=quantized.dtype, device=tensor.device) for s in quantized_sizes]
	with timer('gather'):
		dist.all_gather(quantized_list, quantized, group=group)
	with timer('unpack'):
		for t, q, p in zip(tensor_list, quantized_list, padding_list):
			t.copy_(_unpack(q, p, bits))

def all_reduce_quantised(timer, tensor, op=ReduceOp.SUM, bits=1, group=group.WORLD):
	with timer('preprocess'):
		tensor_list = [torch.empty(tensor.shape, device=tensor.device) for _ in range(dist.get_world_size(group))]
	with timer('all_gather'):
		rank = dist.get_rank()
		all_gather_quantized(timer, tensor_list, tensor, bits=bits, group=group)
	# reduce tensors on master node, as gather is synchronous we know the tensor list is ready
	with timer('postprocess'):
		ops = {ReduceOp.SUM: lambda t_l: torch.sum(t_l, dim=0),
			   ReduceOp.PRODUCT: lambda t_l: torch.prod(t_l, dim=0)}
		tensor.copy_(ops[op](torch.stack(tensor_list)))

def reduce_quantised_centralised(timer, tensor, dst, op=ReduceOp.SUM, bits=1, group=group.WORLD):
	with timer('preprocess'):
		tensor_list = [torch.empty(tensor.shape, device=tensor.device) for _ in range(dist.get_world_size(group))]
	with timer('all_gather'):
		rank = dist.get_rank()
		all_gather_quantized(timer, tensor_list, tensor, bits=bits, group=group)
	# reduce tensors on master node, as gather is synchronous we know the tensor list is ready
	if rank == dst:
		with timer('postprocess'):
			ops = {ReduceOp.SUM: lambda t_l: torch.sum(t_l, dim=0),
			   	   ReduceOp.PRODUCT: lambda t_l: torch.prod(t_l, dim=0)}
				tensor.copy_(ops[op](torch.stack(tensor_list)))
