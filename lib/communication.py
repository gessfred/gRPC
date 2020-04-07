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

class AsyncHandler:

	def __init__(self, tensor, quantized, padding, bits, handler):
		self.loop = asyncio.get_event_loop()
		self.t = threading.Thread(target=self.run, args=(self.loop, tensor, quantized, padding, bits, handler))
		self.t.start()

	def is_completed(self):
		return not self.t.is_alive()

	def wait(self):
		self.t.join()

	def run(self, loop, tensor, quantized, padding, bits, handler):
		asyncio.set_event_loop(loop)
		loop.run_until_complete(self.irun(tensor, quantized, padding, bits, handler))

	async def irun(self, tensor, quantized, padding, bits, handler):
		handler.wait()
		tensor.copy_(_unpack(quantized, padding, bits))
		return True


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

def recv_quantized(tensor, src, bits):
	assert(bits in [1,2,4,8])
	tensor_size = tensor.view(-1).shape[0]
	padding = (32 - tensor_size) % 32
	quantized_size = ceil(tensor.view(-1).shape[0]/(32/bits))
	quantized = torch.zeros(quantized_size, dtype=torch.int32, device=tensor.device)
	private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(quantized, src, group=private)
	tensor.copy_(_unpack(quantized, padding, bits))

def isend(tensor, dst):
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	return dist.broadcast(tensor, rank, group=private, async_op=True)

def isend_quantized(tensor, dst, bits):
	assert(bits in [1,2,4,8])
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	quantized, padding = _pack(tensor, bits)
	h = dist.broadcast(quantized, rank, group=private, async_op=True)
	return h

def irecv(tensor, src):
	private = dist.new_group([src, dist.get_rank()])
	return dist.broadcast(tensor, src, group=private, async_op=True)

def irecv_quantized(tensor, src, bits):
	assert(bits in [1,2,4,8])
	tensor_size = tensor.view(-1).shape[0]
	padding = (32 - tensor_size) % 32
	quantized_size = ceil(tensor_size/(32/bits))
	quantized = torch.zeros(quantized_size, dtype=torch.int32, device=tensor.device)
	private = dist.new_group([src, dist.get_rank()])
	handler = dist.broadcast(quantized, src, group=private, async_op=True)
	return AsyncHandler(tensor, quantized, padding, bits, handler)

def all_gather(tensor_list, tensor, group=group.WORLD):
	dist.all_gather(tensor_list, tensor, group=group)

def all_gather_quantized(tensor_list, tensor, bits=1, group=group.WORLD):
	quantized, padding = _pack(tensor, bits)
	tensor_sizes = [t.view(-1).shape[0] for t in tensor_list]
	padding_list = [(32 - s) % 32 for s in tensor_sizes]
	quantized_sizes = [ceil(s/(32/bits)) for s in tensor_sizes]
	quantized_list = [torch.empty(s, dtype=quantized.dtype, device=tensor.device) for s in quantized_sizes]
	dist.all_gather(quantized_list, quantized, group=group)
	for t, q, p in zip(tensor_list, quantized_list, padding_list):
		t.copy_(_unpack(q, p, bits))

# Not usuable with NCCL
def gather_quantized(tensor, gather_list=None, bits=1, dst=0, group=group.WORLD):
	quantized, padding = _pack(tensor, bits)
	tensor_sizes = [t.view(-1).shape[0] for t in gather_list]
	padding_list = [(32 - s) % 32 for s in tensor_sizes]
	quantized_sizes = [ceil(s/(32/bits)) for s in tensor_sizes]
	quantized_list = [torch.empty(s, dtype=quantized.dtype, device=tensor.device) for s in quantized_sizes]
	dist.all_gather(quantized_list, quantized, group=group)
	if dist.get_rank() == dst:
		for t, q, p in zip(gather_list, quantized_list, padding_list):
			t.copy_(_unpack(q, p, bits))

def all_reduce(tensor, group=group.WORLD):
    rank = dist.get_rank()
    chunks = list(tensor.view(dist.get_world_size(), -1))
    for i, chunk in enumerate(chunks):
        dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    chunk = chunks[rank]
    dist.all_gather(chunks, chunk, group=group)

def all_reduce_quantised_centralised(tensor, master=0, op=ReduceOp.SUM, bits=1, group=group.WORLD):
	#gather tensors on master node
	rank = dist.get_rank()
	if rank == master:
		tensor_list = [torch.empty(tensor.shape, device=tensor.device) for _ in range(dist.get_world_size())]
	else:
		tensor_list = None
	gather_quantized(tensor, gather_list=tensor_list, bits=bits, dst=master, group=group)
	# reduce tensors on master node, as gather is synchronous we know the tensor list is ready
	if rank == master:
		ops = {ReduceOp.SUM: lambda t_l: torch.sum(t_l, dim=0),
			   ReduceOp.PRODUCT: lambda t_l: torch.prod(t_l, dim=0)}
		tensor.copy_(ops[op](torch.stack(tensor_list)))
	# broadcasting non quantized tensor
	dist.broadcast(tensor, master, group=group)

def reduce_quantised_centralised(tensor, dst, op=ReduceOp.SUM, bits=1, group=group.WORLD):
	#gather tensors on master node
	rank = dist.get_rank()
	if rank == dst:
		tensor_list = [torch.empty(tensor.shape, device=tensor.device) for _ in range(dist.get_world_size())]
	else:
		tensor_list = None
	gather_quantized(tensor, gather_list=tensor_list, bits=bits, dst=dst, group=group)
	#reduce tensors on master node, as gather as synchronous we know the tensor list is ready
	if rank == dst:
		ops = {ReduceOp.SUM: lambda t_l: torch.sum(t_l, dim=0),
			   ReduceOp.PRODUCT: lambda t_l: torch.prod(t_l, dim=0)}
		tensor.copy_(ops[op](torch.stack(tensor_list)))
