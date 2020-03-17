import torch
import torch.distributed as dist

def send(tensor, dst):
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	dist.broadcast(tensor, rank, group=private)

def recv(tensor, src):
	private = dist.new_group([src, dist.get_rank()])
	dist.broadcast(tensor, src, group=private)

def isend(tensor, dst):
	rank = dist.get_rank()
	private = dist.new_group([rank, dst])
	return dist.broadcast(tensor, rank, group=private, async_op=True)

def irecv(tensor, src):
	private = dist.new_group([src, dist.get_rank()])
	return dist.broadcast(tensor, src, group=private, async_op=True)

def all_gather(tensor_list, tensor):
	dist.all_gather(tensor_list, tensor)

def all_reduce(tensor, group):
    rank = dist.get_rank()
    chunks = list(tensor.view(dist.get_world_size(), -1))
    for i, chunk in enumerate(chunks):
        dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    chunk = chunks[rank]
    dist.all_gather(chunks, chunk, group=group)
