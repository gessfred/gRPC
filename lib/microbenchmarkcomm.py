import torch
import torch.distributed as dist
import os
import datetime
import time
from contextlib import contextmanager
import numpy as np
from timer import Timer

def allreduce_instrumented(timer, tensor, group):
    with timer('reduce'):
        rank = dist.get_rank()
        chunks = list(tensor.view(dist.get_world_size(), -1))
        reqs = [dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group, async_op=True) for i, chunk in enumerate(chunks)]
        [req.wait() for req in reqs]
    with timer('all_gather'):
        chunk = chunks[rank]
        dist.all_gather(chunks, chunk, group=group)

def allreduce_uninstrumented(timer, tensor, group):
    rank = dist.get_rank()
    chunks = list(tensor.view(dist.get_world_size(), -1))
    for i, chunk in enumerate(chunks):
        dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    chunk = chunks[rank]
    dist.all_gather(chunks, chunk, group=group)

def allreducebaseline(timer, tensor, group):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

functions = [allreducebaseline, allreduce_instrumented, allreduce_uninstrumented]

def rendezvous(backend, rank, world_size):
    dist.init_process_group(backend, rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
    return dist.new_group(range(world_size))
