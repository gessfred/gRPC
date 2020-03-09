import torch
import torch.distributed as dist
import os
import datetime
import time
from contextlib import contextmanager
import numpy as np
from timer import Timer

def allreduce_(timer, tensor, group):
    with timer('reduce'):
        rank = dist.get_rank()
        chunks = list(tensor.view(dist.get_world_size(), -1))
        reqs = [dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group, async_op=True) for i, chunk in enumerate(chunks)]
        [req.wait() for req in reqs]
    with timer('all_gather'):
        chunk = chunks[rank]
        dist.all_gather(chunks, chunk, group=group)

"""def allreduce__(timer, tensor, group):
    with timer('reduce'):
        rank = dist.get_rank()
        chunks = list(tensor.view(dist.get_world_size(), -1))
        for i, chunk in enumerate(chunks):
            dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group, async_op=True)
    with timer('all_gather'):
        chunk = chunks[rank]
        dist.all_gather(chunks, chunk, group=group)"""

def allreduce(timer, tensor, group):
    rank = dist.get_rank()
    chunks = list(tensor.view(dist.get_world_size(), -1))
    for i, chunk in enumerate(chunks):
        dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    chunk = chunks[rank]
    dist.all_gather(chunks, chunk, group=group)

def allreducebaseline(timer, tensor, group):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

def rendezvous(rank, world_size):
    dist.init_process_group('nccl', rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
    return dist.new_group(range(world_size))

"""def measure(fn, name):
    t = Timer()
    with t(name):
        fn()"""

def main():
    rank = int(os.environ['RANK'])
    group = rendezvous(rank, 2)
    #allreduce(tensor, group)
    runs = 3
    size = 2**30
    for i in range(runs):
        tensor = torch.ones(size).cuda()
        t = Timer()
        with t('all_reduce_bare'):
            allreduce(tensor, group)
        t.dump()
    for i in range(runs):
        tensor = torch.ones(size).cuda()
        t = Timer()
        with t('all_reduce_bare'):
            allreduce_(t, tensor, group)
        t.dump()
    for i in range(runs):
        tensor = torch.ones(size).cuda()
        t4 = Timer()
        with t4('all_reduce_baseline'):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        t4.dump()
    """tensor = torch.ones(2**20).cuda()
    t2 = Timer()
    with t2('all_reduce_precise'):
        allreduce_(t2, tensor, group)
    t2.dump()
    tensor = torch.ones(2**20).cuda()
    t3 = Timer()
    with t3('all_reduce_compressed'):
        allreduce__(t3, tensor, group)
    t3.dump()
    tensor = torch.ones(2**20).cuda()
    t4 = Timer()
    with t4('all_reduce_baseline'):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    t4.dump()"""
    print(tensor)

if __name__ == '__main__':
    main()