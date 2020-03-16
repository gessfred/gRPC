#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
from functools import reduce
import numpy as numpy
from all_reduce import ms_allreduce_un, ring_all_reduce, allreduce, allreduce_quant, ms_allreduce
from quantizy import quantizy
def run(rank, size):
    r = dist.get_rank()
    world = dist.get_world_size()
    peers = list(filter(lambda i: i != r, list(range(world))))
    tensor = torch.ones(64)
    group = dist.new_group(list(range(size)))
    q = quantizy('numpy')
    allreduce_quant(r, world, peers,tensor, *q)
    tensor2 = torch.ones(64)
    ms_allreduce(r, world, peers, tensor2, *q)
    print(tensor)
    print('rank', tensor2)

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == '__main__':
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()