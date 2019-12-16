#!/usr/bin/env python
import sys
sys.path.append('../lib')
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
from functools import reduce
import numpy as numpy
from all_reduce import ms_allreduce_un, ring_all_reduce, allreduce

def run(rank, size):
    tensor = torch.ones(2**3)
    group = dist.new_group(list(range(size)))
    allreduce(tensor)
    print('rank', tensor)

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == '__main__':
    size = 8
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()