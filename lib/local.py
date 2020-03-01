#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
from functools import reduce
import numpy as numpy

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def init_processes(f, size):
    processes=[]
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, f))
        p.start()
        processes.append(p)
    return lambda: [p.join() for p in processes]