import torch
import torch.distributed as dist
import os
import datetime
import time
from contextlib import contextmanager
import numpy as np
from timer import Timer
from microbenchmarkcomm import functions, rendezvous

def microbenchmark(fn, name, group, runs=10, size=2**20):
    elapsed_time = 0
    for _ in range(runs):
        tensor = torch.ones(size).cuda()
        t = Timer(name)
        with t(name):
            fn(t, tensor, group)
        t.upload()
        elapsed_time += (t.elapsed_time / runs)
    print('[{}]:elapsed_time: {}'.format(name, elapsed_time))

def main():
    rank = int(os.environ['RANK'])
    group = rendezvous('nccl', rank, 2)
    #allreduce(tensor, group)
    runs = 3
    size = 2**30
    for function in functions:
        microbenchmark(function, function.__name__, group)
    
if __name__ == '__main__':
    main()