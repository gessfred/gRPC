import torch
import torch.distributed as dist
import os
import datetime
import time
from contextlib import contextmanager
import numpy as np
from timer import CUDATimer
from microbenchmarkcomm import functions, rendezvous


def send_hack(tensor, dst):
    dist.broadcast(tensor, dist.get_rank())

def recv_hack(tensor, src):
    dist.broadcast(tensor, src)

def microbenchmark(runs=10, size=2**26):
    elapsed_time = 0
    rank = dist.get_rank()
    other = (rank + 1) % 2
    f = dist.send if rank == 0 else dist.recv
    for _ in range(runs):
        tensor = torch.ones(size)
        t = Timer(name)
        with t(name):
            with t('cpu->cuda'):
                tensor.to(torch.device('cpu'))
            f(tensor, other)
            with t('cuda->cpu'):
                tensor.to(torch.device('cuda'))
        t.close()
        elapsed_time += (t.elapsed_time / runs)
    print('[{}]:elapsed_time: {}'.format(name, elapsed_time))

def main():
    rank = int(os.environ['RANK'])
    group = rendezvous('gloo', rank, 2)
    microbenchmark()
    
if __name__ == '__main__':
    main()