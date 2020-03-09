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
        tensor = torch.ones(size)
        t = Timer()
        with t(name):
            with t('cpu->cuda'):
                tensor.to(torch.device('cpu'))
            fn(t, tensor, group)
            with t('cuda->cpu'):
                tensor.to(torch.device('cuda'))
        elapsed_time += t.elapsed_time
    print('[{}]:elapsed_time: {}'.format(name, elapsed_time))

def main():
    rank = int(os.environ['RANK'])
    group = rendezvous('gloo', rank, 2)
    runs = 3
    size = 2**30
    for function in functions:
        microbenchmark(function, function.__name__, group)
    
if __name__ == '__main__':
    main()