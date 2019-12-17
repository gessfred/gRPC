#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
import argparse
from all_reduce import ms_allreduce, ms_allreduce_un, ring_all_reduce, allreduce, allreduce_quant
from quantizy import quantizy
from torch.multiprocessing import Process
import time
from subprocess import run, Popen, PIPE
import torch.distributed as dist
import os
import datetime

def run(fn, size, iters=1000):
    tensor = torch.ones(2**size)
    res = [torch.ones(2**(size - 5))] if in_place else []
    start = time.time()
    for _ in range(iters):
        fn(tensor, 24, *res)
    exec_time = time.time() - start
    print(exec_time)

if __name__ == '__main__':
    f, _ = quantizy(sys.argv[1])
    run(f, int(sys.argv[2]))