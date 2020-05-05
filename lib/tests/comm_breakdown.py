import sys
sys.path.append('..')
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import argparse
import os
import datetime
import time
from torch.multiprocessing import Process, spawn
import cProfile
import timed_communication as comm
from quantizy import quantize_gpu, unquantize_gpu
import numpy as np
from timer import TimerBase as Timer
from timer import CUDATimer

# Tests the speed of the quantised send/recv primitives.
# Assumes exactly 2 nodes
def send_recv_speed(runs=1, size=32*2**5, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    bit_list = [1,2,4,8]
    for bits in bit_list:
        timer = CUDATimer('send_recv')
        start = time.time()
        for _ in range(runs):
            rank = dist.get_rank()
            other = (rank + 1) % 2
        # The smallest rank is the sender
            if rank < other:
                comm.send_quantized(timer, tensor1, other, bits)
            else:
                comm.recv_quantized(timer, tensor1, other, bits)
            dist.barrier()
        exec_time = time.time() - start
        timer.dump()
        print('T: {:6.6}, B: {}, runs: {}, size" {}'.format(str(exec_time), bits, runs, size))

# Tests the speed of the quantised all_gather collective.
def all_gather_speed(runs=1, size=32*2**5, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    bit_list = [1,2,4,8]
    for bits in bit_list:
        timer = CUDATimer('all_gather')
        start = time.time()
        for _ in range(runs):
            rank = dist.get_rank()
            N = dist.get_world_size()
            tensor_list1 = [torch.empty(size, device=device) for _ in range(N)]
            comm.all_gather_quantized(timer, tensor_list1, tensor1, bits)
            dist.barrier()
        exec_time = time.time() - start
        timer.dump()
        print('T: {:6.6}, B: {}, runs: {}, size" {}'.format(str(exec_time), bits, runs, size))

# Tests the speed of the quantised all_reduce collective.
def all_reduce_speed(runs=1, size=32*2**5, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    op = ReduceOp.SUM
    bit_list = [1,2,4,8]
    for bits in bit_list:
        timer = CUDATimer('all_reduce')
        start = time.time()
        for _ in range(runs):
            comm.all_reduce_quantised(timer, tensor1, op=op, bits=bits)
            dist.barrier()
        exec_time = time.time() - start
        timer.dump()
        print('T: {:6.6}, B: {}, runs: {}, size" {}'.format(str(exec_time), bits, runs, size))

# Tests the speed of the quantised centralised reduce collective.
def reduce_centralised_speed(runs=1, size=32*2**5, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    op = ReduceOp.SUM
    master = 0
    bit_list = [1,2,4,8]
    for bits in bit_list:
        timer = CUDATimer('reduce')
        start = time.time()
        for _ in range(runs):
            comm.reduce_quantised_centralised(timer, tensor1, master, op=op, bits=bits)
            dist.barrier()
        exec_time = time.time() - start
        timer.dump()
        print('T: {:6.6}, B: {}, runs: {}, size" {}'.format(str(exec_time), bits, runs, size))

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU: {}".format(device))
    else:
        device = None
        print("Using CPU:")

    world_size=int(os.environ['WORLD_SIZE'])
    rank=int(os.environ['RANK'])
    backend='nccl'

    print('Master: {}:60000'.format(os.environ['MASTER_ADDR']))
    dist.init_process_group(backend, rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))

    parser = argparse.ArgumentParser(description='Run gpu quantized communication benchmarks')
    parser.add_argument('function', metavar='fn',
                    help='send, isend, all_gather, all_reduce, reduce')
    args = parser.parse_args()

    max_nodes = 2
    sizes = [14,16,18,20,22]
    # sizes = [1]

    if args.function == 'send':
        print("Send/Recv")
        for s in sizes:
            send_recv_speed(size=32*2**s, device=device)

    if args.function == 'all_gather':
        print("All Gather")
        for s in sizes:
            all_gather_speed(size=32*2**s, device=device)

    if args.function == 'all_reduce':
        print("All Reduce")
        for s in sizes:
            all_reduce_speed(size=32*2**s, device=device)

    if args.function == 'reduce':
        print("Reduce Centralised")
        for s in sizes:
            reduce_centralised_speed(size=32*2**s, device=device)

if __name__ == '__main__':
    main()
