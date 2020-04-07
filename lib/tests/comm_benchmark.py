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
import communication as comm
from quantizy import quantize_gpu, unquantize_gpu
import numpy as np
from timer import TimerBase as Timer
from timer import CUDATimer

# Tests the speed of the quantised send/recv primitives.
# Assumes exactly 2 nodes
def send_recv_speed(runs=100, size=32*2**5, quantized=False, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    if not quantized:
        bit_list = [32]
    else:
        bit_list = [1,2,4,8]
    for bits in bit_list:
        start = time.time()
        for _ in range(runs):
            rank = dist.get_rank()
            other = (rank + 1) % 2
        # The smallest rank is the sender
            if rank < other:
                if not quantized:
                    comm.send(tensor1, other)
                else:
                    comm.send_quantized(tensor1, other, bits)
            else:
                if not quantized:
                    comm.recv(tensor1, other)
                else:
                    comm.recv_quantized(tensor1, other, bits)
        exec_time = time.time() - start
        print('Q: {}, T: {:6.6}, B: {}'.format(quantized, str(exec_time), bits))

# Tests the speed of the quantised isend/irecv primitives.
# Assumes exactly 2 nodes
def isend_irecv_speed(runs=100, size=32*2**5, quantized=False, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    if not quantized:
        bit_list = [32]
    else:
        bit_list = [1,2,4,8]
    for bits in bit_list:
        start = time.time()
        for _ in range(runs):
            rank = dist.get_rank()
            other = (rank + 1) % 2
        # The smallest rank is the sender
            if rank < other:
                if not quantized:
                    h = comm.isend(tensor1, other)
                else:
                    h = comm.isend_quantized(tensor1, other, bits)
            else:
                if not quantized:
                    h = comm.irecv(tensor1, other)
                else:
                    h = comm.irecv_quantized(tensor1, other, bits)
        exec_time = time.time() - start
        print('Q: {}, T: {:6.6}, B: {}'.format(quantized, str(exec_time), bits))

# Tests the speed of the quantised all_gather collective.
def all_gather_speed(runs=100, size=32*2**5, quantized=False, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    if not quantized:
        bit_list = [32]
    else:
        bit_list = [1,2,4,8]
    for bits in bit_list:
        start = time.time()
        for _ in range(runs):
            rank = dist.get_rank()
            N = dist.get_world_size()
            tensor_list1 = [torch.empty(size, device=device) for _ in range(N)]
            if not quantized:
                dist.all_gather(tensor_list1, tensor1)
            else:
                comm.all_gather_quantized(tensor_list1, tensor1, bits)
        exec_time = time.time() - start
        print('Q: {}, T: {:6.6}, B: {}'.format(quantized, str(exec_time), bits))

# Tests the speed of the quantised gather collective.
def gather_speed(runs=100, size=32*2**5, quantized=False, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    if not quantized:
        bit_list = [32]
    else:
        bit_list = [1,2,4,8]
    for bits in bit_list:
        start = time.time()
        for _ in range(runs):
            rank = dist.get_rank()
            N = dist.get_world_size()
            master = 0
            if rank == master:
                tensor_list1 = [torch.empty(size, device=device) for _ in range(N)]
            else:
                tensor_list1 = None

            if not quantized:
                dist.gather(tensor1, gather_list=tensor_list1, dst=master)
            else:
                comm.gather_quantized(tensor1, gather_list=tensor_list1, bits=bits, dst=master)
        exec_time = time.time() - start
        print('Q: {}, T: {:6.6}, B: {}'.format(quantized, str(exec_time), bits))

# Tests the speed of the quantised all_reduce collective.
def all_reduce_centralised_speed(runs=100, size=32*2**5, quantized=False, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    op = ReduceOp.SUM
    if not quantized:
        bit_list = [32]
    else:
        bit_list = [1,2,4,8]
    for bits in bit_list:
        start = time.time()
        for _ in range(runs):
            if not quantized:
                dist.all_reduce(tensor1, op=op)
            else:
                comm.all_reduce_quantised_centralised(tensor1, op=op, bits=bits)
        exec_time = time.time() - start
        print('Q: {}, T: {:6.6}, B: {}'.format(quantized, str(exec_time), bits))

# Tests the speed of the quantised reduce collective.
def reduce_centralised_speed(runs=100, size=32*2**5, quantized=False, device=None):
    tensor1 = torch.zeros(size, device=device).normal_(mean=0,std=1)
    op = ReduceOp.SUM
    master = 0
    if not quantized:
        bit_list = [32]
    else:
        bit_list = [1,2,4,8]
    for bits in bit_list:
        start = time.time()
        for _ in range(runs):
            if not quantized:
                dist.reduce(tensor1, master, op=op)
            else:
                comm.reduce_quantised_centralised(tensor1, master, op=op, bits=bits)
        exec_time = time.time() - start
        print('Q: {}, T: {:6.6}, B: {}'.format(quantized, str(exec_time), bits))


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

    dist.init_process_group(backend, rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))

    parser = argparse.ArgumentParser(description='Run gpu quantized communication benchmarks')
    parser.add_argument('function', metavar='fn',
                    help='send, isend, all_gather, all_reduce, reduce')
    args = parser.parse_args()

    max_nodes = 2
    sizes = [16,18,20,22]
    # sizes = [1]

    if args.function == 'send':
        print("Send/Recv")
        for s in sizes:
            send_recv_speed(size=32*2**s, device=device)
            send_recv_speed(size=32*2**s, quantized=True, device=device)

    # print("ISend/IRecv")
    # for s in sizes:
    #     isend_irecv_speed(size=32*2**s, device=device)
    #     isend_irecv_speed(size=32*2**s, quantized=True, device=device)

    if args.function == 'all_gather':
        print("All Gather")
        for s in sizes:
            all_gather_speed(size=32*2**s, device=device)
            all_gather_speed(size=32*2**s, quantized=True, device=device)

    # NCCL does not implement the gather operation
    # gather_speed(device=device)
    # print("Gather correct")

    if args.function == 'all_reduce':
        print("All Reduce Centralised")
        for s in sizes:
            all_reduce_centralised_speed(size=32*2**s, device=device)
            all_reduce_centralised_speed(size=32*2**s, quantized=True, device=device)

    if args.function == 'reduce':
        print("All Reduce Centralised")
        for s in sizes:
            reduce_centralised_speed(size=32*2**s, device=device)
            reduce_centralised_speed(size=32*2**s, quantized=True, device=device)

if __name__ == '__main__':
    main()
