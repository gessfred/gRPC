import sys
sys.path.append('..')
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
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

# Tests the correctness of the quantised send/recv primitives.
# Assumes exactly 2 nodes
def send_recv_correctness(runs=100, size=32*2**5, device=None):
    for bits in [1,2,4,8]:
        for _ in range(runs):
            rank = dist.get_rank()
            other = (rank + 1) % 2
        # The smallest rank is the sender
            if rank < other:
                tensor1 = torch.empty(size, device=device).normal_(mean=0,std=1)
                tensor2 = tensor1.clone()
                comm.send(tensor1, other)
                comm.send_quantized(tensor2, other, bits)
            else:
                tensor1 = torch.zeros(size, device=device)
                tensor2 = tensor1.clone()
                comm.recv(tensor1, other)
                comm.recv_quantized(tensor2, other, bits)

                q1, p1 = quantize_gpu(tensor1, bits)
                tensor1 = unquantize_gpu(q1, p1, bits)
                if not (tensor1 == tensor2).all():
                    print('bits '+str(bits))
                    print(tensor1)
                    print(tensor2)
                    assert(False)

# Tests the correctness of the quantised isend/irecv primitives.
# Assumes exactly 2 nodes
def isend_irecv_correctness(runs=100, size=32*2**5, device=None):
    for bits in [1,2,4,8]:
        for _ in range(runs):
            rank = dist.get_rank()
            other = (rank + 1) % 2
        # The smallest rank is the sender
            if rank < other:
                tensor1 = torch.empty(size, device=device).normal_(mean=0,std=1)
                tensor2 = tensor1.clone()
                comm.send(tensor1, other)
                h = comm.isend_quantized(tensor2, other, bits)
                h.wait()
            else:
                tensor1 = torch.zeros(size, device=device)
                tensor2 = tensor1.clone()
                comm.recv(tensor1, other)
                h = comm.irecv_quantized(tensor2, other, bits)

                q1, p1 = quantize_gpu(tensor1, bits)
                tensor1 = unquantize_gpu(q1, p1, bits)
                h.wait()
                if not (tensor1 == tensor2).all():
                    print('bits '+str(bits))
                    print(tensor1)
                    print(tensor2)
                    assert(False)

# Tests the correctness of the quantised all_gather collective.
def all_gather_correctness(runs=100, size=32*2**5, device=None):
    for bits in [1,2,4,8]:
        for _ in range(runs):

            rank = dist.get_rank()
            N = dist.get_world_size()
            tensor_list1 = [torch.empty(size, device=device) for _ in range(N)]
            tensor_list2 = [torch.empty(size, device=device) for _ in range(N)]
            tensor1 = torch.empty(size, device=device).normal_(mean=0,std=1)
            tensor2 = tensor1.clone()
            dist.all_gather(tensor_list1, tensor1)
            comm.all_gather_quantized(tensor_list2, tensor2, bits)

            l = [quantize_gpu(t, bits) for t in tensor_list1]
            tensor_list1 = [unquantize_gpu(q,p,bits) for q,p in l]
            if not (torch.stack(tensor_list1) == torch.stack(tensor_list2)).all():
                print('bits '+str(bits))
                torch.set_printoptions(profile="full")
                print(str(rank) + ' t1 '+ str(tensor_list1))
                print(str(rank) + ' t2 '+ str(tensor_list2))
                assert(False)

# Tests the correctness of the quantised gather collective.
def gather_correctness(runs=100, size=32*2**5, device=None):
    for bits in [1,2,4,8]:
        for _ in range(runs):

            rank = dist.get_rank()
            N = dist.get_world_size()
            master = N-1

            if rank == master:
                tensor_list1 = [torch.empty(size, device=device) for _ in range(N)]
                tensor_list2 = [torch.empty(size, device=device) for _ in range(N)]
            else:
                tensor_list1 = None
                tensor_list2 = None
            tensor1 = torch.empty(size, device=device).normal_(mean=0,std=1)
            tensor2 = tensor1.clone()
            dist.gather(tensor1, gather_list=tensor_list1, dst=master)
            comm.gather_quantized(tensor2, gather_list=tensor_list2, bits=bits, dst=master)

            if rank == master:
                l = [quantize_gpu(t, bits) for t in tensor_list1]
                tensor_list1 = [unquantize_gpu(q,p,bits) for q,p in l]
                if not (torch.stack(tensor_list1) == torch.stack(tensor_list2)).all():
                    print('bits '+str(bits))
                    torch.set_printoptions(profile="full")
                    print(str(rank) + ' t1 '+ str(tensor_list1))
                    print(str(rank) + ' t2 '+ str(tensor_list2))
                    assert(False)

# Tests the correctness of the quantised all_reduce collective.
def all_reduce_centralised_correctness(runs=100, size=32*2**5, device=None):

    epsilon = 0.00005

    for op in [ReduceOp.SUM, ReduceOp.PRODUCT]:
        for bits in [1,2,4,8]:
            for _ in range(runs):

                rank = dist.get_rank()

                tensor1 = torch.empty(size, device=device).normal_(mean=0,std=1)
                tensor2 = tensor1.clone()
                q, p = quantize_gpu(tensor1, bits)
                tensor1 = unquantize_gpu(q, p, bits)
                dist.all_reduce(tensor1, op=op)
                comm.all_reduce_quantised_centralised(tensor2, op=op, bits=bits)

                if not ((tensor1 - tensor2).abs() < epsilon).all():
                    print('bits '+str(bits))
                    print('op '+str(op))
                    torch.set_printoptions(profile="full")
                    index = torch.eq(tensor1, tensor2).logical_not().nonzero()
                    print(str(rank) + ' t1 '+ str(tensor1[index]))
                    print(str(rank) + ' t2 '+ str(tensor2[index]))
                    assert(False)

# Tests the correctness of the quantised reduce collective.
def reduce_centralised_correctness(runs=100, size=32*2**5, device=None):

    epsilon = 0.00005

    for op in [ReduceOp.SUM, ReduceOp.PRODUCT]:
        for bits in [1,2,4,8]:
            for _ in range(runs):

                rank = dist.get_rank()
                N = dist.get_world_size()
                master = N-1

                tensor1 = torch.empty(size, device=device).normal_(mean=0,std=1)
                tensor2 = tensor1.clone()
                q, p = quantize_gpu(tensor1, bits)
                tensor1 = unquantize_gpu(q, p, bits)
                dist.reduce(tensor1, master, op=op)
                comm.reduce_quantised_centralised(tensor2, master, op=op, bits=bits)

                if rank == master:
                    if not ((tensor1 - tensor2).abs() < epsilon).all():
                        print('bits '+str(bits))
                        print('op '+str(op))
                        torch.set_printoptions(profile="full")
                        print(str(rank) + ' t1 '+ str(tensor1))
                        print(str(rank) + ' t2 '+ str(tensor2))
                        assert(False)


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

    max_nodes = 2

    send_recv_correctness(device=device)
    print("Send/Recv correct")

    isend_irecv_correctness(device=device)
    print("ISend/IRecv correct")

    all_gather_correctness(device=device)
    print("All Gather correct")

    # NCCL does not implement the gather operation
    # gather_correctness(device=device)
    # print("Gather correct")

    all_reduce_centralised_correctness(device=device)
    print("All Reduce Centralised correct")

    reduce_centralised_correctness(device=device)
    print("Reduce Centralised correct")

if __name__ == '__main__':
    main()
