import sys
sys.path.append('..')
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import os
import datetime
import time
from torch.multiprocessing import Process
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
                dist.send(tensor1, other)
                comm.send_quantized(tensor2, other, bits)
            else:
                tensor1 = torch.zeros(size, device=device)
                tensor2 = tensor1.clone()
                dist.recv(tensor1, other)
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
                dist.send(tensor1, other)
                h = comm.isend_quantized(tensor2, other, bits)
                h.wait()
            else:
                tensor1 = torch.zeros(size, device=device)
                tensor2 = tensor1.clone()
                dist.recv(tensor1, other)
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

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.random.manual_seed(rank) # Make very process have a different RNG
    # fn(runs=1,size=32)
    fn()

def init_processes(f, size):
    processes=[]
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, f))
        p.start()
        processes.append(p)
    return lambda: [p.join() for p in processes]

def main():

    max_nodes = 9
#
#     l = init_processes(send_recv_correctness,2)
#     l()
#     print("Send/Recv correct")
#
#     l = init_processes(isend_irecv_correctness,2)
#     l()
#     print("ISend/IRecv correct")
#
#     for nodes in range(2,max_nodes):
#         l = init_processes(all_gather_correctness,nodes)
#         l()
#         print("All Gather correct: {} nodes".format(nodes))
#
#     for nodes in range(2,max_nodes):
#         l = init_processes(gather_correctness,nodes)
#         l()
#         print("Gather correct: {} nodes".format(nodes))
#
#     for nodes in range(2,max_nodes):
#         l = init_processes(all_reduce_centralised_correctness,nodes)
#         l()
#         print("All Reduce Centralised correct: {} nodes".format(nodes))

    for nodes in range(2,max_nodes):
        l = init_processes(reduce_centralised_correctness,nodes)
        l()
        print("Reduce Centralised correct: {} nodes".format(nodes))

if __name__ == '__main__':
    main()
