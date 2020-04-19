import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import argparse
import os
import datetime
import time
from torch.multiprocessing import Process, spawn
import cProfile
import numpy as np
from lib import CUDATimer, quantize_gpu as _pack, unquantize_gpu as _unpack

from math import ceil

def gather_quantized(tensor, gather_list=None, bits=1, dst=0):
        quantized, padding = _pack(tensor, bits)
        tensor_sizes = [t.view(-1).shape[0] for t in gather_list]
        padding_list = [(32 - s) % 32 for s in tensor_sizes]
        quantized_sizes = [ceil(s/(32/bits)) for s in tensor_sizes]
        quantized_list = [torch.empty(s, dtype=tensor.dtype, device=tensor.device) for s in quantized_sizes]
        dist.all_gather(quantized_list, quantized)
        if dist.get_rank() == dst:
                for t, q, p in zip(gather_list, quantized_list, padding_list):
                        t.copy_(_unpack(q, p, bits))

def all_reduce_quantised_centralised(tensor, timer, master=0, op=ReduceOp.SUM, bits=1):
        #gather tensors on master node
        rank = dist.get_rank()
        tensor_list = [torch.empty(tensor.shape, device=tensor.device) for _ in range(dist.get_world_size())]
        with timer('gather'):
                gather_quantized(tensor, gather_list=tensor_list, bits=bits, dst=master)
        # reduce tensors on master node, as gather is synchronous we know the tensor list is ready
        if rank == master:
                ops = {ReduceOp.SUM: lambda t_l: torch.sum(t_l, dim=0),
                           ReduceOp.PRODUCT: lambda t_l: torch.prod(t_l, dim=0)}
                tensor.copy_(ops[op](torch.stack(tensor_list)))
        # broadcasting non quantized tensor
        with timer('bcast'):
                buffer, padding = _pack(tensor, 2)
                dist.broadcast(buffer, master)
                tensor = torch.sign(_unpack(buffer, padding, 2) - 0.5)
		print(tensor)

if __name__ == '__main__': 
        dist.init_process_group('nccl', rank=int(os.environ['RANK']), world_size=2, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
        print('init!')
        tensor = torch.ones(2**24).cuda()
        #dist.all_reduce(tensor)
        torch.cuda.synchronize()
        timer = CUDATimer('test-correctness-timing')
        start = time.perf_counter()
        for _ in range(100):
                #dist.all_reduce(tensor)
                all_reduce_quantised_centralised(tensor, timer)
                torch.cuda.synchronize()
        end = time.perf_counter()
        print(end-start)
        timer.dump()
