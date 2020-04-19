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
from lib import CUDATimer, quantize_gpu as _pack2, unquantize_gpu as _unpack2
from math import ceil
import bit2byte
def element_num(size):
	num = 1
	for i in range(len(size)):
	    num *= size[i]
	return num
def _pack(src_tensor):
        dev = src_tensor.device
        src_tensor = torch.sign(src_tensor)
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=dev)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32,-1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        return dst_tensor, src_tensor_size

def _unpack(src_tensor, src_tensor_size):
        dev = src_tensor.device
        src_element_num = element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(src_element_num + add_elm, device=dev, dtype=torch.int32)
        new_tensor = new_tensor.view(32,-1)
        new_tensor = bit2byte.unpacking(src_tensor,new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = - new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        return new_tensor

def gather_quantized(tensor, gather_list=None, bits=1, dst=0):
        quantized, padding = _pack(tensor, bits)
        print('quant9zed', quantized)
        tensor_sizes = [t.view(-1).shape[0] for t in gather_list]
        padding_list = [(32 - s) % 32 for s in tensor_sizes]
        quantized_sizes = [ceil(s/(32/bits)) for s in tensor_sizes]
        quantized_list = [torch.empty(s, dtype=tensor.dtype, device=tensor.device) for s in quantized_sizes]
        dist.all_gather(quantized_list, quantized)
        if dist.get_rank() == dst:
                for t, q, p in zip(gather_list, quantized_list, padding_list):
                        print(_unpack(q, p, bits))
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
        print(tensor_list)
        # broadcasting non quantized tensor
        with timer('bcast'):
                buffer, padding = _pack(tensor, 2)
                dist.broadcast(buffer, master)
                tensor = torch.sign(_unpack(buffer, padding, 2) - 0.5)
                print(tensor)

def ar(tensor, timer):
        with timer('packing'):
                compressed, padding = _pack(tensor)
        gather_list = [compressed.clone() for i in range(dist.get_world_size())]
        with timer('gather'):
                dist.all_gather(gather_list, compressed)
        with timer('unpack'):
                gather_list = list(map(lambda recv: _unpack(recv, padding), gather_list))
        with timer('sum'):        
                tensor[:] = torch.sum(torch.cat(gather_list)) / dist.get_world_size()

def ar2(tensor, timer):
        with timer('packing'):
                compressed, padding = _pack2(tensor, 1)
        gather_list = [compressed.clone() for i in range(dist.get_world_size())]
        with timer('gather'):
                dist.all_gather(gather_list, compressed)
        with timer('unpack'):
                gather_list = list(map(lambda recv: _unpack2(recv, padding, 1), gather_list))
        with timer('sum'):        
                tensor[:] = torch.sum(torch.cat(gather_list)) / dist.get_world_size()

if __name__ == '__main__': 
        dist.init_process_group('nccl', rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']), init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
        print('init!')
        torch.cuda.synchronize()
        timer = CUDATimer('test-correctness-timing')
        f = os.environ['function']
        s = int(os.environ['tensor-size'])
        allreduce = lambda tensor: ar(tensor, timer) if f == 'bit2byte' else dist.all_reduce if f == 'dist' else lambda tensor: ar2(tensor, timer)
        start = time.perf_counter()
        tensor = torch.ones(2**s).cuda()
        shots = 100
        for _ in range(shots):
                allreduce(tensor)
                torch.cuda.synchronize()
        end = time.perf_counter()
        print(end-start)
        timer.upload_raw('microbenchmarking', {'function': f, 'inputsize': s, 'shots': shots})
