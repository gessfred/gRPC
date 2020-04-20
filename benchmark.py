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

def ar(tensor, timer):
        with timer('packing'):
                compressed, padding = _pack(tensor)
        gather_list = [compressed.clone() for i in range(dist.get_world_size())]
        with timer('gather'):
                dist.all_gather(gather_list, compressed)
        with timer('unpack'):
                gather_list = list(map(lambda recv: _unpack(recv, padding), gather_list))
        with timer('sum'): 
                return torch.sum(torch.stack(gather_list), dim=0)

if __name__ == '__main__': 
        dist.init_process_group('nccl', rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']), init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
        print('init!')
        torch.cuda.synchronize()
        dev = os.environ['LOCAL_RANK']
        for sz in range(18, 25):
                timer = CUDATimer('allreduce-may')
                tensor = torch.ones(2**sz, device='cuda:{}'.format(dev))
                for _ in range(100):
                        ar(tensor, timer)
                        torch.cuda.synchronize()
                print('==========', sz, '==========')
                timer.dump()
                timer.upload_raw('microbenchmarking', {'version': 'sign.allreduce', 'input': 2**sz, 'device': 'cuda:{}'.format(dev)})                
#print(end-start, 'ms for', sz)
        #timer.upload_raw('microbenchmarking', {'version': 'pytorch.distributed', 'input': 2**sz})
