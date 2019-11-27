#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
from functools import reduce
import numpy as np
import sys
import time
dataSz = 32
tensor = torch.zeros(2**10)
epochs = 1000
#targets = torch.from_numpy(targets)
def quantize(tensor):
    N = list(tensor.size())[0]
    Q = torch.zeros(N, dtype=bool)
    Q = tensor > 0
    return Q
def unquantize(tensor):
    tensor = tensor.type(torch.FloatTensor)
    tensor[tensor == 0] = -1
    return tensor # * data_scale

#https://stackoverflow.com/questions/49791312/numpy-packbits-pack-to-uint16-array
def quantize_vector(tensor):
    quantized = (tensor.numpy() > 0).astype(int)
    packed = np.packbits((quantized.reshape(-1, 4, 8)[:, ::-1]))
    return torch.from_numpy(packed.view(np.int32))

def unquantize_vector(tensor):
    unpacked = np.unpackbits(np.flip(tensor.numpy().view('uint8')))
    #tensor[...] = 1 stays 1
    
    unpacked = unpacked.astype(int)
    unpacked[unpacked == 0] = -1
    return torch.from_numpy(unpacked).type(torch.float64)

def quantize_shrink(tensor):
    N = list(tensor.size())[0]
    print(N)
    #assert N % dataSz == 0
    N2 = N // dataSz
    res = torch.zeros(N2, dtype=torch.int32) 
    for i in range(N2):
        x = 0
        for j in range(dataSz):
            x = x << 1
            z = tensor[dataSz*i + j]
            if z >= 0:
                z = 1
            else:
                z = 0
            x = x | z
        if x > 2147483647:
            x -= 2 ** 32
        res[i] = x
    return res

def unquantize_shrink(tensor):
    N2 = list(tensor.size())[0]
    N = N2 * dataSz
    res = torch.zeros(N, dtype=float)
    for i in range(N2):
        x = tensor[i]
        for j in range(dataSz):
            z = (x >> (dataSz - 1 - j)) & 1
            if z == 1:
                res[dataSz*i + j] = 1
            else:
                res[dataSz*i + j] = -1
    return res
"""
GPU i is responsible for chunk i
"""

def ms_allreduce(tensor, quantize=quantize_vector, unquantize=unquantize_vector):
    r = dist.get_rank()
    arraySize=list(tensor.size())[0]
    acc = torch.zeros(arraySize)
    world = dist.get_world_size()
    chunksize = arraySize // world
    assert chunksize % dataSz == 0
    acc[r*chunksize:(r+1)*chunksize] = tensor[r*chunksize:(r+1)*chunksize]
    reqs = []
    #"Naive all-reduce"
    #i = 0
    #print('actual: {} vs. expected: {}'.format(torch.zeros(int(arraySize / (chunksize * dataSz))).size(), quantize(tensor[i*chunksize:(i+1)*chunksize]).size()))
    for i in range(world): # K steps
        if i != r:
            reqs += [dist.isend(tensor=quantize(tensor[i*chunksize:(i+1)*chunksize]), dst=i)] # K concurrent transfers
    
    recv = torch.zeros(arraySize // (dataSz * world))
    for i in range(world): # K steps
        if i != r:
            dist.recv(tensor=recv,src=i) # K / ??? values...
            acc[r*chunksize:(r+1)*chunksize] += unquantize(recv)
    for req in reqs:
        req.wait()
    reqs = []
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            reqs += [dist.isend(tensor=quantize(acc[r*chunksize:(r+1)*chunksize]),dst=i)]
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            dist.recv(tensor=recv, src=i)
            acc[i*chunksize:(i+1)*chunksize] += unquantize(recv)
    for req in reqs:
        req.wait()
    tensor[:] = acc[:]

def ms_allreduce_un(tensor):
    r = dist.get_rank()
    arraySize=list(tensor.size())[0]
    acc = torch.zeros(arraySize)
    world = dist.get_world_size()
    chunksize = arraySize // world
    assert chunksize % dataSz == 0
    acc[r*chunksize:(r+1)*chunksize] = tensor[r*chunksize:(r+1)*chunksize]
    reqs = []
    #"Naive all-reduce"
    #i = 0
    #print('actual: {} vs. expected: {}'.format(torch.zeros(int(arraySize / (chunksize * dataSz))).size(), quantize(tensor[i*chunksize:(i+1)*chunksize]).size()))
    for i in range(world): # K steps
        if i != r:
            reqs += [dist.isend(tensor=(tensor[i*chunksize:(i+1)*chunksize]), dst=i)] # K concurrent transfers
    
    recv = torch.zeros(arraySize // (world))
    for i in range(world): # K steps
        if i != r:
            dist.recv(tensor=recv,src=i) # K / ??? values...
            acc[r*chunksize:(r+1)*chunksize] += (recv)
    for req in reqs:
        req.wait()
    reqs = []
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            reqs += [dist.isend(tensor=(acc[r*chunksize:(r+1)*chunksize]),dst=i)]
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            dist.recv(tensor=recv, src=i)
            acc[i*chunksize:(i+1)*chunksize] += (recv)
    for req in reqs:
        req.wait()
    tensor[:] = acc[:]



""" Implementation of a ring-reduce with addition. """
def allreduce(tensor):
    send = tensor.clone()
    recv = tensor
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = torch.zeros(send.size())
    recv_buff = torch.zeros(send.size())
    accum = torch.zeros(send.size())
    accum[:] = send[:]

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size
    send_buff[:] = send[:]
    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(tensor=send_buff, dst=right)
            dist.recv(tensor=recv_buff, src=left)
            accum[:] += recv_buff[:]
        else:
            # Send recv_buff
            send_req = dist.isend(tensor=recv_buff, dst=right)
            dist.recv(tensor=send_buff, src=left)
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]

def run(rank, size):
    group = dist.new_group(list(range(size)))
    for i in range(epochs):
        ms_allreduce(tensor)

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    ip = '10.90.38.6'
    os.environ['MASTER_ADDR'] = ip#'192.168.64.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'ens786f0'
    time.sleep(10)# to hide the rendez-vous from profiling 
    dist.init_process_group(backend, rank=rank, world_size=size, init_method='tcp://{}:23456'.format(ip))

    fn(rank, size)

#https://stackoverflow.com/questions/54361763/pytorch-why-is-the-memory-occupied-by-the-tensor-variable-so-small
if __name__ == "__main__":
    
    p = Process(target=init_processes, args=(int(sys.argv[1]), 2, run))
    p.start()
    print(p.pid)
    p.join()

