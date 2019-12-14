#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from quantizy import dataSz

"""
unsaturated ring all-reduce
"""
def ring_all_reduce(tensor):
    N = tensor.size()
    size = dist.get_world_size()
    rank = dist.get_rank()
    requests = []
    res = torch.zeros(N)
    for i in set(range(size)):
        if i != rank:
            send_req = dist.send(tensor=tensor, dst=i)
    for i in range(size):
        recv_buff = torch.zeros(N)
        if i != rank:
            dist.recv(tensor=recv_buff, src=i)
            res += recv_buff
    tensor[:] = recv_buff[:]

"""

"""
def ms_allreduce(tensor, quantize, unquantize):
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
            reqs += [dist.isend(tensor=quantize(tensor[i*chunksize:(i+1)*chunksize], 2), dst=i)] # K concurrent transfers
    
    recv = torch.zeros(arraySize // (dataSz * world), dtype=torch.int32)
    for i in range(world): # K steps
        if i != r:
            dist.recv(tensor=recv,src=i) # K / ??? values...
            acc[r*chunksize:(r+1)*chunksize] += unquantize(recv, 2)
    for req in reqs:
        req.wait()
    reqs = []
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            reqs += [dist.isend(tensor=quantize(acc[r*chunksize:(r+1)*chunksize], 2),dst=i)]
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            dist.recv(tensor=recv, src=i)
            acc[i*chunksize:(i+1)*chunksize] += unquantize(recv, 2)
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