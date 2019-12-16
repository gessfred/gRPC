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
def ms_allreduce(tensor, quantize, unquantize, numberOfThreads=1):
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
            chunk = tensor[i*chunksize:(i+1)*chunksize]
            qchunk = quantize(chunk, numberOfThreads)
            reqs += [dist.isend(tensor=qchunk, dst=i)] # K concurrent transfers
    
    recv = torch.zeros(arraySize // (dataSz * world), dtype=torch.int32)
    for i in range(world): # K steps
        if i != r:
            dist.recv(tensor=recv,src=i) # K / ??? values...
            chunk = unquantize(recv, numberOfThreads)
            acc[r*chunksize:(r+1)*chunksize] += chunk
    for req in reqs:
        req.wait()
    reqs = []
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            chunk = acc[r*chunksize:(r+1)*chunksize]
            qchunk = quantize(chunk, numberOfThreads)
            reqs += [dist.isend(tensor=qchunk,dst=i)]
    #"Naive all-gather"
    for i in range(world):
        if i != r:
            dist.recv(tensor=recv, src=i)
            chunk = unquantize(recv, numberOfThreads)
            acc[i*chunksize:(i+1)*chunksize] += chunk
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

def allreduce(tensor):
    r = dist.get_rank()
    world = dist.get_world_size()
    peers = list(filter(lambda i: i != r, list(range(world))))
    sizeOfTensor=list(tensor.size())[0]
    chunksize = sizeOfTensor // world
    reqs = [dist.isend(tensor=tensor[i*chunksize:(i+1)*chunksize], dst=i) for i in peers] # K concurrent transfers
    recv = torch.zeros(sizeOfTensor // (world))
    for i in peers: # K steps
        dist.recv(tensor=recv,src=i) # K / ??? values...
        tensor[r*chunksize:(r+1)*chunksize] += recv[:]
    for req in reqs:
        req.wait()
    # we have to set to zero the values that we are not responsible (they will be included on their way back)
    tensor[0:r*chunksize] = 0
    tensor[(r+1)*chunksize:sizeOfTensor] = 0
    reqs = [dist.isend(tensor=tensor[r*chunksize:(r+1)*chunksize],dst=i) for i in peers]
    for i in peers:
        dist.recv(tensor=recv, src=i)
        tensor[i*chunksize:(i+1)*chunksize] += recv
    for req in reqs:
        req.wait()