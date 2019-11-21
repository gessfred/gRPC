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
    unpacked = np.unpackbits(tensor.numpy().view(np.uint8))
    #tensor[...] = 1 stays 1
    unpacked[unpacked == 0] = -1
    return torch.from_numpy(unpacked).type(torch.float64)
def quantize_shrink(tensor):
    N = list(tensor.size())[0]
    print(N)
    #assert N % dataSz == 0
    N2 = N // dataSz
    res = torch.zeros(N2, dtype=int) 
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
def ms_allreduce(tensor, quantize=quantize, unquantize=unquantize):
    r = dist.get_rank()
    arraySize=list(tensor.size())[0]
    acc = torch.zeros(arraySize)
    chunksize = arraySize // dist.get_world_size()
    assert chunksize % dataSz == 0
    acc[r*chunksize:(r+1)*chunksize] = tensor[r*chunksize:(r+1)*chunksize]
    reqs = []
    #"Naive all-reduce"
    for i in range(dist.get_world_size()): # K steps
        if i != r:
            reqs += [dist.isend(tensor=quantize(tensor[i*chunksize:(i+1)*chunksize]), dst=i)] # K concurrent transfers
    for i in range(dist.get_world_size()): # K steps
        if i != r:
            recv = torch.zeros(arraySize, dtype=bool)
            dist.recv(tensor=recv[r*chunksize:(r+1)*chunksize],src=i) # K / ??? values...
            acc += unquantize(recv)
    for req in reqs:
        req.wait()
    reqs = []
    #"Naive all-gather"
    for i in range(dist.get_world_size()):
        if i != r:
            reqs += [dist.isend(tensor=quantize(acc[r*chunksize:(r+1)*chunksize]),dst=i)]
    #"Naive all-gather"
    for i in range(dist.get_world_size()):
        if i != r:
            recv = torch.zeros(arraySize, dtype=bool)
            dist.recv(tensor=recv[i*chunksize:(i+1)*chunksize], src=i)
            acc[i*chunksize:(i+1)*chunksize] += unquantize(recv[i*chunksize:(i+1)*chunksize])
    for req in reqs:
        req.wait()
    tensor[:] = acc[:]
def all_reduce(tensor): 
    allreduce(tensor.clone(), tensor)

""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
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
    
# Define the model
def model(x, w, b):
    return x @ w.t() + b

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

def batch_iter(y, tx, batch_size, num_batches=1):
    data_size = len(y)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def run(rank, size):
    group = dist.new_group(list(range(size)))
    time.sleep(10)
    for i in range(1000):
        ms_allreduce(tensor)

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

#https://stackoverflow.com/questions/54361763/pytorch-why-is-the-memory-occupied-by-the-tensor-variable-so-small
if __name__ == "__main__":
    """t = torch.ones(2**5)
    q1 = quantize_vector(t)
    q2 = quantize_shrink(t)
    print('shrink', unquantize_shrink(q2))
    print('vector', unquantize_vector(q1))
    print('t_size=', list(t.size())[0])
    print('q1_size=', list(q1.size())[0])
    print('q2_size=', list(q2.size())[0])
    #print(np.dtype(t.dtype).itemsize)
    print('Original: ', sys.getsizeof(t.storage()))
    print('Vector: ',sys.getsizeof(q1.storage()))
    print('Int: ',sys.getsizeof(q2.storage()))"""
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        print(p.pid)
        processes.append(p)

    for p in processes:
        p.join()
