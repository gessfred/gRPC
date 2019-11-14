#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
from functools import reduce
import numpy as np

numberOfSamples = 32*1000
numberOfFeatures = 1

x = torch.rand(numberOfSamples) * 2 - 1
y = x * 3 + 5 + 2 * torch.rand(numberOfSamples) # 3x + 6 + noise (-1, 1)

#targets = torch.from_numpy(targets)
def quantize(tensor):
    Q = tensor > 0
    return Q

def unquantize(tensor):
    tensor = tensor.type(torch.FloatTensor)
    tensor[tensor == 0] = -1
    return tensor # * data_scale

"""
GPU i is responsible for chunk i
"""
def ms_allreduce(tensor, chunksize=1):
    r = dist.get_rank()
    arraySize=tensor.size()
    acc = torch.zeros(arraySize)
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
    #print(rank, 'has', acc)
    #"Naive all-gather"
    for i in range(dist.get_world_size()):
        if i != r:
            reqs += [dist.isend(tensor=quantize(acc[r*chunksize:(r+1)*chunksize]),dst=i)]
    #"Naive all-gather"
    for i in range(dist.get_world_size()):
        if i != r:
            recv = torch.zeros(arraySize, dtype=bool)
            dist.recv(tensor=recv[i*chunksize:(i+1)*chunksize], src=i)
            acc[i*chunksize:(i+1)*chunksize] = unquantize(recv[i*chunksize:(i+1)*chunksize])
    for req in reqs:
        req.wait()
    tensor[:] = acc[:]

def all_reduce(tensor):
    #allreduce(tensor.clone(), tensor)
    ms_allreduce(tensor)

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
    x = x.reshape(-1,1)
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

def sgdFor(rank, size, group):
    def sgd(targets, inputs, batch_size, max_iter, λ=1e-2):
        losses = []
        w = torch.randn(1, numberOfFeatures, requires_grad=True)
        b = torch.randn(numberOfFeatures, requires_grad=True)
        acc_loss = 0
        i = 0
        for i in range(max_iter):
            for ybatch, xbatch in batch_iter(targets, inputs, batch_size, 1):
                preds = model(xbatch, w, b)
                loss = mse(preds, ybatch)
                print('epoch', i, " loss=", loss)
                loss.backward()
                with torch.no_grad():
                    all_reduce(w.grad)
                    all_reduce(b.grad)
                    w -= w.grad * λ
                    b -= b.grad * λ
                    w.grad.zero_()
                    b.grad.zero_()
        return w, b
    return sgd

def run(rank, size):
    group = dist.new_group(list(range(size)))
    sgd = sgdFor(rank, size, group)
    assert numberOfSamples % size == 0
    C = int(numberOfSamples / size)
    f, t = rank*C, (rank+1)*C
    sgd(y[f:t], x[f:t], 16, 100)

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
