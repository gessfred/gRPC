#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
from functools import reduce
import numpy as np
numberOfSamples = 1000
numberOfFeatures = 1

x = np.random.random(numberOfSamples).reshape(-1, 1)
#y = (np.random.random(numberOfSamples) > 0.5).astype(int)
x = torch.from_numpy(x).float()
y = x * 3 + 5 + 2 * torch.rand(numberOfSamples)

#targets = torch.from_numpy(targets)

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

def sgdFor(rank, size, group):
    def sgd(targets, inputs, batch_size, max_iter, λ=1e-2):
        losses = []
        w = torch.randn(1, numberOfFeatures, requires_grad=True)
        b = torch.randn(numberOfFeatures, requires_grad=True)
        acc_loss = 0
        i = 0
        for ybatch, xbatch in batch_iter(targets, inputs, batch_size, max_iter):
            preds = model(xbatch, w, b)
            loss = mse(preds, ybatch)
            print('epoch', i, " loss=", loss)
            loss.backward()
            with torch.no_grad():
                dist.all_reduce(w.grad, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(b.grad, op=dist.ReduceOp.SUM, group=group)
                w -= w.grad * λ
                b -= b.grad * λ
                w.grad.zero_()
                b.grad.zero_()
            i += 1
        return w, b
    return sgd


"""def train(inputs, targets, λ=1e-5):
    [l, h] = inputs.shape
    print(inputs.shape)
    # Weights and biases
    w = torch.randn(2, h, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    # Train for 100 epochs
    for i in range(100):
        preds = model(inputs, w, b)
        loss = mse(preds, targets)
        print('epoch', i, ' loss=', loss, 'accuracy=', np.sum((preds == targets).astype(int)))
        loss.backward()
        with torch.no_grad():
            w -= w.grad * λ
            b -= b.grad * λ
            w.grad.zero_()
            b.grad.zero_()
    return w, b"""


def run(rank, size):
    group = dist.new_group(list(range(size)))
    sgd = sgdFor(rank, size, group)
    assert numberOfSamples % size == 0
    C = int(numberOfSamples / size)
    f, t = rank*C, (rank+1)*C
    sgd(y[f:t], x[f:t], 5, 100)

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