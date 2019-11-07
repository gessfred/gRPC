"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
from functools import reduce
numberOfNodes = 4
arraySize = 4
tensors = [(torch.rand(arraySize) - 0.5) for i in range(numberOfNodes)]
assert arraySize % numberOfNodes == 0
""" All-Reduce example."""
def run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group(list(range(numberOfNodes)))
    goal = reduce(lambda x,y: x + y, tensors)
    if rank == 0:
        print('Goal',goal)
    ms_allreduce(tensors[rank], chunksize=int(arraySize/numberOfNodes))
    #dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensors[rank])

#pkill -f run.py

def quantize(tensor):
    N = list(tensor.size())[0]
    Q = torch.zeros(N, dtype=bool)
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
    print(rank, 'has', acc)
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

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    
    processes = []
    for rank in range(numberOfNodes):
        p = Process(target=init_process, args=(rank, numberOfNodes, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()