"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
size = 2

""" All-Reduce example."""
def run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(10)
    if rank == 0:
        print(tensor * size)
    all_reduce(tensor)
    #dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor)

#pkill -f run.py

def quantize(tensor):
    N = list(tensor.size())[0]
    Q = torch.zeros(N, dtype=bool)
    Q = tensor > 0
    return Q
def unquantize(tensor):
    return tensor.type(torch.FloatTensor)

def all_reduce_1bit(tensor):
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
    error = torch.zeros(send.size())
    for i in range(size - 1):
        print('round', i)
        if i % 2 == 0:
            # Send send_buff
            G = quantize(send_buff + error)
            send_req = dist.isend(tensor=send_buff, dst=right)
            dist.recv(tensor=recv_buff, src=left)
            recv_buff = unquantize(recv_buff)
            #error = send_buff - recv_buff # send_buff or G?
            accum[:] += recv_buff[:] # + error?
        else:
            # Send recv_buff
            G = quantize(recv_buff + error)
            send_req = dist.isend(tensor=recv_buff, dst=right)
            dist.recv(tensor=send_buff, src=left)
            #send_buff = unquantize(send_buff)
            error = recv_buff - send_buff
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]
    #tensor = recv???

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


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

"""
inefficient implementation just to test
"""

def dumb_test():
    test = torch.rand(48) - 0.5
    print(test)
    #send 
    quantize(test) + error 
    print(unquantize(quantize(test)))
    error = (unquantize(quantize(test)) - test).abs()
    print(error)

if __name__ == "__main__":
    
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()