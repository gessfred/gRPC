"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile
size = 4

tensor = torch.ones(4)#(torch.rand(4) - 0.5) 

""" All-Reduce example."""
def run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group(list(range(size)))
    if rank == 0:
        print('Goal',tensor*size)
    ms_allreduce(tensor)
    #dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor)

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

def all_reduce_1bit2(tensor):
    N = tensor.size()
    send = tensor.clone()
    recv = tensor
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = torch.zeros(N)
    recv_buff = torch.zeros(N)
    accum = torch.zeros(N)
    accum[:] = send[:]
    R = torch.zeros(N, dtype=bool)
    left = ((rank - 1) + size) % size
    right = (rank + 1) % size
    send_buff[:] = send[:]
    error = torch.zeros(N)
    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            G = quantize(send_buff)
            send_req = dist.isend(tensor=G, dst=right)
            dist.recv(tensor=R, src=left)
            recv_buff = unquantize(R)
            error = send_buff - recv_buff # send_buff or G?
            #print('error={}'.format(error))
            accum[:] += recv_buff[:] # + error?
        else:
            # Send recv_buff
            G = quantize(recv_buff + error)
            send_req = dist.isend(tensor=G, dst=right)
            dist.recv(tensor=R, src=left)
            send_buff = unquantize(R)
            error = recv_buff - send_buff
            #print('error={}'.format(error))
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]


def all_reduce_1bit(tensor):
    N = tensor.size()
    send = tensor.clone()
    recv = tensor
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = torch.zeros(N)
    recv_buff = torch.zeros(N)
    accum = torch.zeros(N)
    accum[:] = send[:]
    R = torch.zeros(N, dtype=bool)
    left = ((rank - 1) + size) % size
    right = (rank + 1) % size
    send_buff[:] = send[:]
    error = torch.zeros(N)
    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            G = quantize(send_buff + error)
            send_req = dist.isend(tensor=G, dst=right)
            dist.recv(tensor=R, src=left)
            recv_buff = unquantize(R)
            error = send_buff - recv_buff # send_buff or G?
            #print('error={}'.format(error))
            accum[:] += recv_buff[:] # + error?
        else:
            # Send recv_buff
            G = quantize(recv_buff + error)
            send_req = dist.isend(tensor=G, dst=right)
            dist.recv(tensor=R, src=left)
            send_buff = unquantize(R)
            error = recv_buff - send_buff
            #print('error={}'.format(error))
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]
#pkill python
"""
GPU i is responsible for chunk i
"""
def ms_allreduce(tensor):
    r = dist.get_rank()
    chunk = r
    acc = torch.zeros(4)
    acc[r] = tensor[r]
    """for i in range(dist.get_world_size()):
        if i != dist.get_rank():
            req = dist.isend(tensor=tensor[0], dst=i)"""
    #"Naive all-reduce"
    for i in range(dist.get_world_size()):
        if i != r:
            req = dist.isend(tensor=tensor[i], dst=i)
            recv = torch.zeros(4)
            dist.recv(tensor=recv[r],src=i)
            acc += recv
            req.wait()
    #"Naive all-gather"
    for i in range(dist.get_world_size()):
        if i != r:
            req = dist.isend(tensor=acc[r],dst=i)
            recv = torch.zeros(4)
            dist.recv(tensor=recv[i], src=i)
            acc += recv
            req.wait()
    tensor[:] = acc[:] #print('rank', r, 'has', acc)
    """
    xchg
    recv = torch.zeros(4)
    if dist.get_rank() == 0:
        req = dist.isend(tensor=tensor[0], dst=1)
        dist.recv(tensor=recv[0], src=1)
        acc += recv
    if dist.get_rank() == 1:
        req = dist.isend(tensor=tensor[1],dst=0)
        dist.recv(tensor=recv[0], src=0)
        acc += recv
    print(dist.get_rank(), 'has', acc)"""
    """if dist.get_rank() == 0:
        req = dist.isend(tensor=tensor[0], dst=1)
        req2 = dist.isend(tensor=tensor[1], dst=2)
        req.wait()
        req2.wait()
        print('ok0')
    if dist.get_rank() == 2:
        recv = torch.zeros(4)
        dist.recv(tensor=recv[0], src=0)
        acc += recv
        print('rank2 has ', acc)
    if dist.get_rank() == 1:
        recv = torch.zeros(4)
        dist.recv(tensor=recv[0], src=0)
        acc += recv
        print('rank1 has ', acc)"""

    """send = tensor.clone()
    recv = torch.zeros(4)
    
    reqs = []
    for i in range(dist.get_world_size()):
        if i != dist.get_rank():
            dist.isend(tensor=tensor,dst=i)
    for req in reqs:
        req.wait()
    for i in range(dist.get_world_size()):
        if i != dist.get_rank():
            rec = torch.zeros(4)
            dist.recv(tensor=rec, src=i)"""

    """rank = dist.get_rank()
    for i in range(dist.get_world_size()):
        if i != dist.get_rank():
            req = dist.recv(tensor=recv, src=i)
            req.wait()
            acc[rank] += recv[0]"""
    #print(rank, 'has', acc)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()