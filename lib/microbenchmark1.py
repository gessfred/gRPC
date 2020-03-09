import torch
import torch.distributed as dist
import os
import datetime
import time
from contextlib import contextmanager
import numpy as np


def allreduce(tensor, group):
    torch.cuda.synchronize()
    rank = dist.get_rank()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    chunks = list(tensor.view(dist.get_world_size(), -1))
    start.record()
    for i, chunk in enumerate(chunks):
        dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    end.record()
    chunk = chunks[rank]
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start1.record()
    dist.all_gather(chunks, chunk, group=group)
    end1.record()
    torch.cuda.synchronize()
    print('{} | {}'.format(start.elapsed_time(end), start1.elapsed_time(end1)))

def rendezvous(rank, world_size):
    dist.init_process_group('nccl', rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
    return dist.new_group(range(world_size))

def main():
    tensor = torch.ones(8).cuda()
    rank = int(os.environ['RANK'])
    group = rendezvous(rank, 2)
    start = time.time()
    allreduce(tensor, group)
    print('exec time: {}'.format(time.time() - start))
    print(tensor)

if __name__ == '__main__':
    main()