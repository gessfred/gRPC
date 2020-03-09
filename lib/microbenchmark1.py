import torch
import torch.distributed as dist
import os
import datetime
import time
from contextlib import contextmanager
import numpy as np

class Timer(object):
    def __init__(self):
        super().__init__()
        self.profile = {}
    @contextmanager
    def __call__(self, label):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self.profile[label] = start.elapsed_time(end)
    def dump():
        print(self.profile)

timer = Timer()

def allreduce(tensor, group):
    torch.cuda.synchronize()
    with timer('reduce'):
        rank = dist.get_rank()
        chunks = list(tensor.view(dist.get_world_size(), -1))
        for i, chunk in enumerate(chunks):
            dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    with timer('all_gather'):
        chunk = chunks[rank]
        dist.all_gather(chunks, chunk, group=group)
    torch.cuda.synchronize()
    print('{} | {}'.format(start.elapsed_time(end), start1.elapsed_time(end1)))

def rendezvous(rank, world_size):
    dist.init_process_group('nccl', rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
    return dist.new_group(range(world_size))

def main():
    tensor = torch.ones(2**20).cuda()
    rank = int(os.environ['RANK'])
    group = rendezvous(rank, 2)
    start = time.time()
    with timer('all_reduce'):
        allreduce(tensor, group)
    timer.dump()
    print('exec time: {}'.format(time.time() - start))
    print(tensor)

if __name__ == '__main__':
    main()