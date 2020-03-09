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
        self.timestamps = {}
        self.events = {}
        self.start = time.time()
        self.elapsed_time = 0
        self.closed = False

    @contextmanager
    def __call__(self, label):
        start = self.record(label+'_start')
        yield
        end = self.record(label+'_end')
        self.events['label'] = [start, end]

    def record(self, label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.timestamps[label] = time.monotonic()
        return event

    def dump(self):
        if not self.closed:
            self.close()
        print('--------------------timeline--------------------')
        print('profile: {}'.format(self.profile))
        print('events: {}'.format(self.events))
        print('timeline: {}'.format(self.timestamps))
        print('elapsed_time: {}'.format(self.elapsed_time))
        print('------------------------------------------------')

    def close(self):
        self.closed = True
        self.elapsed_time = time.time() - self.start

def allreduce_(timer, tensor, group):
    with timer('reduce'):
        rank = dist.get_rank()
        chunks = list(tensor.view(dist.get_world_size(), -1))
        for i, chunk in enumerate(chunks):
            dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    with timer('all_gather'):
        chunk = chunks[rank]
        dist.all_gather(chunks, chunk, group=group)

def allreduce__(timer, tensor, group):
    with timer('reduce'):
        rank = dist.get_rank()
        chunks = list(tensor.view(dist.get_world_size(), -1))
        for i, chunk in enumerate(chunks):
            dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    with timer('all_gather'):
        chunk = chunks[rank]
        dist.all_gather(chunks, chunk, group=group)

def allreduce(tensor, group):
    rank = dist.get_rank()
    chunks = list(tensor.view(dist.get_world_size(), -1))
    for i, chunk in enumerate(chunks):
        dist.reduce(chunk, i, op=dist.ReduceOp.SUM, group=group)
    chunk = chunks[rank]
    dist.all_gather(chunks, chunk, group=group)

def rendezvous(rank, world_size):
    dist.init_process_group('nccl', rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
    return dist.new_group(range(world_size))

def main():
    tensor = torch.ones(2**30).cuda()
    rank = int(os.environ['RANK'])
    group = rendezvous(rank, 2)
    allreduce(tensor, group)
    t1 = Timer()
    with t1('all_reduce_bare'):
        allreduce(tensor, group)
    t1.dump()
    tensor = torch.ones(2**20).cuda()
    t2 = Timer()
    with t2('all_reduce_precise'):
        allreduce_(t2, tensor, group)
    t2.dump()
    tensor = torch.ones(2**20).cuda()
    t3 = Timer()
    with t3('all_reduce_compressed'):
        allreduce__(t3, tensor, group)
    t3.dump()
    tensor = torch.ones(2**20).cuda()
    t4 = Timer()
    with t4('all_reduce_baseline'):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    t4.dump()
    print(tensor)

if __name__ == '__main__':
    main()