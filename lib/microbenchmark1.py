import torch
import torch.distributed as dist
import os
import datetime

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
    tensor = torch.ones(8).cuda()
    rank = int(os.environ['RANK'])
    group = rendezvous(rank, 2)
    allreduce(tensor, group)
    print(tensor)

if __name__ == '__main__':
    main()