import torch
import torch.distributed as dist
import os

if __name__ == '__main__':
    IP = os.environ['MASTER_ADDR']
    os.environ['MASTER_PORT'] = os.environ['PYTORCH_PORT']#'29500'
    os.environ['GLOO_SOCKET_IFNAME'] = os.environ['PYTORCH_SOCKET']
    rank = int(os.environ['RANK'])
    dist.init_process_group('nccl', 
        rank=rank, 
        timeout=datetime.timedelta(seconds=10), 
        world_size=int(os.environ['WORLD']), 
        init_method='tcp://{}:60000'.format(IP))
    group = dist.new_group(range(int(os.environ['WORLD'])))
    tensor = torch.randn(32)
    print(tensor)
    if rank == 0:
        dist.broadcast(tensor, 0, group=dist.new_group([0, 2]))
    if rank == 2:
        dist.broadcast(tensor, 0, group=dist.new_group([0, 2]))
    print(tensor)
    dist.barrier()
    