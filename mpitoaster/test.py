import torch
import torch.distributed as dist

import os

if __name__ == '__main__':
    r = int(os.environ['RANK'])
    w = int(os.environ['WORLD_SIZE'])
    ma = os.environ['MASTER_ADDR']
    mp = os.environ['MASTER_PORT']

    print('ok')

    dist.init_process_group('nccl', rank=r, world_size=w, init_method='tcp://{}:60000'.format(ma))
    print('ini')
    tensor = (torch.ones(32) if r == 0 else torch.zeros(32)).cuda()
    print(tensor)
    dist.broadcast(tensor, src=0)
    torch.cuda.synchronize()
    print(tensor)
