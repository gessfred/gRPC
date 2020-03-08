import torch
import nccl
import os
if __name__ == '__main__':
    uuid = nccl.get_local_id()
    print(''.join(uuid).encode('utf-8'))
    rank = int(os.environ['RANK'])
    nccl.init(rank, 2, uuid, (rank + 1) % 2)
    #nccl.allreduce(int(os.environ['RANK']), 2, uuid)
