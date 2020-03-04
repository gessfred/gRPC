import torch
import nccl
import os
if __name__ == '__main__':
    uuid = nccl.get_local_id()
    print(''.join(uuid).encode('utf-8'))
    nccl.allreduce(int(os.environ['RANK']), 2, uuid)
