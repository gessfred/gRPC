import torch
import nccl
import os
if __name__ == '__main__':
    nccl.allreduce(os.environ['RANK'], 2)