import torch
import nccl

if __name__ == '__main__':
    nccl.allreduce()