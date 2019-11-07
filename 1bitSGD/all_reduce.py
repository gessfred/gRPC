import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import cProfile

def quantize(tensor):
    N = list(tensor.size())[0]
    Q = torch.zeros(N, dtype=bool)
    Q = tensor > 0
    return Q
    
def unquantize(tensor):
    tensor = tensor.type(torch.FloatTensor)
    tensor[tensor == 0] = -1
    return tensor
