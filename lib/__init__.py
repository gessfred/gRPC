import torch
from .distributed_sgd import DistributedSGD
from .all_reduce import allreduce, allreduce_quant
from .timer import CUDATimer
from .quantizy import quantize_gpu, unquantize_gpu
def toi1(self):
    q = quantize_gpu(self, 1, torch.cuda.current_device())
    q.bits = 1
    return q

def tof32(self):
    unquantize_gpu(self, self.bits, torch.cuda.current_device())

torch.Tensor.toi1 = toi1
torch.Tensor.toi2 = None
torch.Tensor.toi4 = None
torch.Tensor.toi8 = None
torch.Tensor.tof32 = tof32