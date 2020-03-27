import torch
from .distributed_sgd import DistributedSGD
#from .all_reduce import allreduce, allreduce_quant
from .timer import CUDATimer
from .quantizy import quantize_gpu, unquantize_gpu, CompressedTensorBuffer
def toi1(self):
    q = quantize_gpu(self, 1)
    q.bits = 1
    return q

torch.Tensor.toi1 = toi1
torch.Tensor.toi2 = None
torch.Tensor.toi4 = None
torch.Tensor.toi8 = None