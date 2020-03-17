import torch
from .distributed_sgd import DistributedSGD
from .all_reduce import allreduce, allreduce_quant
from .timer import CUDATimer
torch.Tensor.toi1 = None
torch.Tensor.toi2 = None
torch.Tensor.toi4 = None
torch.Tensor.toi8 = None
torch.Tensor.tof32 = None