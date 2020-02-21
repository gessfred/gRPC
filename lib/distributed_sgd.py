from torch.optim.optimizer import Optimizer, required
from torch.optim.sgd import SGD
import torch.distributed as dist
import os
import datetime
import random
from q_cpp import quantize_shrink, unquantize_shrink
from all_reduce import allreduce_quant

class DistributedSGD(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0,  weight_decay=0, nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.quantization_error = [None]*len(list(self.param_groups[0]['params']))
        self.rank = int(os.environ['RANK'])
        self.init(2)
        self.world = 2
        self.params = self.param_groups[0]['params']
        self.gpu = torch.device('cuda')
        self.cpu = torch.device('cpu')

    def init(self, world_size):
        dist.init_process_group('gloo', rank=self.rank, timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
        self.peers = list(filter(lambda x: x != self.rank, [0,1]))
        return dist.new_group(range(world_size))
    
    def step(self, closure=None):
        #average gradients
        for i, parameter in enumerate(self.params):
            parameter.grad.to(self.cpu)
            local = parameter.grad.clone()
            if self.quantization_error[i] is not None:
                parameter.grad += self.quantization_error[i]
            allreduce_quant(self.rank, self.world, self.peers, parameter.grad)
            parameter.grad /= self.world
            self.quantization_error[i] = local - parameter.grad
            parameter.grad.to(self.gpu)
        super().step(closure)