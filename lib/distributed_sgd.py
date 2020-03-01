import torch
from torch.optim.optimizer import Optimizer, required
from torch.optim.sgd import SGD
import torch.distributed as dist
import os
import datetime
import random
import time
from q_cpp import quantize_shrink, unquantize_shrink
from all_reduce import allreduce_quant
import subprocess

class DistributedSGD(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0,  weight_decay=0, nesterov=False, dtype='32bit', backend='gloo'):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.quantization_error = [None]*len(list(self.param_groups[0]['params']))
        self.rank = int(os.environ['RANK'])
        self.group = self.rendezvous(2)
        self.world = 2
        self.backend = backend
        self.params = self.param_groups[0]['params']
        self.gpu = torch.device('cuda')
        self.cpu = torch.device('cpu')
        self.ping()
        if dtype == '1bit':
            self.step = self.quantized_step
        elif dtype == '32bit':
            self.step = self.step_
        #setup pyflame
        subprocess.Popen(['pyflame', '--pid={}'.format(os.getpid()), '--output=/mnt/data/test.svg'])
        self.profile = {'transfer': 0.0, 'communication': 0.0, 'packing': 0.0, 'computation': 0.0, 'total': 0.0}

    def ping(self):
        rank = self.rank
        neighbour = (rank + 1) % 2
        req = dist.isend(torch.ones(1), dst=neighbour)
        dist.recv(torch.ones(1), src=neighbour)
        req.wait()
        print('pinged')

    def rendezvous(self, world_size):
        dist.init_process_group(self.backend, rank=self.rank, timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
        self.peers = list(filter(lambda x: x != self.rank, [0,1]))
        return dist.new_group(range(world_size))

    def step_(self, closure=None):
        t0 = time.time()
        for i, parameter in enumerate(self.params):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            parameter.grad.to(self.cpu)
            end.record()
            torch.cuda.synchronize()
            self.profile['transfer'] += start.elapsed_time(end) / 1000
            a1 = time.time()
            dist.all_reduce(parameter.grad, group=self.group)
            self.profile['communication'] += time.time() - a1
            parameter.grad /= self.world
            start.record()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            parameter.grad.to(self.gpu)
            end.record()
            torch.cuda.synchronize()
            self.profile['transfer'] += start.elapsed_time(end) / 1000
        self.profile['total'] += time.time() - t0
        super().step(closure)
    
    def quantized_step(self, closure=None):
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