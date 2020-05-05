# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer

from lib import quantize_gpu, unquantize_gpu, CompressedTensorBuffer
import sys
import bit2byte
def element_num(size):
    num = 1
    for i in range(len(size)):
        num *= size[i]
    return num
def _pack(tensor, bits=1):
    tensor_ = tensor.view(-1) #flatten tensor
    dev = tensor_.device
    pack = 32//bits
    bins = 2**bits
    padding = 0
    n = tensor_.shape[0]
    if not (n % 32) == 0:
        pad_size = list(tensor_.size())[0] % 32
        tensor_ = torch.nn.functional.pad(tensor_, (0, (32 - pad_size) % 32))
        n = tensor_.shape[0]
        padding = (32 - pad_size) % 32
    clamped_tensor = tensor_.abs().clamp(3/(2*bins), 1)*(tensor_.lt(0).logical_not()*2-1)
    rounded_tensor = (((clamped_tensor)*(bins//2)).clamp(-bins//2, bins//2)).round()
    res = ((rounded_tensor + rounded_tensor.lt(0)*1 +(bins//2 -1)).to(torch.int32) \
            * (torch.zeros(n, dtype=torch.int32, device=dev)+2).pow(torch.arange(0, 32, bits, device=dev).repeat(n//pack)) \
            ).reshape((-1, pack)).cumsum(dim=1)[:,pack-1].to(torch.int32)
    return res, padding

def _unpack(tensor, padding, bits=1):
    dev = tensor.device
    pack = 32//bits
    bins = 2**bits
    n = tensor.shape[0] * pack
    res = tensor.repeat_interleave(pack)
    b = (torch.zeros(n, dtype=torch.int32, device=dev)+2).pow(torch.arange(0, 32, bits, device=dev).repeat(n//pack))
    tmp = (res & (b*(bins-1)))/b
    tmp2 = (tmp + tmp.lt(0)*bins).float() - (bins/2)
    res = (tmp2 + (tmp2.lt(0).logical_not()))/(bins/2)
    if padding == 0:
        return res
    else:
        return res[:-padding]

class IntTensorBuffer:
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors, use_cuda=True):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]

        self.buffer = torch.cat(list(map(lambda tensor: tensor.view(-1), tensors)))  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def nelement(self):
        return self.buffer.nelement()

class TB():
    def __init__(self, ref_tb, buffer):
        self._start_idx = ref_tb._start_idx
        self._end_idx = ref_tb._end_idx
        self._tensors_len = ref_tb._tensors_len
        self._tensors_sizes = ref_tb._tensors_sizes
        self.buffer = buffer
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def nelement(self):
        return self.buffer.nelement()

def signum(tensor):
    compressed, padding = _pack(tensor)
    gather_list = [compressed.clone() for i in range(dist.get_world_size())]
    dist.all_gather(gather_list, compressed)
    gather_list = list(map(lambda recv: _unpack(recv, padding), gather_list))
    return torch.sum(torch.stack(gather_list), dim=0) / dist.get_world_size()

class Local_EFSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
        model=None
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Local_EFSGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        self.local_step = conf.local_step
        self.turn_on_local_step_from_epoch = conf.turn_on_local_step_from
        self.bits = conf.compress_width

        # define the aggregator.
        self.world_aggregator = comm.get_aggregators(
            conf,
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )
        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # initialize the concensus
        self._init_consensus()
        self._init_memory()

    def _init_consensus(self):
        params, _ = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.consensus_params_tb = deepcopy(TensorBuffer(params))

    def _init_memory(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.memory_tb = TensorBuffer(params)
        self.memory_tb.buffer = torch.zeros_like(self.memory_tb.buffer)

    def __setstate__(self, state):
        super(Local_EFSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        with kargs['timer']('sync', epoch=self.conf.epoch_):
            # do the local update steps.
            with kargs["timer"]("local_update", epoch=self.conf.epoch_):
                utils.apply_gradient(
                    self.param_groups, self.state, apply_grad_to_model=True
                )

            # enter the global sync if it satisfies the condition.
            if (
                self.conf.epoch_ < self.turn_on_local_step_from_epoch
                or self.conf.local_index % self.local_step == 0
            ):
                with kargs["timer"]("get_params", epoch=self.conf.epoch_):
                    # get parmas.
                    params, _ = comm.get_data(
                        self.param_groups, self.param_names, is_get_grad=False
                    )
                    params_tb = TensorBuffer(params)
                with kargs['timer']('memory_and_compress', epoch=self.conf.epoch_):
                    # get the params difference w.r.t. previous synced model.
                    local_scale, local_sign = [], []
                    for consensus_param, param, memory in zip(
                        self.consensus_params_tb, params_tb, self.memory_tb
                    ):
                        memory.data.copy_(consensus_param - param + memory)
                        # compress.
                with kargs["timer"]("directions", epoch=self.conf.epoch_):
                    direction = signum(self.memory_tb.buffer)
                with kargs['timer']('memory_and_compress', epoch=self.conf.epoch_):
                    for consensus_param, param, memory in zip(
                        self.consensus_params_tb, params_tb, self.memory_tb
                    ):
                        _local_scale, _local_sign = scaled_sign(memory)
                        local_scale.append(_local_scale)
                        local_sign.append(_local_sign)
                        memory.data.copy_(memory - _local_scale * _local_sign)
                with kargs["timer"]("directions", epoch=self.conf.epoch_):
                    global_direction = TB(self.memory_tb, direction)
                with kargs["timer"]("magnitudes", epoch=self.conf.epoch_):
                    magnitudes_tb = TensorBuffer(local_scale)
                    magnitudes_tb.buffer = self.world_aggregator._agg(
                        magnitudes_tb.buffer, "avg", distributed=self.conf.distributed
                    )
                # unpack the synced info and update the consensus params.
                with kargs["timer"]("update_consensus", epoch=self.conf.epoch_):
                    for update_magnitude, update_direction, consensus_param in zip(
                        magnitudes_tb, global_direction, self.consensus_params_tb
                    ):
                        consensus_param.add_(-1.0, update_direction.mul(update_magnitude))

                # consistent the local models by assigning the consensus params.
                self.consensus_params_tb.unpack(params)
                n_bits = get_n_bits(magnitudes_tb.buffer)
            else:
                n_bits = 0
            return n_bits

def scaled_sign(x, name=None):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm divided by the number of elements
    """
    _scale = x.norm(p=1) / x.numel()
    _sign = torch.sign(x)

    return _scale, _sign
