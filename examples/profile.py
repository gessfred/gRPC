#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
from torch import ones, zeros
import os
from quantizy import quantizy
from all_reduce import ms_allreduce
import socket
import torch.distributed as dist
import json
import datetime
import argparse
def profile(iterations, size, quantization):
    with open('config.json') as cfg:
        config = json.load(cfg)
        master = config["master"]
        IP = master["ip"]#ifconfig
        print(IP)
        os.environ['MASTER_ADDR'] = IP
        os.environ['MASTER_PORT'] = '29500'
        os.environ['GLOO_SOCKET_IFNAME'] = master["interface"]
        dist.init_process_group('gloo', rank=os.environ['RANK'], timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(IP))
        dist.new_group(range(2))
        subject = torch.ones(2**size)
        qn = quantizy(quantization)
        for _ in range(iterations):
            ms_allreduce(subject, *qn)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-iterations', type=int, dest='iterations', action='store', nargs=1,
                        default=10,
                        help='number of iterations')
    parser.add_argument('-quantize', dest='quantize', default='cast', action='store', nargs=1, help='implementation of quantize among (cast, concept, numpy, extension)')
    parser.add_argument('-size', type=int, dest='size', default=12, action='store', nargs=1, help='size of the input tensor')
    args = parser.parse_args()
    profile(args.iterations[0], args.size[0], args.quantize)