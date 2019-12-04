#!/usr/bin/env python
import sys
sys.path.append("../lib/")
import torch
from torch import ones, zeros
import os
from quantizy import quantizy
import socket
import torch.distributed as dist
import json

with open('config.json') as cfg:
    config = json.load(cfg) 
    master = config["master"]
    q, u = quantizy("cast")
    IP = master["ip"]#ifconfig
    print(IP)
    os.environ['MASTER_ADDR'] = IP
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = master["interface"]
    rank = 0 if socket.gethostbyname(socket.gethostname()) == IP else 1
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method='tcp://{}:23456'.format(IP))
    print('connected to {} nodes'.format(dist.get_world_size()))
print(u(q(ones(1024))))