#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
from torch import ones, zeros
import os
from quantizy import quantizy
import socket
import torch.distributed as dist
import json
import datetime
with open('config.json') as cfg:
    config = json.load(cfg)
    master = config["master"]
    q, u = quantizy("cast")
    IP = master["ip"]#ifconfig
    print(IP)
    os.environ['MASTER_ADDR'] = IP
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = master["interface"]
    dist.init_process_group('gloo', rank=os.environ['RANK'], timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(IP))
