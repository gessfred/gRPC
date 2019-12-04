#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
from torch import ones, zeros
import os
from lib.quantizy import quantizy
from lib.all_reduce import ms_allreduce
import socket
import torch.distributed as dist
import json
import datetime
with open('config.json') as cfg:
    config = json.load(cfg)
    master = config["master"]
    IP = master["ip"]#ifconfig
    print(IP)
    os.environ['MASTER_ADDR'] = IP
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = master["interface"]
    dist.init_process_group('gloo', rank=os.environ['RANK'], timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(IP))
    dist.new_group(2)
    print(ms_allreduce(torch.ones(1024), *quantizy("cast")))