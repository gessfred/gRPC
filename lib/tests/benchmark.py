import sys
sys.path.append('..')
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import argparse
import os
import datetime
import time
from torch.multiprocessing import Process, spawn
import cProfile
import communication as comm
from quantizy import quantize_gpu, unquantize_gpu
import numpy as np
from timer import TimerBase as Timer
from timer import CUDATimer

if __name__ == '__main__':
  