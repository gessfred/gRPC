#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
import argparse
from lib.all_reduce import ms_allreduce
from quantizy import quantizy
from torch.multiprocessing import Process
import time
from subprocess import run, Popen, PIPE
import torch.distributed as dist
import os
import datetime

def run_allreduce(iters, size, version):
    IP = "10.90.38.4"
    os.environ['MASTER_ADDR'] = IP
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = "ens786f0"
    dist.init_process_group('gloo', rank=os.environ['RANK'], timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(IP))
    dist.new_group(range(2))
    time.sleep(3)
    subject = torch.ones(2**size)
    qn = quantizy(version)
    for _ in range(iters):
        ms_allreduce(subject, *qn)

def run_quantize(iters, size, version):
    time.sleep(3)
    tensor = torch.ones(2**size)
    qu, unqu = quantizy(version)
    for _ in range(iters):
        q = qu(tensor)
        unqu(q)

def pyflame(pid, output, mode):
    #NOTE: we put a timeout of 20s but it's whatever
    p1 = Popen(['pyflame', '-x', '-r 0.00001', '-s 360', '-p {}'.format(pid)], stdout=PIPE)
    if mode == 'text':
        with open('{}.txt'.format(output), 'wb') as txt:
            txt.write(p1.stdout.read())
        print('Saved text output to {}.txt'.format(output))
    elif mode == 'flamegraph': 
        p2 = Popen(['/Flamegraph/flamegraph.pl'], stdin=p1.stdout, stdout=PIPE)
        with open('{}.svg'.format(output), 'wb') as svg:
            svg.write(p2.stdout.read())
        print('Saved FlameGraph svg output to {}.svg'.format(output))

def perf(pid, output, mode):
    p1 = Popen(['perf', 'record', '-F 10000', '-p', pid, '--', 'sleep 10'], stdout=PIPE)

def vtune(pid, output, mode):
    p1 = Popen(['amplxe-cl', '--collect', 'hotspots', '-target-pid', pid])

tools = {
    "vtune": vtune,
    "perf": perf,
    "pyflame": pyflame,
}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-it', type=int, dest='iterations', action='store',
                        default=10,
                        help='number of iterations')
    parser.add_argument('-v', dest='version', default='cast', action='store', help='implementation of the subject function')
    parser.add_argument('-sz', type=int, dest='size', default=12, action='store', help='size of the input tensor')
    parser.add_argument('--flamegraph', help='store the output in a flamegraph', action='store_true')
    parser.add_argument('-m', dest='mode', action='store', help='output type in [flamegraph, folded, txt]')
    parser.add_argument('-o', dest='output', default='bench', action='store', help='where to store the output file')
    parser.add_argument('-t', dest='tool', default='pyflame', action='store', help='profiling tool to use')
    parser.add_argument('-f', dest='func', default='quantize', help='function to profile', action='store')
    parser.add_argument('--all', help='run a batch benchmark for different input sizes and algorithm versions', action='store_true')
    args = parser.parse_args()
    profile = tools[args.tool] if args.tool in [k for k in tools] else pyflame
    run = run_allreduce if args.func == 'all-reduce' else run_quantize
    if args.all:
        for size in range(8, 30, 2):
            iters = 1000
            for version in ['numpy', 'ext']:
                p = Process(target=run, args=(iters, size, version))
                p.start()
                profile(str(p.pid), 'data-{}-{}'.format(version, size), args.mode)
                p.join()
    else:
        p = Process(target=run, args=(args.iterations, args.size, args.version))
        p.start()
        profile(str(p.pid), args.output, args.mode)
        p.join()
