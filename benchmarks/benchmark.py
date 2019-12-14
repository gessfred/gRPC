#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
import argparse
from all_reduce import ms_allreduce, ms_allreduce_un
from quantizy import quantizy
from torch.multiprocessing import Process
import time
from subprocess import run, Popen, PIPE
import torch.distributed as dist
import os
import datetime

def init():
    IP = "10.90.38.4"
    os.environ['MASTER_ADDR'] = IP
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = "ens786f0"
    dist.init_process_group('gloo', rank=os.environ['RANK'], timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(IP))
    dist.new_group(range(2))
def ping(rank):
    req = dist.isend(torch.ones(1), dst=rank + 1 % 2)
    dist.recv(torch.ones(1), src=rank + 1 % 2)
    req.wait()
    print('pinged')
"""
    run:
    iters: number of iterations
    size: input size or range of input sizes
"""
def run(fn, args, size, iters=100):
    init()
    time.sleep(2)
    start = time.time()
    tensor = torch.ones(2**size)
    for _ in range(iters):
        fn(tensor, *args)
    exec_time = time.time() - start
    print(exec_time)

def pyflame(pid, output, mode, rate):
    #NOTE: we put a timeout of 20s but it's whatever
    p1 = Popen(['pyflame', '-x', '-r {}'.format(rate), '-s 360', '-p {}'.format(pid)], stdout=PIPE)
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

functions = {
    "ring-all-reduce": None,
    "all-reduce-unsaturated": ms_allreduce_un,
    "all-reduce": ms_allreduce
}

def benchmark(fn, q, size, iterations, profile, output, mode, rate):
    #profile = tools[args.tool] if args.tool in [k for k in tools] else lambda pid, out, mode: None
    
    p = Process(target=run, args=(fn, q, args.size, args.iterations))
    p.start()
    if profiled:
        time.sleep(1)
        profile(str(p.pid), args.output, mode, rate)
    p.join()

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    parser = argparse.ArgumentParser(description='Benchmark runner')
    parser.add_argument('-it', type=int, dest='iterations', action='store',help='number of iterations')
    parser.add_argument('-v', dest='version', default='cast', action='store', help='all-reduce:numpy, all-reduce:ext, all-reduce-unsaturated implementation of the subject function')
    parser.add_argument('-sz', type=int, dest='size', action='store', help='size of the input tensor')
    parser.add_argument('-o', dest='output', default='bench', action='store', help='where to store the output file')
    parser.add_argument('-prof', dest='tool', action='store', help='profiling tool to use pyflame:txt, pyflame:flame, pyflame:folded, perf:flame, perf:folded, vtune')
    parser.add_argument('--ping', action='store_true', help='sends a RTT ping to rightmost neighbour')
    args = parser.parse_args()
    if args.ping:
        init()
        ping(rank)
    else:
        func = args.version.split(':')
        fn = functions[func[0]] if func[0] in [k for k in functions] else None
        q = []
        if len(func) == 2:
            q = quantizy(func[1])
        prof = args.tool.split(':') if args.tool is not None else []
        mode = None
        if len(prof) > 1:
            mode = prof[1]
        profiled = not len(prof) == 0 and not prof[0] == '' 
        profile = tools[prof[0]] if profiled and prof[0] in [k for k in tools] else lambda pid, out, mode: None
        rate = 0.00001 if len(prof) < 3 else prof[2]
        iters = args.iterations if args.iterations is not None else 10
        if args.size is None:
            for size in [14, 18, 22, 26]:
                print('{}'.format(size))
                benchmark(fn, q, size, iters, profile, args.output, mode, rate)
        else:
            size = args.size if args.size is not None else 10
            benchmark(fn, q, size, iters, profile, args.output, mode, rate)