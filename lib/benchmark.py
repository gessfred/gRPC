#!/usr/bin/env python
import torch
import argparse
from all_reduce import ms_allreduce, ms_allreduce_un, ring_all_reduce, allreduce, allreduce_quant
from quantizy import quantizy
from torch.multiprocessing import Process
import time
from subprocess import run, Popen, PIPE
import torch.distributed as dist
import os
import datetime

def init():
    print('init...')
    IP = os.environ['MASTER_ADDR']
    os.environ['MASTER_PORT'] = os.environ['PYTORCH_PORT']#'29500'
    os.environ['GLOO_SOCKET_IFNAME'] = os.environ['PYTORCH_SOCKET']
    dist.init_process_group('gloo', rank=int(os.environ['RANK']), timeout=datetime.timedelta(seconds=10), world_size=2, init_method='tcp://{}:60000'.format(IP))
    print('init process group!')
    return dist.new_group(range(2))
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
def run(fn, args):
    size = 32
    iters = 1
    print('running...')
    group = init()
    r = dist.get_rank()
    world = dist.get_world_size()
    peers = list(filter(lambda i: i != r, list(range(world))))
    # Barrier here
    tensor = torch.ones(2**size)
    dist.barrier(group)
    start = time.time()
    for _ in range(iters):
        fn(r, world, peers, tensor, *args)
    exec_time = time.time() - start
    print(exec_time)

def run_baseline(iters):
    group = init()
    for size in [14, 18, 22, 26]:
        tensor = torch.ones(2**size)
        start = time.time()
        for _ in range(iters):
            dist.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
        exec_time = time.time() - start
        print(exec_time)

def pyflame(pid, output, mode, rate):
    #NOTE: we put a timeout of 20s but it's whatever
    p1 = Popen(['pyflame', '-x', '-r {}'.format(rate), '-s 3600', '-p {}'.format(pid)], stdout=PIPE)
    if mode == 'text':
        with open('{}.txt'.format(output), 'wb') as txt:
            txt.write(p1.stdout.read())
        print('Saved text output to {}.txt'.format(output))
    elif mode == 'flamegraph': 
        p2 = Popen(['/Flamegraph/flamegraph.pl'], stdin=p1.stdout, stdout=PIPE)
        with open('{}.svg'.format(output), 'wb') as svg:
            svg.write(p2.stdout.read())
        print('Saved FlameGraph svg output to {}.svg'.format(output))

def perf(pid, output, mode, rate):
    p1 = Popen(['perf', 'record', '-F 10000', '-p', pid, '--', 'sleep 10'], stdout=PIPE)

def vtune(pid, output, mode, rate):
    p1 = Popen(['amplxe-cl', '--collect', 'hotspots', '-target-pid', pid])

def pyspy(pid, output, mode, rate):
    p1 = Popen(['py-spy', 'record', '-o', output, '--pid', pid])

tools = {
    "vtune": vtune,
    "perf": perf,
    "pyflame": pyflame,
    "py-spy": pyspy,
}

functions = {
    "ring-all-reduce": ring_all_reduce,
    "ms-all-reduce-unquantized": ms_allreduce_un,
    "ms-all-reduce": ms_allreduce,
    "all-reduce": allreduce,
    "all-reduce-quant": allreduce_quant,
}

def test():
    group = init()
    r = dist.get_rank()
    world = dist.get_world_size()
    start = time.time()
    req = dist.isend(torch.ones(1), dst=(r + 1) % 2)
    dist.recv(torch.ones(1), src=(r + 1) % 2)
    req.wait()
    exec_time = time.time() - start
    print('exec_time:', exec_time)

def benchmark(fn, q, size, iterations, profile, output, mode, rate, numberOfThreads):
    #profile = tools[args.tool] if args.tool in [k for k in tools] else lambda pid, out, mode: None
    q = q if len(q) == 0 else q + [numberOfThreads]
    p = Process(target=run, args=(fn, q))
    print('starting process...', p.pid)
    p.start()
    print('started process...')
    if profiled:
        time.sleep(1)
        profile(str(p.pid), output, mode, rate)
    p.join()

def benchmarkQ(iters):
    quantize, unquantize = quantizy('ext_par')
    tensor = torch.one(2**20)
    
    for numberOfThreads in [1, 2, 4, 8, 16, 24, 32, 48]:
        start = time.time()
        for _ in range(iters):
            unquantize(quantize(tensor, numberOfThreads), numberOfThreads)
        runtime = time.time() - start
        print('{}: {}'.format(numberOfThreads, runtime))
    

def main():
    rank = int(os.environ['RANK'])
    print('rank is {}'.format(rank))
    parser = argparse.ArgumentParser(description='Benchmark runner')
    parser.add_argument('-it', type=int, dest='iterations', action='store',help='number of iterations')
    parser.add_argument('-v', dest='version', default='cast', action='store', help='all-reduce:numpy, all-reduce:ext, all-reduce-unsaturated implementation of the subject function')
    parser.add_argument('-sz', type=int, dest='size', action='store', help='size of the input tensor')
    parser.add_argument('-o', dest='output', default='bench', action='store', help='where to store the output file')
    parser.add_argument('-prof', dest='tool', action='store', help='profiling tool to use pyflame:txt, pyflame:flame, pyflame:folded, perf:flame, perf:folded, vtune')
    parser.add_argument('--ping', action='store_true', help='sends a RTT ping to rightmost neighbour')
    parser.add_argument('--threads', dest='numberOfThreads', action='store', default=1, type=int)
    args = parser.parse_args()
    func = args.version.split(':') if args.version is not None else None
    iters = args.iterations if args.iterations is not None else 10
    if args.ping:
        init()
        ping(rank)
    #elif func is not None and len(func) > 0 and func[0] == 'baseline':
    #    run_baseline(iters)
    elif func is not None and len(func) > 0 and func[0] == 'quantize':
        benchmarkQ(iters)
    else:
        print('Benchmarking...')
        fn = functions[func[0]] if func[0] in [k for k in functions] else None
        q = []
        if len(func) == 2:
            print(func[1])
            q = quantizy(func[1])
        prof = args.tool.split(':') if args.tool is not None else []
        mode = None
        if len(prof) > 1:
            mode = prof[1]
        profiled = not len(prof) == 0 and not prof[0] == '' 
        profile = tools[prof[0]] if profiled and prof[0] in [k for k in tools] else lambda pid, out, mode: None
        rate = 0.00001 if len(prof) < 3 else prof[2]
        if args.size is None:
            for size in [14, 18, 22, 26]:
                print('{}'.format(size))
                benchmark(fn, q, size, iters, profile, '{}-{}'.format(args.output, size), mode, rate, args.numberOfThreads)
        else:
            size = args.size if args.size is not None else 10
            benchmark(fn, q, size, iters, profile, args.output, mode, rate, args.numberOfThreads)

if __name__ == '__main__':
    p = Process(target=test)
    p.start()
    p.join()