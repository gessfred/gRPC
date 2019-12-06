#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
import argparse
from quantizy import quantizy
from torch.multiprocessing import Process
import time
from subprocess import run
def profile(iters, size, version):
#    time.sleep(10)
    tensor = torch.ones(2**size)
    qu, unqu = quantizy(version)
    for _ in range(iters):
        unqu(qu(tensor))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-it', type=int, dest='iterations', action='store', nargs=1,
                        default=10,
                        help='number of iterations')
    parser.add_argument('-v', dest='version', default='cast', action='store', nargs=1, help='implementation of quantize among (cast, concept, numpy, extension)')
    parser.add_argument('-sz', type=int, dest='size', default=12, action='store', nargs=1, help='size of the input tensor')
    args = parser.parse_args()
    iters = args.iterations[0]
    profile(args.iterations[0], args.size[0], args.version[0])
#    p = Process(target=profile, args=(args.iterations[0], args.size[0], args.version[0]))
#    p.start()
    #time.sleep(1)
    #cmd = '../../pyflame/src/pyflame'# -pid {}'.format(p.pid)
 #   run(['perf', 'record', '-p {}'.format(p.pid), '-g'])
    #run(['perf script | ../../FlameGraph/stackcollapse-perf.pl > out.perf-folded'])
#    run(['../../FlameGraph/flamegraph.pl out.perf-folded > fl.svg'])
#    print(p.pid)
#    p.join()
