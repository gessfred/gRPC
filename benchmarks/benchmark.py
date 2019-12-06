#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
import argparse
from quantizy import quantizy
from torch.multiprocessing import Process
import time
from subprocess import run, Popen, PIPE
def profile(iters, size, version):
    time.sleep(3)
    tensor = torch.ones(2**size)
    qu, unqu = quantizy(version)
    for _ in range(iters):
        unqu(qu(tensor))

def run(iterations, size, version, output, flamegraph):
    p = Process(target=profile, args=(iterations, size, version))
    p.start()
    #NOTE: we put a timeout of 20s but it's whatever
    p1 = Popen(['pyflame', '-x', '-r 0.00001', '-s 60', '-p {}'.format(p.pid)], stdout=PIPE)
    if not flamegraph:
        with open('{}.txt'.format(output), 'wb') as txt:
            txt.write(p1.stdout.read())
        print('Saved text output to {}.txt'.format(output))
    else: 
        p2 = Popen(['/Flamegraph/flamegraph.pl'], stdin=p1.stdout, stdout=PIPE)
        with open('{}.svg'.format(output), 'wb') as svg:
            svg.write(p2.stdout.read())
        print('Saved FlameGraph svg output to {}.svg'.format(output))
    p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-it', type=int, dest='iterations', action='store',
                        default=10,
                        help='number of iterations')
    parser.add_argument('-v', dest='version', default='cast', action='store', help='implementation of quantize among (cast, concept, numpy, extension)')
    parser.add_argument('-sz', type=int, dest='size', default=12, action='store', help='size of the input tensor')
    parser.add_argument('--flamegraph', help='store the output in a flamegraph', action='store_true')
    parser.add_argument('-o', dest='output', default='bench', action='store', help='where to store the output file')
    parser.add_argument('--all', help='run a batch benchmark for different input sizes and algorithm versions', action='store_true')
    args = parser.parse_args()
    if args.all:
        for size in range(8, 30, 2):
            iters = 1000 if size < 16 else 100
            for version in ['numpy', 'ext']:
                run(iters, size, version, 'data-{}-{}'.format(version, size), args.flamegraph)
    else:
        run(args.iterations, args.size, args.version, args.output, args.flamegraph)
 