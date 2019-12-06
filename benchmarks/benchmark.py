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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-it', type=int, dest='iterations', action='store',
                        default=10,
                        help='number of iterations')
    parser.add_argument('-v', dest='version', default='cast', action='store', help='implementation of quantize among (cast, concept, numpy, extension)')
    parser.add_argument('-sz', type=int, dest='size', default=12, action='store', help='size of the input tensor')
    parser.add_argument('--flamegraph', help='store the output in a flamegraph', action='store_true')
    parser.add_argument('-o', dest='output', default='bench', action='store', help='where to store the output file')
    args = parser.parse_args()
#    profile(args.iterations[0], args.size[0], args.version[0])
    p = Process(target=profile, args=(args.iterations, args.size, args.version))
    p.start()
   #time.sleep(1)
    #cmd = '../../pyflame/src/pyflame'# -pid {}'.format(p.pid)
    #run(['perf', 'record',  '-g',  '-F 31750', '-p {}'.format(p.pid)])
    #run(['perf script | ../../FlameGraph/stackcollapse-perf.pl > out.perf-folded'])
#    run(['../../FlameGraph/flamegraph.pl out.perf-folded > fl.svg'])
#NOTE: we put a timeout of 20s but it's whatever
    txt = open('{}.txt'.format(args.output), 'wb')
    svg = open('{}.svg'.format(args.output), 'wb')
    p1 = Popen(['pyflame', '-x', '-r 0.00001', '-s 60', '-p {}'.format(p.pid)], stdout=PIPE)
    if not args.flamegraph:
        txt.write(p1.stdout.read())
        print('Saved text output to {}.txt'.format(args.output))
    else: 
        p2 = Popen(['/Flamegraph/flamegraph.pl'], stdin=p1.stdout, stdout=PIPE)
        svg.write(p2.stdout.read())
        print('Saved FlameGraph svg output to {}.svg'.format(args.output))
    txt.close()
    svg.close()
    p.join()
