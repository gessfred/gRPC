#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
import argparse
from quantizy import quantizy
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-iterations', type=int, dest='iterations', action='store', nargs=1,
                        default=10,
                        help='number of iterations')
    parser.add_argument('-quantize', dest='quantize', default='cast', action='store', nargs=1, help='implementation of quantize among (cast, concept, numpy, extension)')
    parser.add_argument('-size', type=int, dest='size', default=12, action='store', nargs=1, help='size of the input tensor')
    args = parser.parse_args()
    iters = args.iterations[0]
    tensor = torch.ones(2**args.size[0])
    qn, un = quantizy(args.quantize[0])
    for _ in range(iters):
        un(qn(tensor))