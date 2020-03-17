#!/usr/bin/env python
import sys
sys.path.append('..')
import torch
import time
from quantizy import quantizy

def check_quantize(q, uq, device=None):
    # Bins are not of the same size as PyTorch uses half to even rounding.

    # tensor size is a multiple of 32
    tensor = torch.cat((torch.arange(-1,1+2/256,2/256, device=device), torch.zeros(31, device=device)))

    # zero is mapped to the quantisation below
    res_1bit = torch.cat(((torch.zeros(128)-1),(torch.zeros(129)+1),(torch.zeros(31)+1)))

    res_2bit = torch.cat(((torch.zeros(33)-1),(torch.zeros(95)-.5),(torch.zeros(96)+.5),(torch.zeros(33)+1),(torch.zeros(31)+.5)))

    res_4bit = torch.cat(((torch.zeros(9)-1),(torch.zeros(15)-.875),(torch.zeros(17)-.75),(torch.zeros(15)-.625)
                ,(torch.zeros(17)-.5),(torch.zeros(15)-.375),(torch.zeros(17)-0.25),(torch.zeros(23)-.125)
                ,(torch.zeros(24)+.125),(torch.zeros(17)+.25),(torch.zeros(15)+.375),(torch.zeros(17)+.5)
                ,(torch.zeros(15)+.625),(torch.zeros(17)+.75),(torch.zeros(15)+.875),(torch.zeros(9)+1),(torch.zeros(31)+.125)))

    res_8bit = tensor.clone().cpu()
    res_8bit[128] = res_8bit[129]
    for i in range(257,288):
        res_8bit[i] = res_8bit[129]

    res = [res_1bit,res_2bit,res_4bit,res_8bit]

    for bits, expected in zip([1,2,4,8], res):
        print('bits: {}'.format(bits))
        quantized = q(tensor, bits, device)
        unquantized = uq(quantized, bits, device).cpu()
        if len(expected) == len(unquantized):
            print(' length correct')
        else:

            print('Expected: {}'.format(expected.shape))
            print('Received: {}'.format(unquantized.shape))
            return

        if torch.eq(unquantized, expected).all():
            print(' elements correct')
            print()
        else:
            print('Expected: {}'.format(expected))
            print('Quantized: {}'.format(quantized))
            print('Unquantized: {}'.format(unquantized))
            index = torch.eq(unquantized, expected).logical_not().nonzero()
            print(index)
            print(tensor[index])
            print(expected[index])
            print(unquantized[index])
            return

def check_speed(q, uq, size=10, iters=1000, device=None):

    tensor = torch.rand(32*2**size)*2-1

    for bits in [1,2,4,8]:
        print('bits: {}'.format(bits))
        start = time.time()
        for _ in range(iters):
            quantized = q(tensor, bits, device)
        exec_time_q = time.time() - start
        start = time.time()
        for _ in range(iters):
            unquantized = uq(quantized, bits, device)
        exec_time_uq = time.time() - start
        print(' Quantisation: {:6.6} / Unquantisation: {:6.6}'.format(str(exec_time_q), str(exec_time_uq)))

def check_cuda():
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        print("Using CUDA: {}".format(cuda_device))
    else:
        cuda_device = None
        print("Using CPU:")
    return cuda_device

if __name__ == '__main__':
    q, uq = quantizy('gpu')
    device = check_cuda()
    check_quantize(q, uq, device= device)
    check_speed(q, uq, size=16, device=device)
    print("Using CPU:")
    check_speed(q, uq, size=16, device=None)
