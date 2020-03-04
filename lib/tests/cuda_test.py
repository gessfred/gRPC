#!/usr/bin/env python
import sys
sys.path.append('..')
import torch
import time
from quantizy import quantizy

def check_quantize(fn, cuda):
    print('Checking quantization...')
    tensor = torch.tensor([
    -0.3868, -0.3625,  1.4073,  1.3122,  0.2161, -0.0865, -0.6423, -0.4744,
    -1.5699, -0.0692, -0.3361,  2.4448, -0.1353,  0.2083, -1.4788, -0.7977,
    -0.5271, -0.9620, -0.4317, -0.0900,  1.3423,  0.5249, -2.0423, -0.0221,
    -0.6187,  0.8197, -0.3350,  1.0201,  0.7726,  1.2855,  0.1228,  0.4905], device=cuda)

    res_1bit = [0b00111000000101000000110001011111]
    res = [res_1bit]


    #print(tensor)
    for bits, expected in zip([1],res):
        print('bits: {}'.format(bits))
        actual = fn(tensor, bits, 1)
        torch.cuda.synchronize()
        assert( len(expected) == len(actual) )
        print(' length correct')
        for val, e in zip(actual.data, expected):
            a = val.item()
            print(' expected: {}, actual: {}'.format(e,a), end='')
            assert(bin(a) == bin(e))
            print(' --- correct')
        print()

def check_cuda():
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")
    return cuda_device

if __name__ == '__main__':
    q, uq = quantizy('ext_par')
    cuda = check_cuda()
    check_quantize(q, cuda)
