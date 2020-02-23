#!/usr/bin/env python
import sys
sys.path.append('../lib')
import torch
from quantizy import quantizy


def check_equality(q, uq):
    tensor = torch.empty(32).normal_(mean=0,std=1)
    for bits in [1,2,4]:
        quantized = q(tensor, bits, 24)
        requantized = q(uq(quantized, bits, 24), bits, 24)
        print(bin(quantized.data.item()), bin(requantized.data.item()))
        assert(quantized==requantized)

def check_unquantize(q,uq):
    tensor = torch.tensor([
    -0.3868, -0.3625,  1.4073,  1.3122,  0.2161, -0.0865, -0.6423, -0.4744,
    -1.5699, -0.0692, -0.3361,  2.4448, -0.1353,  0.2083, -1.4788, -0.7977,
    -0.5271, -0.9620, -0.4317, -0.0900,  1.3423,  0.5249, -2.0423, -0.0221,
    -0.6187,  0.8197, -0.3350,  1.0201,  0.7726,  1.2855,  0.1228,  0.4905])

    res_1bit = []

    for bits in [1,2,4]:
        quantized = q(tensor, bits, 24)


def check_quantize(fn):
    tensor = torch.tensor([
    -0.3868, -0.3625,  1.4073,  1.3122,  0.2161, -0.0865, -0.6423, -0.4744,
    -1.5699, -0.0692, -0.3361,  2.4448, -0.1353,  0.2083, -1.4788, -0.7977,
    -0.5271, -0.9620, -0.4317, -0.0900,  1.3423,  0.5249, -2.0423, -0.0221,
    -0.6187,  0.8197, -0.3350,  1.0201,  0.7726,  1.2855,  0.1228,  0.4905])

    res_1bit = [0b00111000000101000000110001011111]
                 # 1 1 3 3 2 1 1 1 0 1 1 3 1 2 0 0,    1 0 1 1 3 2 0 1 1 3 1 3 3 3 2 2
    res_2bit = [0b01011111100101010001011101100000, 0b01000101111000010111011111111010]
                 # 5 5 14 14 9 7 4 5, 0 7 5 15 7 9 1 3,
                 # 4 2 5 7 14 11 0 7, 4 12 5 13 12 14 8 11
    res_4bit = [0b01010101111011101001011101000101, 0b00000111010111110111100100010011,
                0b01000010010101111110101100000111, 0b01001100010111011100111010001011]

    res = [res_1bit, res_2bit, res_4bit]


    print(tensor)
    for bits, expected in zip([1,2,4],res):
        print('bits: {}'.format(bits))
        actual = fn(tensor, bits, 24)
        assert( len(expected) == len(actual) )
        print(' length correct')
        for val, e in zip(actual.data, expected):
            a = val.item()
            print('expected: {}, actual: {}'.format(e,a), end='')
            assert(bin(a) == bin(e))
            print(' --- correct')


        print()

if __name__ == '__main__':
    q, uq = quantizy('general')
    #check_quantize(q)
    check_equality(q,uq)
