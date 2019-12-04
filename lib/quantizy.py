import torch
import numpy as np
from q_cpp import quantize_shrink, unquantize_shrink
dataSz = 32
"""
Naive functions
"""
def quantize(tensor):
    N = list(tensor.size())[0]
    Q = torch.zeros(N, dtype=bool)
    Q = tensor > 0
    return Q
def unquantize(tensor):
    tensor = tensor.type(torch.FloatTensor)
    tensor[tensor == 0] = -1
    return tensor # * data_scale
"""
NumPy implementations
"""
#https://stackoverflow.com/questions/49791312/numpy-packbits-pack-to-uint16-array
def quantize_vector(tensor):
    quantized = (tensor.numpy() > 0).astype(int)
    packed = np.packbits((quantized.reshape(-1, 4, 8)[:, ::-1]))
    return torch.from_numpy(packed.view(np.int32))

def unquantize_vector(tensor):
    unpacked = np.unpackbits(tensor.numpy().view(np.uint8))
    #tensor[...] = 1 stays 1
    unpacked[unpacked == 0] = -1
    return torch.from_numpy(unpacked).type(torch.float64)
"""
Python proof of concept of the basic packing algorithm
"""
def quantize_pof(tensor):
    N = list(tensor.size())[0] #tensor.size = arraySize / world
    print(N)
    #assert N % dataSz == 0
    N2 = N // dataSz
    res = torch.zeros(N2, dtype=int) 
    for i in range(N2):
        x = 0
        for j in range(dataSz):
            x = x << 1
            z = tensor[dataSz*i + j]
            if z >= 0:
                z = 1
            else:
                z = 0
            x = x | z
        res[i] = x
    return res

def unquantize_pof(tensor):
    N2 = list(tensor.size())[0]
    N = N2 * dataSz
    res = torch.zeros(N, dtype=float)
    for i in range(N2):
        x = tensor[i]
        for j in range(dataSz):
            z = (x >> (dataSz - 1 - j)) & 1
            if z == 1:
                res[dataSz*i + j] = 1
            else:
                res[dataSz*i + j] = -1
    return res
def quantizy(version):
    versions = {
        "cast": [quantize, unquantize],
        "numpy": [quantize_vector, unquantize_vector],
        "concept": [quantize_pof, unquantize_pof],
        "ext": [quantize_shrink, unquantize_shrink],
    }
    return versions[version]