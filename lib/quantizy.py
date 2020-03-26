import torch
import numpy as np
from q_cpp import quantize_shrink, unquantize_shrink
from q_par_cpp import quantize_shrink_par, unquantize_shrink_par
from q_general_cpp import quantize_general, unquantize_general

dataSz = 32

"""
GPU functions
"""
#assumes 1-d tensor and normalized (range -1,1), otherwise clamping will be performed.
def quantize_gpu(tensor, bits):
    tensor_ = tensor.view(-1) #flatten tensor
    dev = tensor_.device
    pack = 32//bits
    bins = 2**bits
    padding = 0
    n = tensor_.shape[0]
    if not (n % 32) == 0:
        pad_size = list(tensor_.size())[0] % 32
        tensor_ = torch.nn.functional.pad(tensor_, (0, (32 - pad_size) % 32))
        n = tensor_.shape[0]
        padding = (32 - pad_size) % 32
    clamped_tensor = tensor_.abs().clamp(3/(2*bins), 1)*(tensor_.lt(0).logical_not()*2-1)
    rounded_tensor = (((clamped_tensor)*(bins//2)).clamp(-bins//2, bins//2)).round()
    res = ((rounded_tensor + rounded_tensor.lt(0)*1 +(bins//2 -1)).to(torch.int32) \
            * (torch.zeros(n, dtype=torch.int32, device=dev)+2).pow(torch.arange(0, 32, bits, device=dev).repeat(n//pack)) \
            ).reshape((-1, pack)).cumsum(dim=1)[:,pack-1].to(torch.int32)
    return res, padding

    # return  ((((tensor+1)*(bins//2)).clamp(0.1, bins-0.1)-1).ceil().to(torch.int32) \
    #         * (torch.zeros(n, dtype=torch.int32, device=cuda)+2).pow(torch.arange(0, 32, bits, device=cuda).repeat(n//pack)) \
    #         ).reshape((-1, pack)).cumsum(dim=1)[:,pack-1].to(torch.int32)

#assumes 1-d tensor and normalized (range -1,1), otherwise clamping will be performed.
def unquantize_gpu(tensor, padding, bits):
    tensor_ = tensor.int()
    dev = tensor_.device
    pack = 32//bits
    bins = 2**bits
    n = tensor_.shape[0] * pack
    res = tensor_.repeat_interleave(pack)
    b = (torch.zeros(n, dtype=torch.int32, device=dev)+2).pow(torch.arange(0, 32, bits, device=dev).repeat(n//pack))
    tmp = (res & (b*(bins-1)))/b
    tmp2 = (tmp + tmp.lt(0)*bins).float() - (bins/2)
    res = (tmp2 + (tmp2.lt(0).logical_not()))/(bins/2)
    if padding == 0:
        return res
    else:
        return res[:-padding]

def flatten(tensors, shapes=None, use_cuda=True):
    # from https://github.com/epfml/LocalSGD-Code
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    vec = torch.empty(
        pointers[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)
    return vec

class CompressedTensorBuffer:
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors, bits, use_cuda=True):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]
        buf, pad = quantize_gpu(flatten(tensors, use_cuda=use_cuda), bits)
        self.buffer = buf  # copies
        self.padding = pad
        self.bits = bits
        self._buffer = None

    def __getitem__(self, index):
        return self._buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def unpack(self, tensors):
        self.buffer = self.buffer
        self._buffer = self._buffer
        for tensor, entry in zip(tensors, self):
            tensor.data[:] = entry

    def decompress(self):
        torch.cuda.synchronize()
        self._buffer = unquantize_gpu(self.buffer, self.padding, self.bits)

"""
Naive functions
"""
def quantize(tensor, numberOfThreads):
    N = list(tensor.size())[0]
    Q = torch.zeros(N, dtype=bool)
    Q = tensor > 0
    return Q
def unquantize(tensor, numberOfThreads):
    tensor = tensor.type(torch.FloatTensor)
    tensor[tensor == 0] = -1
    return tensor # * data_scale
"""
NumPy implementations
"""
#https://stackoverflow.com/questions/49791312/numpy-packbits-pack-to-uint16-array
def quantize_vector(tensor, numberOfThreads):
    quantized = tensor.numpy()
    quantized = quantized > 0
    quantized = quantized.astype(int)
    quantized = quantized.reshape(-1, 4, 8)[:, ::-1]
    packed = np.packbits(quantized)
    packed = packed.view(np.int32)
    return torch.from_numpy(packed)

def unquantize_vector(tensor, numberOfThreads):
    unpacked = tensor.numpy()
    unpacked = unpacked.view(np.uint8)
    unpacked = np.unpackbits(unpacked)
    #tensor[...] = 1 stays 1
    unpacked[unpacked == 0] = -1
    res = torch.from_numpy(unpacked)
    return res.type(torch.float64)
"""
Python proof of concept of the basic packing algorithm
"""
def quantize_pof(tensor, numberOfThreads):
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

def unquantize_pof(tensor, numberOfThreads):
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
        "ext_par": [quantize_shrink_par, unquantize_shrink_par],
        "general": [quantize_general, unquantize_general],
        "gpu": [quantize_gpu, unquantize_gpu]
    }
    return versions[version]
#
