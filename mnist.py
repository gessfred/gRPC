#!/usr/bin/env python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.multiprocessing import Process
import os
import math
import numpy as np
from q_cpp import quantize_shrink, unquantize_shrink
quantize_vector = quantize_shrink
unquantize_vector = quantize_shrink
dataSz = 32
def quantize(tensor, numberOfThreads=1):
    N = list(tensor.size())[0]
    Q = torch.zeros(N, dtype=bool)
    Q = tensor > 0
    return Q
def unquantize(tensor, numberOfThreads=1):
    tensor = tensor.type(torch.FloatTensor)
    tensor[tensor == 0] = -1
    return tensor # * data_scale
"""def quantize_vector(tensor, numberOfThreads=1):
    quantized = tensor.numpy()
    quantized = quantized > 0
    quantized = quantized.astype(int)
    quantized = quantized.reshape(-1, 4, 8)[:, ::-1]
    packed = np.packbits(quantized)
    packed = packed.view(np.int32)
    return torch.from_numpy(packed)

def unquantize_vector(tensor, numberOfThreads=1):
    unpacked = tensor.numpy()
    unpacked = unpacked.view(np.uint8)
    unpacked = np.unpackbits(unpacked)
    #tensor[...] = 1 stays 1
    unpacked[unpacked == 0] = -1
    res = torch.from_numpy(unpacked)
    return res.type(torch.float64)"""
def pad(sz, world_size):
    precision = 32
    sz1 = int(math.ceil(sz / float(precision)))
    return int(math.ceil(sz1 / float(world_size))) * precision * world_size
"""def quantize_vector(tensor, numberOfThreads=1):
    N = list(tensor.size())[0] #tensor.size = arraySize / world
    #assert N % dataSz == 0
    N2 = N // dataSz
    res = torch.zeros(N2, dtype=int) 
    for i in range(N2):
        x = 0
        for j in range(dataSz):
            x = x << 1
            z = tensor[dataSz*i + j]
            if z > 0:
                z = 1
            else:
                z = 0
            x = x | z
        res[i] = x
    return res

def unquantize_vector(tensor, numberOfThreads=1):
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
    return res"""
"""class TestQuantization(unittest.TestCase):
    def test_ones(self):
        T = torch.ones(10)
        self.assertTrue(torch.all(unquantize(quantize(T)) == torch.ones(10)))
    def test_minus_ones(self):
        T = -torch.ones(10)
        self.assertTrue(torch.all(-torch.ones(10) == unquantize(quantize(T))))"""
#the padding is just so that a chunk can be quantized by 32 bit increments. also it should be possible for each worker to get a chunk
def allreduce_quant(r, world, peers, tensor, quantize=quantize_vector, unquantize=unquantize_vector, numberOfThreads=24):
    #preparation (flattening, padding, ...)
    
    originalShape = tensor.shape
    tensor = tensor.flatten()
    originalSize=list(tensor.size())[0]
    paddedSize = pad(originalSize, world)
    tensor = torch.nn.functional.pad(tensor, (0,paddedSize-originalSize))
    sizeOfTensor=list(tensor.size())[0]
    flatSize = sizeOfTensor // dataSz
    chunksize = sizeOfTensor // world
    reqs = []
    for i in peers: # K steps
        chunk = tensor[i*chunksize:(i+1)*chunksize]
        qchunk = quantize_shrink(chunk, numberOfThreads) #qchunk is int32
        reqs += [dist.isend(tensor=qchunk, dst=i)] # K concurrent transfers
    recv = torch.zeros(sizeOfTensor // (world * dataSz), dtype=torch.int32)
    for i in peers: # K steps
        dist.recv(tensor=recv,src=i) # K / ??? values...
        chunk = unquantize_shrink(recv, numberOfThreads)
        tensor[r*chunksize:(r+1)*chunksize] += chunk
    for req in reqs:
        req.wait()
    # we have to set to zero the values that we are not responsible (they will be included on their way back)

    reqs = []
    for i in peers:
        chunk = tensor[r*chunksize:(r+1)*chunksize]
        qchunk = quantize_shrink(chunk, numberOfThreads)
        reqs += [dist.isend(tensor=qchunk,dst=i)]
    for i in peers:
        dist.recv(tensor=recv, src=i)
        chunk = unquantize_shrink(recv, numberOfThreads)
        tensor[i*chunksize:(i+1)*chunksize] = chunk
    for req in reqs:
        req.wait()
    #clean-up/put everything back in place
    tensor = tensor[:originalSize].reshape(originalShape)
    return tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
# WIth data distributed ===============================
#Test set: Average loss: 0.1699, Accuracy: 9636/10000 (96%)


#Test set: Average loss: 0.1574, Accuracy: 9654/10000 (97%)
"""
With same data
Test set: Average loss: 0.1153, Accuracy: 9698/10000 (97%)


Test set: Average loss: 0.1142, Accuracy: 9701/10000 (97%)
"""
"""
Test set: Average loss: 0.2159, Accuracy: 9349/10000 (93%)


Test set: Average loss: 0.2275, Accuracy: 9317/10000 (93%)
"""
#0.001

"""
one node(lr=0.01)
Test set: Average loss: 0.0923, Accuracy: 9711/10000 (97%)
lr=0.001
#all-reduce-no-error-feedback

Test set: Average loss: 0.3441, Accuracy: 9713/10000 (97%)


Test set: Average loss: 0.3298, Accuracy: 9731/10000 (97%)
#feedback   
0.02 

Test set: Average loss: 0.1319, Accuracy: 9585/10000 (96%)


Test set: Average loss: 0.1214, Accuracy: 9629/10000 (96%)

0.01
Test set: Average loss: 0.1854, Accuracy: 9452/10000 (95%)


Test set: Average loss: 0.1990, Accuracy: 9415/10000 (94%)
"""
def train(args, model, device, train_loader, optimizer, epoch):
    group = dist.new_group([0, 1])
    model.train()
    quantization_error = [None]*len(list(model.parameters()))
    rank = dist.get_rank()
    log = lambda msg: print(msg) if rank == 0 else None
    peers = list(filter(lambda x: x != rank, [0,1]))
    world = dist.get_world_size()
    for batch_idx, (data, target) in enumerate(train_loader):
        dim = len(target) // 4
        data = data[:dim] if rank == 0 else data[dim:]
        target = target[:dim] if rank == 0 else target[dim:]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        #update model gradients here
        loss.backward()
        # accumulate grad here
        #for i, parameter in enumerate(model.parameters()):
        #    local = parameter.grad.clone()
        #    if quantization_error[i] is not None:
        #        parameter.grad += quantization_error[i]
        #    allreduce_quant(rank, world, peers, parameter.grad)
        #    parameter.grad /= world
        #    quantization_error[i] = local - parameter.grad
        optimizer.step()
    
        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def init():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    return args
def sanity_test():
    pass
def main(args, rank=0, size=0):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)#ASGD
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
def init_process(args, rank,size,fn,backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args, rank, size)


def init_processes(args, fn, size=2):
    processes=[]
    for rank in range(size):
        p = Process(target=init_process, args=(args, rank, size, fn))
        p.start()
        processes.append(p)
    return lambda: [p.join() for p in processes]

if __name__ == '__main__':
    #main(init())
    join = init_processes(init(), main)
    join()