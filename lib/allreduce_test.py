from all_reduce import allreduce_quant
from local import init_processes
import unittest
import torch



class TestAllReduce(unittest.TestCase):
    def setTest(self, input, output):
        def aggregate(rank, size):
            peers = list(filter(lambda x: x != rank,range(size)))
            allreduce_quant(rank, size, peers, input)
            self.assertTrue(torch.all(input.eq(output)))
        return aggregate
            
    def test_tensor_16bits(self):
        input = torch.zeros(16)
        init_processes(self.setTest(input, torch.zeros(16)), 2)()
    
    def test_tensor_32bits(self):
        input = torch.zeros(32)
        init_processes(self.setTest(input, torch.zeros(32)), 2)()

    def test_tensor_45bits(self):
        input = torch.zeros(45)
        init_processes(self.setTest(input, torch.zeros(45)), 2)()

if __name__ == '__main__':
    unittest.main()
