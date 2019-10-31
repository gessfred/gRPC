import torch
from all_reduce import quantize, unquantize
import unittest

class TestQuantization(unittest.TestCase):
    def test_ones(self):
        T = torch.ones(10)
        self.assertTrue(torch.all(unquantize(quantize(T)) == torch.ones(10)))
    def test_minus_ones(self):
        T = -torch.ones(10)
        self.assertTrue(torch.all(-torch.ones(10) == unquantize(quantize(T))))

if __name__ == '__main__':
    unittest.main()