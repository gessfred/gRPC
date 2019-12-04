import torch
import sys
sys.path.append("../lib/")

from quantizy import quantizy

q, u = quantizy("cast")
print(u(q(torch.ones(1024))))