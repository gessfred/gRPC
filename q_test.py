import torch
from q_cpp import quantize_shrink as q2
from q_cpp import unquantize_shrink as q2_
from q import quantize_shrink as q1
from q import unquantize_shrink as q1_
from q import quantize_vector as vq1
from q import unquantize_vector as vq1_

tensor = torch.rand(2**12) - 0.5
#tensor = torch.ones(2**5)

print(tensor)

#python
'''for i in range(100):
    q = q1(tensor)

for i in range(100):
    q_1 = q1_(q)'''

# vector
for i in range(100):
    q1_t = vq1(tensor)

for i in range(100):
    vq_1 = vq1_(q1_t)

# c
for i in range(100):
    q2_t = q2(tensor)

for i in range(100):
    q_2 = q2_(q2_t)

assert((q1_t == q2_t).all())
assert((vq_1 == q_2).all())
