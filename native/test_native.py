import native
import torch
import os

if __name__ == '__main__':
  world_size = int(os.environ['WORLD_SIZE'])
  rank = int(os.environ['RANK'])
  native.init(rank, 0, world_size)
  tensor = torch.ones(1024) * rank
  gather_list = [tensor.clone() for i in range(world_size)]
  print(gather_list)
  native.gather(tensor, gather_list, 0)
  print(gather_list)