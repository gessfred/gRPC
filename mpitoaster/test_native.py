
import torch
import os
import mpitoaster

if __name__ == '__main__':
  world_size = int(os.environ['WORLD_SIZE'])
  mpi = mpitoaster.MPIToaster()
  mpi.init()
  tensor = torch.ones(1024).cuda()
  gather_list = [torch.zeros(1024).cuda() for i in range(world_size)]
  print('before', tensor, gather_list)
  mpi.gather(tensor, gather_list, 0)
  torch.cuda.synchronize()
  print('after', tensor, gather_list)
  """native.init(rank, 0, world_size)
   * rank
  gather_list = [tensor.clone() for i in range(world_size)]
  print(gather_list)
  native.gather(tensor, gather_list, 0)
  print(gather_list)"""
