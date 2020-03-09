import torch
import torch.distributed as dist
import os


def allreduce(tensor, rank, group):
    #dist.reduce_multigpu()
    sizeOfTensor=list(tensor.size())[0]
    chunksize = sizeOfTensor // world
    world = group.get_world_size()
    for i in range(world):
        chunk = tensor[i*chunksize:(i+1)*chunksize]
        dist.reduce(chunk, i, op=ReduceOp.SUM, group=group)
    chunk = tensor[rank*chunksize:(rank+1)*chunksize]
    tensor_list = [chunk]*world
    dist.all_gather(tensor_list, chunk, group=group)
    print(tensor_list)
        

def rendezvous(rank, world_size):
    
    dist.init_process_group('nccl', rank=rank, timeout=datetime.timedelta(seconds=10), world_size=world_size, init_method='tcp://{}:60000'.format(os.environ['MASTER_ADDR']))
    
    return dist.new_group(range(world_size))

def main():
    tensor = torch.ones(8)
    rank = int(os.environ['RANK'])
    group = rendezvous(rank, 2)
    allreduce(tensor, rank, group)

if __name__ == '__main__':
    main()