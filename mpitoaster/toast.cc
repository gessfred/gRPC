#include <iostream>
#include <cstdlib>
#include <nccl.h>
#include <ATen.h>
class mpitoaster_t {
  int device;
  int rank;
  int world_size;
  ncclComm_t comm;
  cudaStream_t stream;

  public:
  mpitoaster_t(int, int, int);
  ~mpitoaster_t();
  void init();
  void bcast_org(float*, size_t);
  void bcast(float*, size_t);
  void bcast_grouped(float*, size_t);
};

mpitoaster_t::mpitoaster_t(int d, int r, int w) {
  device = d;
  rank = r;
  world_size = w;
}

mpitoaster_t::~mpitoaster_t() {
  
}

void mpitoaster_t::init(  ) {
  cudaSetDevice(device);
  cudaStreamCreate(&stream);
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclCommInitRank(&comm, world_size, id, rank);
  std::cout << "init" << std::endl;
}
/*

ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
*/
void mpitoaster_t::bcast_org(float* tensor, size_t count) {
  ncclBroadcast(tensor, tensor, count, ncclFloat32, 0, comm, stream);
}

void mpitoaster_t::bcast(float* tensor, size_t count) {
  if(rank == 0) {
    for(size_t i = 1; i < world_size; ++i) {
      ncclSend(tensor, count, ncclFloat32, i, comm, stream);
    }
  } else {
    ncclRecv(tensor, count, ncclFloat32, i, comm, stream);
  }
}

void mpitoaster_t::bcast_grouped(float* tensor, size_t count) {
  ncclGroupStart();
  if(rank == 0) {
    for(size_t i = 1; i < world_size; ++i) {
      ncclSend(tensor, count, ncclFloat32, i, comm, stream);
    }
  } else {
    ncclRecv(tensor, count, ncclFloat32, i, comm, stream);
  }
  ncclGroupEnd();
}

void print_tensor(float** tensor, size_t count) {
  float *buff = (float*)calloc(  count, sizeof(float));
  cudaDeviceSynchronize();
  cudaMemcpy(buff, *tensor, count*sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "tensor([";
  for(size_t i = 0; i < count; ++i) std::cout << buff[i] << ",";
  std::cout << "])" << std::endl;
  free(buff);
}

int main( void ){
  int rank = atoi(std::getenv("RANK"));
  int device = atoi(std::getenv("LOCAL_RANK"));
  int world = atoi(std::getenv("WORLD_SIZE"));

  std::cout << "hello, world" << std::endl; 
  mpitoaster_t mpi(device, rank, world);
  mpi.init();
 

  size_t count = 1024;


  float* input = (float*)calloc(count, sizeof(float));
  for(size_t i = 0; i < count; ++i) input[i] = 1.0;

   
  float* sendbuff, *recvbuff;
  cudaMalloc(&sendbuff, count * sizeof(float));
  cudaMalloc(&recvbuff, count * sizeof(float));
  cudaMemcpy(sendbuff, input, count * sizeof(float), cudaMemcpyHostToDevice);//cudaMemset(sendbuff, 1.0, count * sizeof(float));
  print_tensor(&sendbuff, 1024);
  mpi.bcast_org(tensor, count);
  //mpi.allreduce(sendbuff, count); 
  print_tensor(&sendbuff, 1024); 
  
}
