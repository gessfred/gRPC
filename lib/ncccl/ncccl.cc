#include <torch/extension.h>
#include "init.h"

ncclNet_t* net;

void init(int rank, int nRanks, std::array<char, 128> uuid, int dst)  {
    int size = 32*1024*1024;

  int myRank = rank;
  int localRank = 0;
    int argc = 1;
     char** argv;

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;
  std::cout << hostname << std::endl;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  std::copy_n(uuid.begin(), 128, std::begin(id.internal));  
//if (myRank == 0) ncclGetUniqueId(&id);
  //std::cout << std::string(id.internal) << std::endl;
  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));


  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "init");
}

