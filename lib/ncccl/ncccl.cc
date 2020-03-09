#include <torch/extension.h>
#include "init.h"
#include <iostream>
#include "net_socket.h"
#include "primitives.h"

ncclNet_t* net;

void init(int rank, int nRanks, std::array<char, 128> uuid, int dst)  {
  INFO();
  int size = 32*1024*1024;

  int myRank = rank;
  int localRank = 0;
  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024, ':');
  hostHashs[myRank] = getHostHash();
  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;
  std::cout << hostname << std::endl;
  ncclNet_t* net = ncclNetSocket();
  //get NCCL unique ID at rank 0 and broadcast it to all others
  std::copy_n(uuid.begin(), 128, std::begin(id.internal));  
//if (myRank == 0) ncclGetUniqueId(&id);
  //std::cout << std::string(id.internal) << std::endl;
  //picking a GPU based on localRank, allocate device buffers
  (cudaSetDevice(localRank));
  (cudaMalloc(&sendbuff, size * sizeof(float)));
  (cudaMalloc(&recvbuff, size * sizeof(float)));
  (cudaStreamCreate(&s));

  std::cout << "init..." << std::endl;
  //initializing NCCL
  (ncclCommInitRank(net, &comm, nRanks, id, myRank));
  net->
  //std::cout << "net& (init) "<< net << std::endl;
  free((void*)net);
}

std::array<char, 128> get_local_id() {
  std::array<char, 128> res;
  ncclUniqueId id;
  ncclNet_t* net = ncclNetSocket();
  ncclGetUniqueId(net, &id);
  std::cout << "net& " << net << std::endl; 
  std::copy_n(std::begin(id.internal), 128, res.begin());
  free((void*)net);
  return res;
  //return reinterpret_cast<std::array<char, 128>&>(id.internal);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "init");
  m.def("get_local_id", &get_local_id, "get_local_id");
}

