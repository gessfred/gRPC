#include <nccl.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <string>
#include <array>
#include <algorithm>
#include <iterator>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

std::array<char, 128> get_local_id() {
  std::array<char, 128> res;
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  std::copy_n(std::begin(id.internal), 128, res.begin());
  return res;
  //return reinterpret_cast<std::array<char, 128>&>(id.internal);
}

//ncclGetErrorString
void init(int nDev) {
    ncclComm_t comms[nDev];
    //int size = 32*1024*1024;
    int devs[nDev];
    for(int i = 0; i < nDev; ++i) {
        devs[i] = i;
    }
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
}

void send(int rank, int nRanks, std::array<char, 128> uuid, int dst)  {
    int size = 1024;

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
  int *buff = (int*)calloc(size, sizeof(int));
  int *sendbuff, *recvbuff;
  cudaStream_t s;
  std::cout << hostname << std::endl;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  std::copy_n(uuid.begin(), 128, std::begin(id.internal));  
//if (myRank == 0) ncclGetUniqueId(&id);
  //std::cout << std::string(id.internal) << std::endl;
  //picking a GPU based on localRank, allocate device buffers
  std::cout << "(device) " << localRank << std::endl; 
  //CUDACHECK(cudaSetDevice(localRank));
  
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(int)));
  CUDACHECK(cudaMemset(sendbuff, 0xFF, size * sizeof(int)));
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(buff, sendbuff, size, cudaMemcpyDeviceToHost));
  for(size_t i = 0; i < size; ++i) std::cout << buff[i] << ",";
  std::cout << std::endl;
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(int)));
  CUDACHECK(cudaStreamCreate(&s));

  std::cout << "init" << std::endl;
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  int root = 0;
  NCCLCHECK(ncclBroadcast(sendbuff, recvbuff, size, ncclInt32, root, comm, s));
  //communicating using NCCL
  //NCCLCHECK(ncclSend(dst, (const void*)sendbuff, size, ncclFloat,
  //      comm, s));
  CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaMemcpy(buff, sendbuff, size, cudaMemcpyDeviceToHost));
    for(size_t i = 0; i < size; ++i) std::cout << buff[i] << ",";
    std::cout << std::endl;

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));


  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);


  printf("[MPI Rank %d] Success \n", myRank);
}

int main(void) {
  auto id = get_local_id();
  std::cout << "id" << std::endl;
  int rank = atoi(std::getenv("RANK"));
  send(rank, 2, id, (rank+1)%2);
}