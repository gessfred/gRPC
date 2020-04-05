#include <nccl.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <string>
#include <array>
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
  //std::copy_n(std::begin(id.internal), 128, res.begin());
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


int main(void) {
  init(1);
}