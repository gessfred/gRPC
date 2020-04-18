#include <nccl.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <string>
#include <array>
#include <algorithm>
#include <iterator>
#define info(str) std::cout << "[\033[0;36mINFO\033[0m] " << str << std::endl; 
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

void log(float** array, int size) {
  float *buff = (float*)calloc(size, sizeof(float));
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(buff, *array, size, cudaMemcpyDeviceToHost));
  for(size_t i = 0; i < size; ++i) std::cout << buff[i] << ",";
  std::cout << std::endl;
  free(buff);

}
class dist_t {
  int rank;
  int dev;
  int world_size;
  ncclDatatype_t dtype;

  ncclComm_t comm;
  cudaStream_t stream;


  dist_t();
  ~dist_t();
  void init();
  void gather(float*, float**, int);
};

dist_t::dist_t() {
  rank = atoi(std::getenv("RANK"));
  dev = atoi(std::getenv("LOCAL_RANK"));
  world_size = atoi(std::getenv("WORLD_SIZE"));
  dtype = ncclFloat32;
}
dist_t::~dist_t() {
  ncclCommDestroy(comm);
}

void dist_t::init() {
  CUDACHECK(cudaSetDevice(dev));
  CUDACHECK(cudaStreamCreate(&s));
  auto id = get_local_id();
  NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));
  info("init");
} 

/*
tensor has size |count|
gather_list has size |world|
*/
void dist_t::gather(float* tensor, size_t count, float** gather_list, int dst) {
  if(rank == dst) {
    for(size_t i = 0; i < world_size; ++i) {
      if(i != dst) {
        ncclRecv(gather_list[i], count, ncclFloat32, i, comm, stream);
      }
    }
    gather_list[i] = tensor;
  } else {
    ncclSend(tensor, count, ncclFloat32, dst, comm, stream);
  }
}

void dist_t::allreduce(float* tensor, size_t tensorcount) {
  size_t chunkcount = tensorcount / world;
  for(size_t i = 0; i < world; ++i) {
    size_t offset = i*chunkcount;
    NCCLCHECK(ncclReduce(tensor+offset, tensor+offset, chunkcount, ncclFloat32, ncclSum, i, comm, stream));
  }
  NCCLCHECK(ncclAllGather(tensor+rank*tensorcount, tensor, tensorcount, ncclFloat32, comm, stream));
}

/*void dist_t::allreduce_(float* tensor, size_t tensorcount) {

}*/

int main(void) {
  size_t count = 1024;
  float* tensor;
  CUDACHECK(cudaMalloc(&tensor, count));
  for(size_t i = 0; i < count; ++i) {
    tensor[i] = 1f;
  }
  log(tensor);
  /*Testing data*/
  dist_t dist;
  dist.init();
  dist.allreduce(tensor, count);
  log(tensor);
}