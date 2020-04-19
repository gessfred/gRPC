#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nccl.h>
#include <vector>
/*#include <cstdlib>
#include <unistd.h>
#include <string>
#include <array>
#include <algorithm>
#include <iterator>*/
#define INFO_TAG "[\033[0;36mINFO\033[0m]"
#define info(str) std::cout << "[\033[0;36mINFO\033[0m]" << " "  << str << std::endl; 
//#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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

/*static uint64_t getHostHash(const char* string) {
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
}*/

/*From https://github.com/jiaweizzhao/signSGD-with-Majority-Vote/blob/master/main/bit2byte-extension/bit2byte.cpp*/
torch::Tensor packing(torch::Tensor src) {
    //src is dim(32*-1) IntTensor
    //make sure shift just gnerates zero

    auto options = torch::TensorOptions()
                       .dtype(torch::kInt)
                       .device(src.device());
    torch::Tensor a = torch::zeros(2, options);
    a[0] = 1;
    a[1] = -2;
    auto mask_0 = a[0];
    auto mask_1 = a[1];

    src[0].__irshift__(mask_0);
    src[0].__iand__(mask_0);

    for (int i = 1; i < 32; i++)
    {
        src[0].__ilshift__(mask_0);
        src[0].__iand__(mask_1);
        src[i].__irshift__(mask_0);
        src[i].__iand__(mask_0);
        src[0].__ior__(src[i]);
    }

    return {src[0]};
}

torch::Tensor unpacking(torch::Tensor src, torch::Tensor dst) {
    //src is dim(1*-1) IntTensor
    //dst is dim(32*-1) IntTensor(ones)
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt)
                       .device(src.device());
    torch::Tensor a = torch::zeros(1, options);
    a[0] = 1;
    auto mask_0 = a[0];

    for (int i = 31; i >= 0; i--)
    {
        dst[i].__iand__(src);
        dst[i].__ilshift__(mask_0);
        src.__irshift__(mask_0);
    }

    return {dst};
    //outside we should -(dst-1)
}


class dist_t {
  int rank;
  int dev;
  int world_size;
  //ncclDataType_t dtype;

  ncclComm_t comm;
  cudaStream_t stream;

  public:
  dist_t();
  ~dist_t();
  void init();
  void gather(torch::Tensor, std::vector<torch::Tensor>, int);
  //void allreduce(float*, size_t);*/
};

dist_t::dist_t() {
  rank = atoi(std::getenv("RANK"));
  dev = atoi(std::getenv("LOCAL_RANK"));
  world_size = atoi(std::getenv("WORLD_SIZE"));
  //dtype = ncclFloat32;
}
dist_t::~dist_t() {
  //ncclCommDestroy(comm);
}

void dist_t::init() {
  CUDACHECK(cudaSetDevice(dev));
  CUDACHECK(cudaStreamCreate(&stream));
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));
  info("init");
} 

/*
tensor has size |count|
gather_list has size |world|
*/
void dist_t::gather(torch::Tensor tensor, std::vector<torch::Tensor> gather_list, int dst) {
  std::cout << INFO_TAG << "tensor.size " << tensor.size(0) << std::endl;
  std::cout << INFO_TAG << "gather_list.size " << gather_list.size() << std::endl;
  size_t count = torch::size(tensor, 0);
  if(rank == dst) {
    for(size_t i = 0; i < world_size; ++i) {
      if(i != dst) {
        float* tensor_ = gather_list[i].data<float>();
        ncclRecv(tensor_, tensor.size(0)-1, ncclFloat32, i, comm, stream);
      }
    }
    gather_list[rank] = tensor;
  } else {
    float* tensor_ = tensor.data<float>();
    ncclSend(tensor_, tensor.size(0)-1, ncclFloat32, dst, comm, stream);
  }
}


/*void dist_t::allreduce(float* tensor, size_t tensorcount) {
  size_t chunkcount = tensorcount / world_size;
  for(size_t i = 0; i < world_size; ++i) {
    size_t offset = i*chunkcount;
    NCCLCHECK(ncclReduce(tensor+offset, tensor+offset, chunkcount, ncclFloat32, ncclSum, i, comm, stream));
  }
  NCCLCHECK(ncclAllGather(tensor+rank*tensorcount, tensor, tensorcount, ncclFloat32, comm, stream));
}*/

PYBIND11_MODULE(mpitoaster, m) {
    py::class_<dist_t>(m, "MPIToaster")
      .def(py::init<>())
      .def("init", &dist_t::init)
      .def("gather", &dist_t::gather);
}

