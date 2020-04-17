import <torch/extension.h>
#include <nccl.h>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <string>
#include <array>
#include <vector>
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

///Globals
ncclComm_t world_comm;
ncclUniqueId id;
cudaStream_t world_stream;
unsigned int rank_global = 0;
unsigned int world_size_global = 0;

std::array<char, 128> get_local_id() {
  std::array<char, 128> res;
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  std::copy_n(std::begin(id.internal), 128, res.begin());
  return res;
  //return reinterpret_cast<std::array<char, 128>&>(id.internal);
}

/*Print a CUDA array*/
void print_array_dev(int** array, int size) {
  int *buff = (int*)calloc(size, sizeof(int));
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(buff, *array, size, cudaMemcpyDeviceToHost));
  for(size_t i = 0; i < size; ++i) std::cout << buff[i] << ",";
  std::cout << std::endl;
  free(buff);
}

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

void init(unsigned int rank, unsigned int device, unsigned int world_size) {
  auto id = get_local_id();
  rank_global = rank;
  world_size_global = world_size;
  NCCLCHECK(ncclCommInitRank(&world_com, world_size, id, rank));
}

void gather(torch::Tensor tensor, torch::Tensor* gather_list, unsigned int dst) {
  if(dst == rank_global) {
    for(size_t i = 0; i < world_size_global; ++i) {
      if(i != dst) {
        ncclRecv((float*)gather_list[i], (size_t)gather_list[i].size(), ncclFloat32, i, world_comm, world_stream);
      } else {
        gather_list[i] = tensor;
      }
    }
  } else {
    ncclSend((float*)tensor, (size_t)tensor.size(), ncclFloat32, dst, world_comm, world_stream);
  }
}

/*void allreduce(torch::Tensor tensor) {
    torch::Tensor compressed = packing(tensor);
    
}*/

PYBIND11_MODULE(native, m) {
  m.def("init", &init, "init");
  m.def("gather", &gather, "gather");
}

