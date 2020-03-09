#include <stdio.h>
#include <torch/extension.h>
#include <map>
#include <vector>
#include <stdexcept>

#define integer_bits 32
#define bits 1

inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
  }
  return err;
}

__global__ void quantizeGPU(torch::Tensor tensor, torch::Tensor res)
{
  size_t pack_limit = 32/bits; // number of floats to pack into a single integer
  size_t N = torch::size(tensor, 0);
  size_t N2 = torch::size(res, 0);
  auto tensor_a = tensor.accessor<float,1>();
  auto res_a = res.accessor<int,1>();

  int startIndex = threadId*pack_limit + 0; // inclusive lower limit
  int endIndex = startIndex + pack_limit; // exclusive upper limit

  if (endIndex <= N) {
    int x = 0;
    for (int i = startIndex; i < endIndex; ++i) {
        x = x << bits;
        auto z = tensor_a[i];
        for (size_t k = 0; k <= 1; k++) {
          if (z < 0){
            x = x | 0;
          } else{
            x = x | 1;
          }
        }
      }
    res_a[threadId] = x;
  }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */
void quantizeCPU(torch::Tensor tensor, torch::Tensor res)
{
  auto tensor_a = tensor.accessor<float,1>();
  size_t N = torch::size(tensor, 0);

  size_t N2 = (N / integer_bits) * bits;
  auto res_a = res.accessor<int,1>();

  size_t pack_limit = integer_bits/bits;

  for(size_t i = 0; i < N2; i++){
    int x = 0;
    for(size_t j = 0; j < pack_limit; j++){
      x = x << bits;
      auto z = tensor_a[pack_limit*i + j];
      for (size_t k = 0; k <= length; k++) {
        if (z < 0){
          x = x | 0;
        } else{
          x = x | 1;
        }
      }
    }
    res_a[i] = x;
  }
}
