#include <torch/extension.h>
#include "cuda_runtime.h"

__global__ void testFunctionL(float* input, unsigned int size) {
if(blockIdx.x < size)
	input[blockIdx.x] = blockIdx.x;    
}

int computeTestLauncher(float* input, unsigned int size) {
	testFunctionL<<<1, 1>>>(input, size);
    return 0;
}
