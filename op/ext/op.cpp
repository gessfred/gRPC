/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density potential on CUDA according to NTUPlace3 (https://doi.org/10.1109/TCAD.2008.923063)
 */
//#include "utility/src/torch.h"
#include <torch/extension.h>

int computeTestLauncher(float* input, unsigned int size);

int computeTest(torch::Tensor input) {

 return computeTestLauncher((float*)input.data_ptr(), at::size(input, 32));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("comp", &computeTest, "comp");
}
