/**
 * @file   hpwl_cuda.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 * @brief  Compute density potential on CUDA according to NTUPlace3 (https://doi.org/10.1109/TCAD.2008.923063)
 */
#include "utility/src/torch.h"

int computeTestLauncher(at::Tensor input);

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel()&1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

typedef double T; 

 
int density_potential_forward(at::Tensor input)  {
    return computeTestLauncher(input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::density_potential_forward, "DensityPotential forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::density_potential_backward, "DensityPotential backward (CUDA)");
  m.def("fixed_density_map", &DREAMPLACE_NAMESPACE::fixed_density_potential_map, "DensityPotential Map for Fixed Cells (CUDA)");
}