#include <torch/extension.h>
#include <map>
#include <vector>
#include <stdexcept>

/**
std::vector<float> _1bit = {-0.6745, 0, 0.6745};
std::vector<float> _2bit = {-1.1503, -0.6745, -0.3186, 0, 0.3186, 0.6745, 1.1503};
std::vector<float> _4bit = {-1.8627, -1.5341, -1.3180, -1.1503, -1.0100, -0.8871, -0.7764, -0.6745,
                            -0.5791, -0.4888, -0.4023, -0.3186, -0.2372, -0.1573, -0.0784, 0,
                            0.0784, 0.1573, 0.2372, 0.3186, 0.4023, 0.4888, 0.5791, 0.6745,
                            0.7764, 0.8871, 1.0100, 1.1503, 1.3180, 1.5341, 1.8627};
*/

std::vector<float> _1bit_q = {0};
std::vector<float> _1bit_uq = {-0.6745, 0.6745};
std::vector<float> _2bit_q = {-0.6745, 0, 0.6745};
std::vector<float> _2bit_uq = {-1.1503, -0.3186, 0.3186, 1.1503};
std::vector<float> _4bit_q = {-1.5341, -1.1503, -0.8871, -0.6745, -0.4888, -0.3186, -0.1573, 0,
                            0.1573, 0.3186, 0.4888, 0.6745, 0.8871, 1.1503, 1.5341};
std::vector<float> _4bit_uq = {-1.8627, -1.3180, -1.0100, -0.7764, -0.5791, -0.4023, -0.2372, -0.0784,
                            0.0784,  0.2372, 0.4023,  0.5791, 0.7764, 1.0100, 1.3180, 1.8627};

std::map<int, std::vector<float>> quantization_levels = {
  {1, _1bit_q},
  {2, _2bit_q},
  {4, _4bit_q}
};
std::map<int, std::vector<float>> unquantization_levels = {
  {1, _1bit_uq},
  {2, _2bit_uq},
  {4, _4bit_uq}
};

int integer_bits = 32;


torch::Tensor quantize_general(torch::Tensor tensor, size_t bits, size_t numberOfThreads){
    if (!(bits==1 || bits==2 || bits==4)){
      throw std::invalid_argument( "Quantization level not implemented. Please use 1, 2, or 4." );
    }

    auto tensor_a = tensor.accessor<float,1>();
    size_t N = torch::size(tensor, 0);
    if (N%integer_bits != 0){
      throw std::invalid_argument( "Tensor must have a multiple of 32 elements." );
    }

    size_t N2 = (N / 32) * bits;
    auto res = torch::zeros(N2, torch::kInt32);
    auto res_a = res.accessor<int,1>();

    std::vector<float> quantization_limits = quantization_levels[bits];
    size_t length = quantization_limits.size();
    size_t pack_limit = 32/bits;

    #pragma omp parallel for num_threads(numberOfThreads)
    for(size_t i = 0; i < N2; i++){
        int x = 0;
        for(size_t j = 0; j < pack_limit; j++){
            x = x << bits;
            auto z = tensor_a[pack_limit*i + j];
            for (size_t k = 0; k <= length; k++) {
              if (k == length || z < quantization_limits[k]){
                x = x | k;
                break;
              }
            }
        }
        res_a[i] = x;
    }
    return res;
}


torch::Tensor unquantize_general(torch::Tensor tensor, size_t bits, size_t numberOfThreads){
    if (!(bits==1 || bits==2 || bits==4)){
      throw std::invalid_argument( "Quantization level not implemented. Please use 1, 2, or 4." );
    }

    auto tensor_a = tensor.accessor<int,1>();
    size_t N2 = torch::size(tensor, 0);

    size_t pack_limit = 32/bits;
    unsigned int mask = (1<<bits)-1;
    std::vector<float> unquantization_values = unquantization_levels[bits];

    size_t N = N2 * pack_limit;
    auto res = torch::zeros(N, torch::kFloat32);
    auto res_a = res.accessor<float,1>();

    #pragma omp parallel for num_threads(numberOfThreads)
    for(size_t i = 0; i < N2; i++){
        unsigned int x = (unsigned int)tensor_a[i];
        for(size_t j = 0; j < pack_limit; j++){
            int z = (x >> (32 - (j+1)*bits)) & mask;
            res_a[pack_limit*i + j] = unquantization_values[z];
        }
    }
    return res;
}

PYBIND11_MODULE(q_general_cpp, m) {
  m.def("quantize_general", &quantize_general, "quantize general");
  m.def("unquantize_general", &unquantize_general, "unquantize general");
}
