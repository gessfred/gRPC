#include <torch/extension.h>
#include <map>
#include <vector>
#include <stdexcept>


std::vector<float> _1bit = {-0.6745, 0, 0.6745};
std::vector<float> _2bit = {-1.1503, -0.6745, -0.3186, 0, 0.3186, 0.6745, 1.1503};
std::vector<float> _4bit = {-1.5341, -1.1503, -0.8871, -0.6745, -0.4888, -0.3186, -0.1573,
                       0, 0.1573, 0.3186, 0.4888, 0.6745, 0.8871, 1.1503, 1.5341};

std::map<int, std::vector<float>> quantization_levels = {
  {1, _1bit},
  {2, _2bit},
  {4, _4bit}
};

int integer_bits = 32;


torch::Tensor quantize_general(torch::Tensor tensor, size_t bits, size_t numberOfThreads){
    if (!(bits==1 || bits==2 || bits==4)){
      throw std::invalid_argument( "Quantization level not implemented. Please use 1, 2, or 4." );
    }

    auto tensor_a = tensor.accessor<float,1>();
    int N = torch::size(tensor, 0);
    if (N%integer_bits != 0){
      throw std::invalid_argument( "Tensor must have a multiple of 32 elements." );
    }

    int N2 = N / 32;
    auto res = torch::zeros(N2, torch::kInt32);
    auto res_a = res.accessor<int,1>();

    for(int i = 0; i < N2; i++){
        int x = 0;
        for(int j = 0; j < 32; j++){
            x = x << bits;
            auto z = tensor_a[32*i + j];
            std::vector<float> quantization_limits = quantization_levels[bits];
            size_t length = quantization_limits.size();
            for (size_t k = 1; k < length; k+=2) {
              if (z < quantization_limits[i]){
                x = x | (i-1/2);
                continue;
              }
            }
            x = x | (length-1)/2;
        }
        res_a[i] = x;
    }
    return res;
}


torch::Tensor unquantize_general(torch::Tensor tensor, size_t numberOfThreads){
    auto tensor_a = tensor.accessor<int,1>();
    int N2 = torch::size(tensor, 0);

    int N = N2 * 32;
    auto res = torch::zeros(N, torch::kFloat32);
    auto res_a = res.accessor<float,1>();

    for(int i = 0; i < N2; i++){
        unsigned int x = (unsigned int)tensor_a[i];
        for(int j = 0; j < 32; j++){
            int z = (x >> (32 - 1 - j)) & 1;
            if(z > 0){
                res_a[32*i + j] = 1;
            } else {
	        res_a[32*i + j] = -1;
	    }
        }
    }
    return res;
}

PYBIND11_MODULE(q_cpp, m) {
  m.def("quantize_general", &quantize_general, "quantize general");
  m.def("unquantize_general", &unquantize_general, "unquantize general");
}
