#include <torch/extension.h>

torch::Tensor quantize_shrink(torch::Tensor tensor, size_t numberOfThreads){
    auto tensor_a = tensor.accessor<float,1>();
    int N = torch::size(tensor, 0);
    
    //assert N % dataSz == 0
    int N2 = N / 32;
    auto res = torch::zeros(N2, torch::kInt32);
    auto res_a = res.accessor<int,1>();

    for(int i = 0; i < N2; i++){
        int x = 0;
        for(int j = 0; j < 32; j++){
            x = x << 1;
            auto z = tensor_a[32*i + j];
            if(z >= 0){
                x = x | 1;
            }
        }
        res_a[i] = x;
    }
    return res;
}

/*
torch::Tensor quantize_vector(torch::Tensor tensor, size_t numberOfThreads) {
    auto tensor_a = tensor.accessor<float,1>();
    int originalSize = torch::size(tensor, 0);
    int quantizedSize = originalSize / 32;
    auto res = torch::zeros(quantizedSize, torch::kInt32);
    auto res_a = res.accessor<int,1>();
    char work[32];
    #pragma omp parallel for num_threads(numberOfThreads)
    for(unsigned int i = 0; i < originalSize; ++i) {
        for(unsigned int j = 0; j < 32; ++i) {
            res[j] += int(tensor_a[32*i + j]) << j;
        }
    }
    return res;
}*/
/*
torch::Tensor quantize(torch::Tensor tensor, size_t numberOfThreads){
    auto tensor_a = tensor.accessor<float,1>();
    int N = torch::size(tensor, 0);
    
    //assert N % dataSz == 0
    int N2 = N / 32;
    auto res = torch::zeros(N2, torch::kInt32);
    auto res_a = res.accessor<int,1>();
    auto work = torch::zeros(32); 
    #pragma omp parallel for 
    for(int i = 0; i < N2; i++){
        auto x = bitset<32>;
        
        #pragma omp parallel for simd 
        for(int j = 0; j < 32; ++j) {
            x[j] = int(tensor_a[32*i + j] >= 0) * (1 << j)
        }
    }
    for {
        tensor_a[32*i + j] = tensor_a[32*i + j] >= 0
    }
    for {
        tensor_a[32*i + j] = tensor_a[32*i + j] * 2**i
    }
    for {
        tensor_a[32*i + j] = tensor_a[32*i + j] * 2**i
    }
    return res;
}*/
/*
torch::Tensor unquantize_collapse(torch::Tensor tensor, size_t numberOfThreads) {
    auto tensor_a = tensor.accessor<int,1>();
    int shrunkSize = torch::size(tensor, 0);

    //assert N % dataSz == 0
    int bloatedSize = shrunkSize * 32;
    auto res = torch::zeros(bloatedSize, torch::kFloat32);
    auto res_a = res.accessor<float,1>();

    for(int i = 0; i < shrunkSize; i++){
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
*/
torch::Tensor unquantize_shrink(torch::Tensor tensor, size_t numberOfThreads){
    auto tensor_a = tensor.accessor<int,1>();
    int N2 = torch::size(tensor, 0);

    //assert N % dataSz == 0
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
  m.def("quantize_shrink", &quantize_shrink, "quantize shrink");
  m.def("unquantize_shrink", &unquantize_shrink, "unquantize shrink");
}

