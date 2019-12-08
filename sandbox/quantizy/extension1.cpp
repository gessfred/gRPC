#include <torch/extension.h>

torch::Tensor quantize_shrink(torch::Tensor tensor){
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
            /*
            x |= (x << 1) | (z >= 0)
            */
        }
        res_a[i] = x;
    }
    return res;
}

torch::Tensor unquantize_shrink(torch::Tensor tensor){
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

int main(void) {
    
}