#include <cuda_runtime.h>

// Q, K, V, output are device pointers
/*
Constraints
    Matrix Q is of size M×d and matrices K and V are of size N×d
    1 ≤ M, N ≤ 100,000
    1 ≤ d ≤ 1024
*/

const int warp_size = 32;
const int tile_size = 32;
const int coarse_factor = 4;
__global__ void matrix_transpose(const float* input, float* output,int N,int d) {
    __shared__ float tile[tile_size][tile_size+1];

}

__global__ void matrix_multiplication(const float* A, const float*B,float* C,int M,int N,int d) {

}

__global__ void softmax(float* scores,int M,int N,int d) {

}


extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {


}
