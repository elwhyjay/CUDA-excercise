#include <cuda_runtime.h>


/*
Write a program to invert the colors of an image. The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer (unsigned char).

Color inversion is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

The input array image will contain width * height * 4 elements. The first 4 elements represent the RGBA values of the top-left pixel, the next 4 elements represent the pixel to its right, and so on.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array image
*/


__global__ void colorInversion(unsigned char *input, int width,int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pix = width * height;
    if(x < total_pix){
        int index = x * 4;
        input[index] = 255 - input[index];
        input[index + 1] = 255 - input[index + 1];
        input[index + 2] = 255 - input[index + 2];
    }
}

void solve(unsigned char *input, int width, int height){
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    colorInversion<<<blocksPerGrid, threadsPerBlock>>>(input, width, height);
}

