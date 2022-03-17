
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <random>

__constant__ double device_b = 3.1415*3.1415;

__global__ void kernel_compute_distance(double *distance, const double* a, size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
    //for (int i = index; i < N; i += stride)
    if(index < N)
        distance[index] = sqrtf(a[index]*a[index] + device_b);
}

void cuda_error(cudaError_t error)
{
    fprintf(stderr, "CUDA ERROR - %s\n", cudaGetErrorString(error));
    exit(error);
}

int main()
{
    cudaError_t cudaStatus = cudaError_t::cudaSuccess;
    /*
        Init CUDA Api (200ms)
    */
    cudaFree(0);

    /*
        Host variables
    */
    const int h_array_size = 100000000;
    static double h_a[h_array_size] = {0};
    static double h_distance[h_array_size] = {0};
    
    /*
        Init array with random values
    */
    for (int i = 0 ; i < h_array_size; i++)
    {
        h_a[i] = (double)i;
    }

    /*
        Device Variables
    */
    double* device_a = 0;
    double* device_distance = 0;
    /*
        Cuda Mallocs    
    */
    cudaStatus = cudaMalloc((void**)&device_a, h_array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA - Malloc (a) on global memory failed !\n");
        cuda_error(cudaStatus);
    }
    cudaStatus = cudaMalloc((void**)&device_distance, h_array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA - (distance) Malloc on global memory failed !\n");
        cuda_error(cudaStatus);
    }

    /*
        Cuda Memcpy
    */
    cudaStatus = cudaMemcpy(device_a, h_a, h_array_size * sizeof(double), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA - Memcpy to device (a) failed !\n");
        cuda_error(cudaStatus);
    }

    /*
    Kernel Variables
    */
    dim3 dimBlock = 1024;
    dim3 dimGrid = (h_array_size + dimBlock.x - 1) / dimBlock.x;
    /*
        Execute Kernel
    */
    kernel_compute_distance <<<dimGrid, dimBlock>>> (device_distance, device_a, h_array_size);

    /*
        Wait kernel ends
    */
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA - Device Synchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cuda_error(cudaStatus);
    }
    printf("Finished !\n");

    /*
        Gather result
    */
    cudaStatus = cudaMemcpy(h_distance, device_distance, h_array_size * sizeof(double), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA - Memcpy to host (distance) failed !\n");
        cuda_error(cudaStatus);
    }
        
    /*
        Free VRAM
    */
    cudaFree(device_a);
    cudaFree(&device_b);
    cudaFree(device_distance);

    printf("Distance at 0 : %f\n", h_distance[0]);

    return 0;
}
