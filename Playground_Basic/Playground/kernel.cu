
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <random>

__constant__ double b = 3.141596;

__device__ double get_distance(double a, double b)
{
    return sqrt(pow(a,2) +pow(b,2));
}

__global__ void kernel_compute_distance(double *distance, const double* a, const double *b)
{
    int i = threadIdx.x;
    //distance[i] = get_distance(a[i], b);
    distance[i] = a[i] + b[i];
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
    const int h_array_size = 1024;
    double h_a[h_array_size] = {0};
    double h_c[h_array_size] = {0};
    double h_distance[h_array_size] = { 0 };
    
    
    /*
        Init array with random values
    */
    for (int i = 0 ; i < h_array_size; i++)
    {
        h_a[i] = (double)i-h_array_size;
        h_c[i] = (double)i;
    }

    /*
        Device Variables
    */
    double* device_a = 0;
    double* device_c = 0;
    double* device_distance = 0;
    /*
        Cuda Mallocs    
    */
    if (cudaMalloc((void**)&device_a, h_array_size * sizeof(double))) {
        fprintf(stderr, "CUDA - Malloc (a) on global memory failed !");
    }
    if (cudaMalloc((void**)&device_c, h_array_size * sizeof(double))) {
        fprintf(stderr, "CUDA - Malloc (c) on global memory failed !");
    }
    if (cudaMalloc((void**)&device_distance, h_array_size * sizeof(double))) {
        fprintf(stderr, "CUDA - (distance) Malloc on global memory failed !");
    }
    /*
    if (cudaMalloc((void**)&b, sizeof(double))) {
        fprintf(stderr, "CUDA - Malloc on constant memory failed !");
    }
    */

    /**
        Cuda Memcpy
    */
    if (cudaMemcpy(device_a, h_a, h_array_size*sizeof(double), cudaMemcpyHostToDevice)) {
        fprintf(stderr, "CUDA - Memcpy to device (a) failed !");
    }
    if (cudaMemcpy(device_c, h_c, h_array_size * sizeof(double), cudaMemcpyHostToDevice)) {
        fprintf(stderr, "CUDA - Memcpy to device (a) failed !");
    }

    /*
        Kernel Variables
    */
    dim3 dimGrid = 48;
    dim3 dimBlock = 32;
    /*
        Execute Kernel
    */
    kernel_compute_distance <<<dimGrid, dimBlock>>> (device_distance, device_a, device_c);

    /*
        Wait kernel ends
    */
    if(!cudaDeviceSynchronize()) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
    printf("Finished !\n");

    /*
        Gather result
    */
    if (cudaMemcpy(h_distance, device_distance, h_array_size * sizeof(double), cudaMemcpyDeviceToHost)) {
        fprintf(stderr, "CUDA - Memcpy to device (a) failed !");
    }

    /*
        Free VRAM
    */
    cudaFree(device_a);
    cudaFree(device_c);
    cudaFree(device_distance);

    printf("Distance at 0 : %f\n", h_distance[0]);

    return cudaStatus;
}
