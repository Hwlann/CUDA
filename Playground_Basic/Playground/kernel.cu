
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <random>

__constant__ double h_b = 3.00;

__device__ double get_distance(double a, double b)
{
    return sqrt(pow(a,2) +pow(b,2));
}

__global__ void kernel_compute_distance(double *distance, const double* a, const double b)
{
    int i = threadIdx.x;
    distance[i] = get_distance(a[i], b);
}


int main()
{
    /*
        Init CUDA Api (200ms)
    */
    cudaFree(0);

    /*
        Host variables
    */
    const int h_array_size = 1024;
    double h_a[h_array_size] = {0};
    double h_distance[h_array_size] = {0};
    
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
    double device_b = 0;
    double* device_distance = 0;
    /*
        Cuda Mallocs    
    */
    if (cudaMalloc((void**)&device_a, h_array_size * sizeof(double))) {
        fprintf(stderr, "CUDA - Malloc (a) on global memory failed !\n");
    }
    if (cudaMalloc((void**)&h_b, sizeof(double))) {
        fprintf(stderr, "CUDA - Malloc (b) on global memory failed !\n");
    }
    if (cudaMalloc((void**)&device_distance, h_array_size * sizeof(double))) {
        fprintf(stderr, "CUDA - (distance) Malloc on global memory failed !\n");
    }

    /*
        Cuda Memcpy
    */
    if (cudaMemcpy(device_a, h_a, h_array_size*sizeof(double), cudaMemcpyHostToDevice)) {
        fprintf(stderr, "CUDA - Memcpy to device (a) failed !\n");
    }

    if(cudaMemcpyToSymbol(&device_b, &h_b, sizeof(double), cudaMemcpyHostToDevice)) {
        fprintf(stderr, "CUDA - Memcpy to device (b) failed !\n");
        goto Error;
    }

    /*
    Kernel Variables
    */
    dim3 dimGrid = 48;
    dim3 dimBlock = 32;
    /*
        Execute Kernel
    */
    kernel_compute_distance << <dimGrid, dimBlock >> > (device_distance, device_a, device_b);

    /*
        Wait kernel ends
    */
    if (!cudaDeviceSynchronize()) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
    printf("Finished !\n");

    /*
        Gather result
    */
    if (cudaMemcpy(h_distance, device_distance, h_array_size * sizeof(double), cudaMemcpyDeviceToHost)) {
        fprintf(stderr, "CUDA - Memcpy to host (distance) failed !\n");
    }
        
    /*
        Free VRAM
    */
    Error :
        cudaFree(device_a);
        cudaFree(&device_b);
        cudaFree(device_distance);

    printf("Distance at 0 : %f\n", h_distance[0]);

    return 0;
}
