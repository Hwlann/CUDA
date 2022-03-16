
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <random>

__constant__ double device_b = 3.1415*3.1415;

__device__ double get_distance(double a, double b)
{
    return sqrt(pow(a,2) +pow(b,2));
}

__global__ void kernel_compute_distance(double *distance, const double* a, size_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
         //distance[i] = get_distance(a[i], device_b); // 1,01 cycles /ns
         distance[i] = sqrtf(powf(a[i], 2) + device_b); //device_b squared in once in memory 985.34 cycles/µs
         //distance[i] = sqrtf(powf(a[i], 2) + powf(device_b, 2)); // 1 cycle/ns
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
    const int h_array_size = 1000000;
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
    /*
    cudaStatus = cudaMalloc((void**)&h_b, sizeof(double));
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA - Malloc (b) on global memory failed !\n");
        cuda_error(cudaStatus);
    }
    */
    cudaStatus = cudaMalloc((void**)&device_distance, h_array_size * sizeof(double));
    if(cudaStatus != cudaSuccess) {
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
    cudaStatus = cudaMemcpy(&device_b, &h_b, sizeof(double), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA - Memcpy to device (b) failed !\n");
        cuda_error(cudaStatus);
    }
    */

    /*
    Kernel Variables
    */
    dim3 dimGrid = ceil(h_array_size/1024)+1;
    dim3 dimBlock = 1024;
    /*
        Execute Kernel
    */
    kernel_compute_distance << <dimGrid, dimBlock >> > (device_distance, device_a, h_array_size);

    /*
        Wait kernel ends
    */
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
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
