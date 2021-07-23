#include <iostream>
#include <stdio.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>

#define IMROWS 1204
#define IMCOLS 1880 
#define IMCHANNELS 1
#define KERNELRADIUS 16 // 16x2 + 1

#define BLOCKDIM_1 16
#define BLOCKDIM_2 8
#define STEP 4

__constant__ float c_kernel[KERNELRADIUS * 2 + 1];

__global__ void conv2d_row(float *d_input, float *d_output){

    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ s_data[];

};

__global__ void conv2d_col(float *d_input, float *d_output){



};

// add err checking
void processing(float* h_input, float *h_output, float *h_kernel){
    // variables for device
    float *d_input, *d_intermediate_output, *d_output, *d_kernel;
    int buf_size = IMROWS * IMCOLS * sizeof(float);
    int kernel_size = (KERNELRADIUS*2+1) * sizeof(float);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocate device mem
    cudaMalloc((void **)&d_input, buf_size);
    cudaMalloc((void **)&d_intermediate_output, buf_size);
    cudaMalloc((void **)&d_output, buf_size);

    // send data
    cudaMemcpy(d_input, h_input, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size);

    // dimensions
    int temp = STEP * BLOCKDIM_1;
    dim3 dimBlock_row(BLOCKDIM_1, BLOCKDIM_2);
    dim3 dimGrid_row((IMCOLS + temp - 1)/temp, (IMROWS + BLOCKDIM_2 - 1)/BLOCKDIM_2);

    // where magic happens
    // row
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_row<<<dimGrid_row, dimBlock_row>>>(d_input, d_intermediate_output);
    cudaProfilerStop();
    sdkStopTimer(&timer);
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));

    dim3 dimBlock_col(BLOCKDIM_2, BLOCKDIM_1);
    dim3 dimGrid_col((IMCOLS + BLOCKDIM_2 - 1)/BLOCKDIM_2, (IMROWS + temp - 1)/temp);
    
    //col
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_col<<<dimGrid_col, dimBlock_col>>>(d_intermediate_output, d_output);
    cudaProfilerStop();
    sdkStopTimer(&timer);
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));
    
    cudaDeviceSynchronize();

    // return data
    cudaMemcpy(h_output, d_output, buf_size, cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_intermediate_output);

}
