#include <iostream>
#include <stdio.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>

#define BLOCKDIM 256
#define STEP 4

__constant__ float c_kernel[256];

__global__ void conv2d_row(float *d_input, float *d_output, int img_w, int img_h, int kernelradius){

    
};
__global__ void conv2d_col(float *d_input, float *d_output, int img_w, int img_h, int kernelradius){


};

// add err checking
void processing(float* h_input, float *h_output, float *h_kernel, int img_w, int img_h, int kernelradius){
    // variables for device
    float *d_input, *d_intermediate_output, *d_output, *d_kernel;
    int buf_size = img_w * img_h * sizeof(float);
    int kernel_size = (kernelradius * 2 + 1) * sizeof(float);

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
    int temp = STEP * BLOCKDIM;
    dim3 dimBlock(BLOCKDIM, BLOCKDIM);
    dim3 dimGrid_row((img_w + temp - 1)/temp, (img_h + BLOCKDIM - 1)/BLOCKDIM);

    // where magic happens
    // row
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_row<<<dimGrid_row, dimBlock>>>(d_input, d_intermediate_output, img_w, img_h, kernelradius);
    cudaProfilerStop();
    sdkStopTimer(&timer);
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));

    dim3 dimGrid_col((img_w + BLOCKDIM - 1)/BLOCKDIM, (img_h + temp - 1)/temp);
    
    //col
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_col<<<dimGrid_col, dimBlock>>>(d_intermediate_output, d_output, img_w, img_h, kernelradius);
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
