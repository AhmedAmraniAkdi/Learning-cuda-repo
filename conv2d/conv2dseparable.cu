#include <iostream>
#include <stdio.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>

#define IMROWS 1204
#define IMCOLS 1880 
#define IMCHANNELS 1
#define KERNELRADIUS 16 // 16x2 + 1
#define BLOCKSIZE 16

__constant__ float c_kernel[KERNELRADIUS * 2 + 1];

__global__ void conv2d(float *d_input, float *d_output){

};

void processing(float* h_input, float *h_output, float *h_kernel){
    // variables for device
    float *d_input,*d_output, *d_kernel;
    int buf_size = IMROWS * IMCOLS * sizeof(float);
    int kernel_size = (KERNELRADIUS*2+1) * sizeof(float);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocate device mem
    cudaMalloc((void **)&d_input, buf_size);
    cudaMalloc((void **)&d_output, buf_size);

    // send data
    cudaMemcpy(d_input, h_input, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size);

    // where magic happens
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d(d_input, d_output);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    cudaProfilerStop();
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));
    
    // return data
    cudaMemcpy(h_output, d_output, buf_size, cudaMemcpyDeviceToHost);

    // free
    free(d_input);
    free(d_output);


}
