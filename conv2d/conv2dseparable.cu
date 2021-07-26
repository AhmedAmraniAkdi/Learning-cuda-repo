#include <iostream>
#include <stdio.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
#include "conv2dseparable_common.h"

__constant__ float c_kernel[64 + 1];

__global__ void conv2d_row(float *d_input, float *d_output, int img_w, int img_h, int kernelradius){
    
    extern __shared__ float s_data[];

    int idx_x = blockIdx.x * blockDim.x * STEP + threadIdx.x - blockDim.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // neat pointer arithmetic
    d_input += idx_y * img_w + idx_x;
    d_output += idx_y * img_w + idx_x;

    int temp = (STEP + 2) * blockDim.x;

    // main data
    #pragma unroll
    for(int i = 1 ; i <= STEP; i++){
        s_data[threadIdx.y * temp + threadIdx.x + i * blockDim.x] =  d_input[i * blockDim.x];
    }

    // there's an if-else but all the threads in warp evaluate to the same condition 
    // bcs it divisible by blocksize
    // left halo
    s_data[threadIdx.y * temp + threadIdx.x] =  (idx_x >= 0) ? d_input[0] : 0;

    // right halo
    s_data[threadIdx.y * temp + threadIdx.x + (STEP + 1) * blockDim.x] =  (img_w > ((STEP + 1) * blockDim.x + idx_x)) ? d_input[(STEP + 1 ) * blockDim.x] : 0;

    //__syncthreads;

    float sum;

    #pragma unroll
    for(int i = 1; i <= STEP; i++){

        sum = 0;

        #pragma unroll
        for(int j = -kernelradius; j <= kernelradius; j++){
            /*if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0) {
                printf("%f %f %f\n", c_kernel[kernelradius - j], s_data[threadIdx.y * temp + i * blockDim.x + j], sum);
            }*/
            sum += c_kernel[kernelradius + j] * s_data[threadIdx.y * temp + i * blockDim.x + threadIdx.x +j];
        }

        d_output[i * blockDim.x] = sum;

    }
};

__global__ void conv2d_col(float *d_input, float *d_output, int img_w, int img_h, int kernelradius){

    extern __shared__ float s_data[];

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y * STEP + threadIdx.y - blockDim.y;

    // neat pointer arithmetic
    d_input += idx_y * img_w + idx_x;
    d_output += idx_y * img_w + idx_x;

    int temp = (STEP + 2) * blockDim.y;

    // main data
    #pragma unroll
    for(int i = 1 ; i <= STEP; i++){
        s_data[threadIdx.x * temp + threadIdx.y + i * blockDim.y] =  d_input[i * img_w * blockDim.y]; //read vertical - write horizontal
    }


    // there's an if-else but all the threads in warp evaluate to the same condition 
    // bcs it divisible by blocksize
    // left halo
    s_data[threadIdx.x * temp + threadIdx.y] = idx_y >= 0 ? d_input[0] : 0;

    // right halo
    s_data[threadIdx.x * temp + threadIdx.y + (STEP + 1) * blockDim.y] =  (img_h > (STEP + 1) * blockDim.y + idx_y) ? d_input[(STEP + 1 ) * blockDim.y * img_w] : 0;

    //__syncthreads;

    float sum;

    #pragma unroll
    for(int i = 1; i <= STEP; i++){

        sum = 0;

        #pragma unroll
        for(int j = -kernelradius; j <= kernelradius; j++){
            sum += c_kernel[kernelradius + j] * s_data[threadIdx.x * temp + i * blockDim.y + threadIdx.y + j];
        }

        d_output[i * blockDim.y * img_w] = sum;

    }

};

// add err checking
void processing(float* h_input, float *h_output, float *h_kernel, int img_w, int img_h, int kernelradius){
    // variables for device
    float *d_input, *d_intermediate_output, *d_output;
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
    dim3 dimGrid_row(img_w/temp, img_h/BLOCKDIM);  // we padded the img
    // the 2 * halo is inside the 2 * BLOCKDIM - makes it easier to fill
    int shared_mem_size = BLOCKDIM * (BLOCKDIM * STEP + 2 * BLOCKDIM) * sizeof(float);

    //std::cout<< img_w/temp << " " << img_h/BLOCKDIM << " " << BLOCKDIM * (BLOCKDIM * STEP + 2 * BLOCKDIM);

    // where magic happens
    // row
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_row<<<dimGrid_row, dimBlock, shared_mem_size, 0>>>(d_input, d_intermediate_output, img_w, img_h, kernelradius);
    cudaProfilerStop();
    sdkStopTimer(&timer);
    checkCudaErrors(cudaGetLastError());
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));

    dim3 dimGrid_col(img_w/BLOCKDIM, img_h/temp);
    
    //col
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_col<<<dimGrid_col, dimBlock, shared_mem_size, 0>>>(d_intermediate_output, d_output, img_w, img_h, kernelradius);
    cudaProfilerStop();
    sdkStopTimer(&timer);
    checkCudaErrors(cudaGetLastError());
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));
    
    cudaDeviceSynchronize();

    // return data
    cudaMemcpy(h_output, d_output, buf_size, cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_intermediate_output);

}
