#include "scan_header.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>


__device__ void sweeps(float *d_input, float *intermediate_sum, float *d_output, int step){

    __shared__ float s_data[BLOCKDIM * STEP];

    int idx = blockDim.x * blockIdx.x * STEP + threadIdx.x;

    d_input += idx;
    d_output += idx;

    #pragma unroll
    for(int i = 0; i < STEP; i++){
        s_data[i * BLOCKDIM + threadIdx.x] = d_input[i * BLOCKDIM];
    }

    __syncthreads();

    /* each thread working 4 times (STEP = 4)

        |512|512|512|512|

        after the up sweep, we will have to sum each sub block last element, 
        which will add log2STEP iterations.

        first loop is for the subblocks of 512 (9 iterations)

        second loop is for the 2 last iterations

        total is 11 (log2 (4*512))
    */
    #pragma unroll
    for(int offset = 1; offset < LOG2_BLOCKSIZE; offset <<= 1){

        #pragma unroll
        for(int i = 0; i < STEP; i++){
            
        }

        __syncthreads();
    }

}


__device__ void scanBlockSums(float *d_input, float *d_output, int step){

}


__device__ void sumScannedSum(float *scanned_sum, float *output, int step){

}


void scan(float4* d_input, float4* d_output, int arr_size){

    int temp = (arr_size + BLOCKDIM - 1)/BLOCKDIM;
    dim3 dimBlock(BLOCKDIM);
    dim3 dimGrid(temp);

    float *intermediate_sum;
    float *intermediate_sum_scanned;
    cudaMalloc((void **)&intermediate_sum, arr_size * sizeof(float));
    cudaMalloc((void **)&intermediate_sum_scanned, arr_size * sizeof(float));

    sweeps<<<dimGrid, dimBlock>>>(d_input, intermediate_sum, d_output, STEP);
    cudaDeviceSynchronize();

    int new_step = (temp + BLOCKDIM - 1)/BLOCKDIM;

    scanBlockSums<<<1, BLOCKDIM>>>(intermediate_sum, intermediate_sum_scanned, new_step);
    cudaDeviceSynchronize();

    sumScannedSum<<<dimGrid, dimBlock>>>(intermediate_sum_scanned, d_output, STEP);
    cudaDeviceSynchronize();

    cudaFree(intermediate_sum);
    cudaFree(intermediate_sum_scanned);
}   


void fill_array(float4 *h_input, int arr_size){

    float *temp = (float*) h_input;
    for(int i = 0; i < arr_size; i++){
        temp[i] = rand() & 15;
    }
}


int check_solution(float4 *h_input, float4 *h_output, int arr_size){
    float *temp, *h_input1, *h_output1;

    h_input1 = (float*)h_input;
    h_output1 = (float*)h_output;

    temp = (float*) malloc(arr_size * sizeof(float));
    
    temp[0] = 0;
    for(int i = 1; i < arr_size; i++){
        temp[i] = temp[i - 1] + h_input1[i - 1];
    }

    int correct = 1;
    for(int i = 0; i < arr_size; i++){
        if(temp[i] != h_output1[i]){
            correct = 0;
            break;
        }
    }

    return correct;
}


int main(void){

    srand(0);

    float4 *h_input, *h_output;
    float4 *d_input, *d_output;
    int arr_size = 1000000;

    h_input = (float4*) malloc(arr_size * sizeof(float4));
    d_output = (float4*) malloc(arr_size * sizeof(float4));

    fill_array(h_input, arr_size);

    cudaMalloc((void **)&d_input, arr_size * sizeof(float4));
    cudaMalloc((void **)&d_output, arr_size * sizeof(float4));

    cudaMemcpy(d_input, h_input, arr_size, cudaMemcpyHostToDevice);

    scan(d_input, d_output, arr_size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, arr_size, cudaMemcpyDeviceToHost);

    int correct = check_solution(h_input, h_output, arr_size);

    if(correct){
        std::cout<<"\nCorrect";
    } else {
        std::cout<<"\nNot Correct";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}