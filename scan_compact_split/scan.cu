#include "scan_header.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>


__device__ void sweeps(float *d_input, float *intermediate_sum, float *d_output, int step){

}

__device__ void scanBlockSums(float *d_input, float *d_output, int step){

}

__device__ void sumScannedSum(float *scanned_sum, float *output, int step){

}

void scan(float* d_input, float* d_output, int arr_size){

    int temp = (arr_size/sizeof(float) + STEP * BLOCKDIM - 1)/STEP/BLOCKDIM;
    dim3 dimBlock(BLOCKDIM);
    dim3 dimGrid(temp);

    float *intermediate_sum;
    float *intermediate_sum_scanned;
    cudaMalloc((void **)&intermediate_sum, arr_size);
    cudaMalloc((void **)&intermediate_sum_scanned, arr_size);

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

void fill_array(float *h_input, int arr_size){
    for(int i = 0; i < arr_size; i++){
        h_input[i] = rand() % 10;
    }
}

int check_solution(float *h_input, float * h_output, int arr_size){
    float *temp;
    temp = (float*) malloc(arr_size);
    
    temp[0] = 0;
    for(int i = 1; i < arr_size/sizeof(float); i++){
        temp[i] = temp[i - 1] + h_input[i - 1];
    }

    int correct = 1;
    for(int i = 0; i < arr_size/sizeof(float); i++){
        if(temp[i] != h_output[i]){
            correct = 0;
            break;
        }
    }

    return correct;
}


int main(void){

    srand(0);

    float *h_input, *h_output;
    float *d_input, *d_output;
    int arr_size = SIZE_ARRAY * sizeof(float);

    h_input = (float*) malloc(arr_size);
    d_output = (float*) malloc(arr_size);

    fill_array(h_input, arr_size);

    cudaMalloc((void **)&d_input, arr_size);
    cudaMalloc((void **)&d_output, arr_size);

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