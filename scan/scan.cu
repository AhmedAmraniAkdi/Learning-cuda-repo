/*
Reference:
    Parallel Scan for Stream Architectures1
    Duane Merrill Andrew Grimshaw

inclusive scan

*/


#include "scan_header.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>
#include <cuda.h>

__global__ void reduce(float4 *d_input, float *d_output){

    __shared__ float s_data[BLOCKDIM * 2];//1 cell per thread + another blockdim for easier indx management

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    /*if(threadIdx.x == 1 && blockIdx.x == 0){
        printf("%f %f %f %f\n", d_input->w, d_input->x, d_input->y, d_input->z);
    }*/

    d_input += idx;
    d_output += blockIdx.x;

    /*if(threadIdx.x == 1 && blockIdx.x == 0){
        printf("%f %f %f %f\n", d_input->w, d_input->x, d_input->y, d_input->z);
    }*/
    float4 item = *d_input;
    float sum = item.w + item.x + item.y + item.z;

    s_data[threadIdx.x] = sum;

    __syncthreads();

    // we reduce and put the result on the second half of shared memory

    float *a = s_data;

    #pragma unroll
    for(int d = LOG2_BLOCKDIM; d > 0; d--){

        if( threadIdx.x < (1 << (d - 1)) ){
            a[(1 << d) + threadIdx.x] = a[2 * threadIdx.x] + a[2 * threadIdx.x + 1];
        }

        a = &a[(1 << d)];
        __syncthreads();

    }

    // output the sum
    if(threadIdx.x == 0){
        d_output[0] = a[0];
    }
}

// 1 block
__global__ void middle_scan(float *d_input, int iter_per_thread){

    __shared__ float s_data[BLOCKDIM * 2];
    
    float seed = 0;

    if(threadIdx.x == 0){
        seed = d_input[0];
    }
     
    d_input += threadIdx.x;

    // cyclically scan, with the result of each scan becoming the seed to the next
    #pragma unroll
    for(int batch = 0; batch < iter_per_thread; batch++){

        s_data[threadIdx.x] = d_input[batch * iter_per_thread];

        __syncthreads();


        //upsweep
        float *a = s_data;

        #pragma unroll
        for(int d = LOG2_BLOCKDIM; d > 1; d--){ // we don't need last sum, inclusive scan so, the seed = first element

            if( threadIdx.x < (1 << (d - 1)) ){

                a[(1 << d) + threadIdx.x] = a[2 * threadIdx.x] + a[2 * threadIdx.x + 1];

            }

            a += (1 << d);
            __syncthreads();

        }

        if(threadIdx.x == 0){
            a[1] = a[0];
            a[0] = seed; 
        }
        __syncthreads();


        // downsweep
        #pragma unroll
        for(int d = 2; d <= LOG2_BLOCKDIM; d++){

            a -= (1 << d);
            
            if( threadIdx.x < (1 << (d - 1)) ){

                a[2 * threadIdx.x + 1] = a[2 * threadIdx.x] + a[(1 << d) + threadIdx.x];
                a[2 * threadIdx.x] = a[(1 << d) + threadIdx.x];

            }

        __syncthreads();
        }


        d_input[batch * iter_per_thread] = s_data[threadIdx.x];

        if(threadIdx.x == 0){
            seed = s_data[BLOCKDIM - 1]; 
            //printf("%f\n", seed);
        }
    }
}


void scan(float4* d_input, float4* d_output, int arr_size){

    int temp = ((arr_size >> 2) + BLOCKDIM - 1)/BLOCKDIM; // each thread processes 1 float4
    dim3 dimBlock(BLOCKDIM);
    dim3 dimGrid(temp);

    float *d_scan;
    cudaMalloc((void **)&d_scan, temp * sizeof(float));

    reduce<<<dimGrid, dimBlock>>>(d_input, d_scan);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    middle_scan<<<1, dimBlock>>>(d_scan, temp/BLOCKDIM);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    float *h_output = (float*) malloc(temp * sizeof(float));
    cudaMemcpy(h_output, d_scan, temp * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << temp << "\n";
    std::cout << temp/BLOCKDIM << "\n";

    for(int i = 0; i < 5; i++){
        std::cout << h_output[i] << " ";
    }

    cudaFree(d_scan);
}   


void fill_array(float4 *h_input, int arr_size){

    //float *temp = (float*) h_input;
    for(int i = 0; i < arr_size/4; i++){
        //temp[i] = rand() & 15;
        //temp[i] = 1;
        h_input[i].w = 1;
        h_input[i].x = 2;
        h_input[i].y = 3;
        h_input[i].z = 4;
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


    float4 *h_input;
    float4 *d_input;
    int arr_size = 1 << 25;

    h_input = (float4*) malloc(arr_size * sizeof(float));

    fill_array(h_input, arr_size);

    cudaMalloc((void **)&d_input, arr_size * sizeof(float));
    cudaMemcpy(d_input, h_input, arr_size * sizeof(float), cudaMemcpyHostToDevice);

    scan(d_input, NULL, arr_size);

    cudaDeviceSynchronize();

    cudaFree(d_input);
    free(h_input);

    return 0;
}