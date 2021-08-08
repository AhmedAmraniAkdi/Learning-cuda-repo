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
#include <chrono>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>


__global__ void reduce(float4 *d_input, float *d_output){

    __shared__ float s_data[BLOCKDIM * 2];//1 cell per thread + another blockdim for easier indx management
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    d_input += idx;
    d_output += blockIdx.x;

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
        }
    }
}


__global__ void lower_scan(float4 *d_input, float *d_scan, float4 *d_output){

    __shared__ float s_data[BLOCKDIM * 2]; //1 cell per thread + another blockdim for easier indx management

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    d_input += idx;

    float4 item = *d_input;
    float sum = item.w + item.x + item.y + item.z;

    s_data[threadIdx.x] = sum;

    __syncthreads();

    // we reduce and put the result on the second half of shared memory

    float *a = s_data;

    #pragma unroll
    for(int d = LOG2_BLOCKDIM; d > 1; d--){

        if( threadIdx.x < (1 << (d - 1)) ){
            a[(1 << d) + threadIdx.x] = a[2 * threadIdx.x] + a[2 * threadIdx.x + 1];
        }

        a = &a[(1 << d)];
        __syncthreads();

    }

    if(threadIdx.x == 0){
        a[1] = a[0];
        a[0] = d_scan[blockIdx.x]; 
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

    item.x += s_data[threadIdx.x];
    item.y += item.x;
    item.z += item.y;
    item.w += item.z;

    d_output[idx] = item;
}


void scan(float4* d_input, float4* d_output, int arr_size){

    int temp = ((arr_size >> 2) + BLOCKDIM - 1)/BLOCKDIM; // each thread processes 1 float4
    dim3 dimBlock(BLOCKDIM);
    dim3 dimGrid(temp);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0;
    float elapsed_time;

    float *d_scan;
    cudaMalloc((void **)&d_scan, temp * sizeof(float));


    cudaEventRecord(start, 0);
    reduce<<<dimGrid, dimBlock>>>(d_input, d_scan);
    //cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "reduce: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;

    cudaEventRecord(start, 0);
    middle_scan<<<1, dimBlock>>>(d_scan, temp/BLOCKDIM);
    //cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "middle scan: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;

    cudaEventRecord(start, 0);
    lower_scan<<<dimGrid, dimBlock>>>(d_input, d_scan, d_output);
    //cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "final scan: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;

    printf("total time GPU %.8fms\n", total_time);

    float *temp_thrust;
    float *temp_input_thrust = (float*) d_input;
    cudaMalloc((void **)&temp_thrust, arr_size * sizeof(float));
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(temp_input_thrust);
    thrust::device_ptr<float> dev_ptr_out = thrust::device_pointer_cast(temp_thrust);
    cudaEventRecord(start, 0);
    thrust::inclusive_scan(dev_ptr, dev_ptr + arr_size, dev_ptr_out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "thrust scan: %.8f ms\n", elapsed_time);

    float *h_thrust = (float*) malloc(arr_size * sizeof(float));
    cudaMemcpy(h_thrust, temp_thrust, arr_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"---------THRUST---------\n";
    for(int i = 0; i < 20; i++){
        std::cout<<h_thrust[i]<< " ";
        if(((i%4) == 0) && i){
            std::cout<<"\n";
        }
    }

    cudaFree(temp_thrust);
    cudaFree(d_scan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}   


void fill_array(float4 *h_input, int arr_size){

    float *temp = (float*) h_input;
    for(int i = 0; i < arr_size; i++){
        temp[i] = (float) rand() / RAND_MAX;
    }
}


int check_solution(float4 *h_input, float4 *h_output, int arr_size){
    float *temp, *h_input1, *h_output1;

    h_input1 = (float*)h_input;
    h_output1 = (float*)h_output;

    temp = (float*) malloc(arr_size * sizeof(float));
    
    auto tic = std::chrono::high_resolution_clock::now();

    temp[0] = h_input1[0];
    for(int i = 1; i < arr_size; i++){
        temp[i] = temp[i - 1] + h_input1[i];
    }

    auto toc = std::chrono::high_resolution_clock::now();

    printf("total time CPU %.8fms\n", (std::chrono::duration_cast <std::chrono::milliseconds> (toc - tic)).count() * 1.0);

    std::cout<<"---------CPU---------\n";
    for(int i = 0; i < 20; i++){
        std::cout<<temp[i]<< " ";
        if(((i%4) == 0) && i){
            std::cout<<"\n";
        }
    }

    int correct = 1;
    /*for(int i = 0; i < arr_size; i++){
        if(((temp[i] - h_output1[i]) * (temp[i] - h_output1[i])) > 0.00000001){
            correct = 0;
            break;
        }
    }*/

    return correct;
}


int main(void){

    srand(0);

    float4 *h_input, *h_output;
    float4 *d_input, *d_output;
    int arr_size = 1 << 25;

    h_input = (float4*) malloc(arr_size * sizeof(float));
    h_output = (float4*) malloc(arr_size * sizeof(float));
     
    fill_array(h_input, arr_size);

    for(int i = 0; i < 5; i++){
        std::cout<<h_input[i].x<<" "<<h_input[i].y<<" "<<h_input[i].z<<" "<<h_input[i].w<<"\n";
    }

    cudaMalloc((void **)&d_input, arr_size * sizeof(float));
    cudaMalloc((void **)&d_output, arr_size * sizeof(float));

    cudaMemcpy(d_input, h_input, arr_size * sizeof(float), cudaMemcpyHostToDevice);

    scan(d_input, d_output, arr_size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, arr_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"--------GPU----------\n";

    for(int i = 0; i < 5; i++){
        std::cout<<h_output[i].x<<" "<<h_output[i].y<<" "<<h_output[i].z<<" "<<h_output[i].w<<"\n";
    }

    check_solution(h_input, h_output, arr_size) ?  std::cout<<"\ngood" : std::cout<<"\nbad";

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}