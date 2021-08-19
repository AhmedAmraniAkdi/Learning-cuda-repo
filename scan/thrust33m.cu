#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>


void scan(float4* d_input, float4* d_output, int arr_size, float4* h_input){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

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

    float *temp, *h_input1;

    h_input1 = (float*)h_input;

    temp = (float*) malloc(arr_size * sizeof(float));

    temp[0] = h_input1[0];
    for(int i = 1; i < arr_size; i++){
        temp[i] = temp[i - 1] + h_input1[i];
    }

    for(int i = 0; i < 10000; i++)
        std::cout<<temp[i]<<" "<<h_thrust[i]<<"\n";;

    cudaFree(temp_thrust);
    free(h_thrust);
    free(temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}   


void fill_array(float4 *h_input, int arr_size){

    float *temp = (float*) h_input;
    for(int i = 0; i < arr_size; i++){
        temp[i] = (float) rand() / RAND_MAX;
    }
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

    std::cout<<"\n\n\n";

    cudaMalloc((void **)&d_input, arr_size * sizeof(float));
    cudaMalloc((void **)&d_output, arr_size * sizeof(float));

    cudaMemcpy(d_input, h_input, arr_size * sizeof(float), cudaMemcpyHostToDevice);

    scan(d_input, d_output, arr_size, h_input);

    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}