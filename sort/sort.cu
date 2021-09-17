#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>


#define ARR_SIZE (1 << 25)
#define BLOCKSIZE (1 << 8)
#define GRIDSIZE (1 << 17)
#define LOG2BLOCKSIZE 8

__global__ void merge_path(float *d_input){

}

__global__ void odd_even_merge_sort(float *d_input){

    __shared__ float s_data[256];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    s_data[threadIdx.x] = d_input[idx];

    __syncthreads();

    int p = 1 << (LOG2BLOCKSIZE - 1);
    int temp;

    #pragma unroll
    for(int p = 1 << (LOG2BLOCKSIZE - 1); p > 0; p /= 2) {
        
        int q = 1 << LOG2BLOCKSIZE;
        int r = 0;

        #pragma unroll
        for (int d = p ; d > 0 ; d = q - p) {

            if(threadIdx.x < BLOCKSIZE - d){
            
                if ((threadIdx.x & p) == r) {
                    if (s_data[threadIdx.x] > s_data[threadIdx.x + d]){
                        temp = s_data[threadIdx.x];
                        s_data[threadIdx.x] = s_data[threadIdx.x + d];
                        s_data[threadIdx.x + d] = temp;
                    }
                }
            }

            q /= 2;
            r = p;

            __syncthreads();
        }

    }

    d_input[idx] = s_data[threadIdx.x];

}

// main + interface
void cuda_interface_sort(float* d_input){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    cudaEventRecord(start, 0);


    // odd even + merge path on loop


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "scan: %.8f ms\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}   

void fill_array(float *h_input){

    for(int i = 0; i <  ARR_SIZE; i++){
        h_input[i] = (float) rand();
    }
}

int main(void){

    srand(0);

    float *h_input;
    float *d_input;

    h_input = (float*) malloc(ARR_SIZE * sizeof(float));
     
    fill_array(h_input);

    cudaMalloc((void **)&d_input, ARR_SIZE * sizeof(float));
    cudaMemcpy(d_input, h_input, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cuda_interface_sort(d_input);

    cudaMemcpy(h_input, d_input,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 1024; i++)
        printf("sum %.8f", h_input[i]);

    cudaFree(d_input);
    free(h_input);

    return 0;
}