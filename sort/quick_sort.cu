#define ARR_SIZE (1 << 25)
#define MAX_DEPTH 24

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>
#include <compact.cuh>

/*

    4 5 8 9 8 5 1 2 3 1 6 pivot = 6
    i
   s

    4 5 8 9 8 5 1 2 3 1 6 pivot = 6
    i
    s

    4 5 8 9 8 5 1 2 3 1 6 pivot = 6
      i
      s

    4 5 8 9 8 5 1 2 3 1 6 pivot = 6
        i
      s
    
    4 5 8 9 8 5 1 2 3 1 6 pivot = 6
            i
      s
    
    4 5 5 9 8 8 1 2 3 1 6 pivot = 6
              i
        s

    4 5 5 9 8 8 1 2 3 1 6 pivot = 6
                i
          s

    ...
*/

__device__ void swap(float *d_input, float* a, float* b){

}


__device__ void selection_sort(float *d_input, int left, int right){

}

__global__ void quicksort(float *d_input, int left, int right, int depth){

    if(depth == MAX_DEPTH || (right - left <= 32)){
        selection_sort(d_input, left, right);
    } else {


    }

}

// main + interface
void cuda_interface_scan(float* d_input){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    cudaEventRecord(start, 0);
    quicksort<<<1,1>>>(d_input, 0, ARR_SIZE - 1, 0);
    checkCudaErrors(cudaGetLastError());
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

    cudaMalloc((void **)&d_input, ARR_SIZE * sizeof(float));*
    cudaMemcpy(d_input, h_input, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cuda_interface_scan(d_input);

    cudaMemcpy(h_input, d_input,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 1024; i++)
        printf("sum %.8f", h_input[i]);

    cudaFree(d_input);
    free(h_input);

    return 0;
}