#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>


#define ARR_SIZE (1 << 20)


void cuda_interface_sort(float* d_input){


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    float total_time = 0;

    cudaEventRecord(start, 0);
    
    float *temp_input_thrust = (float*) d_input;
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(temp_input_thrust);

    thrust::sort(dev_ptr, dev_ptr + ARR_SIZE);

    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    printf( "thrust sort: %.8f ms\n", elapsed_time);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}   

void fill_array(float *h_input){
    for(int i = 0; i <  ARR_SIZE; i++){
        /*if(i < 512)
            h_input[i] = (float) i;
        else if(i >= 512 && i < 1024)
            h_input[i] = (float) i - 512;
        else*/
            h_input[i] = (float) (rand() & 255);
    }
}

void check(float *h_input, int period){
    int order = 1;
    for(int i = 0; i < ARR_SIZE/period; i++){
        for(int j = 1; j < period; j++){
            if (h_input[i * period + j] < h_input[i * period + j - 1]){
                printf("\nfails at pair%d block%d at indx%d (j)%f < (j-1)%f\n", i%(ARR_SIZE/period/256), i/256, j, h_input[i * period + j], h_input[i * period + j - 1]);
                order = 0; break;
            }
        }
        //if(!order) {break;}
    }

    printf("\nordered %d\n", order);
}


int main(void){

    srand(0);

    float *h_input;
    float *d_input;

    h_input = (float*) malloc(ARR_SIZE * sizeof(float));
     
    fill_array(h_input);

    cudaMalloc((void **)&d_input, ARR_SIZE * sizeof(float));
    checkCudaErrors(cudaGetLastError());
    
    cudaMemcpy(d_input, h_input, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());

    cuda_interface_sort(d_input);

    cudaMemcpy(h_input, d_input,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());

    check(h_input, ARR_SIZE);

    cudaFree(d_input);
    free(h_input);

    return 0;
}