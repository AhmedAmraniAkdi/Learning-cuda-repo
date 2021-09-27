// since pairs are independent, we can use streams... for later!

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>


#define ARR_SIZE (1 << 20)

#define BLOCKSIZE 32
#define GRIDSIZE 512
#define SMEM 1024
#define ITEMSPERBLOCK 2048
#define ITEMSPERTHREAD 32 // SMEM / 32
#define FLOAT4LOADS 2// ITEMSPERBLOCK/SMEM


__device__ void seq_merge(float *dest, float *A, int start_a, int end_a, float *B, int start_b, int end_b, int howmany){

    float item_A = A[start_a];
    float item_B = B[start_b];

    #pragma unroll
    for(int i = 0; i < howmany; i++){

        bool p = (start_b < end_b) && ((start_a >= end_a) || item_B <= item_A);

        if(p){
            dest[i] = item_B;
            start_b++;
            if (start_b < end_b)
                item_B = B[start_b];  
        } else {
            dest[i] = item_A;
            start_a++;
            if (start_a < end_a)
                item_A = A[start_a]; 
        }

    }
    
}

__device__ void swap_smem_ptr(float **r, float **s){
    float *pSwap = *r;
    *r = *s;
    *s = pSwap;
}

__global__ void merge_sort_small(float4 *d_input, float4 *d_output){

    __shared__ float A_B[SMEM];
    __shared__ float Out[SMEM]; //ping pong
    // this is wierd but works! - https://stackoverflow.com/questions/3393518/swap-arrays-by-using-pointers-in-c
    float *A_Bptr = A_B;
    float *Outptr = Out;

    int idx = blockIdx.x * ITEMSPERBLOCK / 4;
    d_input += idx;
    d_output += idx;


    #pragma unroll
    for(int i = 0; i < FLOAT4LOADS; i++){
        // diff with sort_v1: merge up to size of smem, instead of multiple times reading from gmem, pinpong with smems
        #pragma unroll
        for(int length = 32; length < SMEM; length = length * 2){ 
            int threadsperpair = length * 2 / ITEMSPERTHREAD; 
            int threadIdx_perpair = threadIdx.x & (threadsperpair - 1);

            if(length == 32){ // first time read from gmem, then from out
                float4 *temp_ab = (float4 *) A_B;
                #pragma unroll
                for(int j = threadIdx.x; j < SMEM / 4; j += 32){
                    temp_ab[j] = d_input[j + i * SMEM/4];
                }
            } else {
                swap_smem_ptr(&A_Bptr, &Outptr);
            }

            int x2 = length, y2 = length;
            int offset_A = threadIdx.x / threadsperpair * length * 2; 
            int offset_B = offset_A + length;

            if(threadIdx_perpair != (threadsperpair - 1)){
                
                int diag = (threadIdx_perpair + 1) * length * 2 / threadsperpair; // diagonal relative to how many threads process a block
                int atop = diag > length ? length : diag;
                int btop = diag > length ? diag - length : 0;
                int abot = btop;
                
                int ai, bi;
                int offset;

                while(1){
                    offset = (atop - abot)/2;
                    ai = atop - offset;
                    bi = btop + offset;

                    if (ai >= length || bi == 0 || A_Bptr[offset_A + ai] > A_Bptr[offset_B + bi - 1]){
                        if(ai == 0 || bi >= length || A_Bptr[offset_A + ai - 1] <= A_Bptr[offset_B + bi]){
                            x2 = ai;
                            y2 = bi;
                            break;
                        } else {
                            atop = ai - 1;
                            btop = bi + 1; 
                        }
                    } else {
                        abot = ai + 1;
                    }
                }
            
            }

            int x1 = __shfl_sync(0xffffffff, x2, threadIdx.x - 1);
            int y1 = __shfl_sync(0xffffffff, y2, threadIdx.x - 1);
            x1 = threadIdx_perpair == 0 ? 0 : x1;
            y1 = threadIdx_perpair == 0 ? 0 : y1;

            seq_merge(Outptr + threadIdx.x * ITEMSPERTHREAD, A_Bptr + offset_A, x1, x2, A_Bptr + offset_B, y1, y2, ITEMSPERTHREAD);


        }
        float4 *temp_smem_out = (float4 *) Out;
        #pragma unroll
        for(int j = threadIdx.x; j < SMEM / 4; j += 32){
            d_output[j + i * SMEM/4] = temp_smem_out[j]; // at the end the result is in out
        }

    }

}


__device__ float swap(int x, int mask, int dir){
    float y = __shfl_xor_sync(0xffffffff, x, mask);
    return x < y == dir ? y : x;
}

__device__ unsigned int bfe(unsigned int x, unsigned int bit, unsigned int num_bits=1){
    return (x >> bit) & 1;
}

__global__ void warpsize_bitonic_sort(float *d_input){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float x = d_input[idx];

    x = swap(x, 0x01, bfe(threadIdx.x, 1) ^ bfe(threadIdx.x, 0)); // 2
    x = swap(x, 0x02, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 1)); // 4
    x = swap(x, 0x01, bfe(threadIdx.x, 2) ^ bfe(threadIdx.x, 0));
    x = swap(x, 0x04, bfe(threadIdx.x, 3) ^ bfe(threadIdx.x, 2)); // 8
    x = swap(x, 0x02, bfe(threadIdx.x, 3) ^ bfe(threadIdx.x, 1));
    x = swap(x, 0x01, bfe(threadIdx.x, 3) ^ bfe(threadIdx.x, 0));
    x = swap(x, 0x08, bfe(threadIdx.x, 4) ^ bfe(threadIdx.x, 3)); // 16
    x = swap(x, 0x04, bfe(threadIdx.x, 4) ^ bfe(threadIdx.x, 2));
    x = swap(x, 0x02, bfe(threadIdx.x, 4) ^ bfe(threadIdx.x, 1));
    x = swap(x, 0x01, bfe(threadIdx.x, 4) ^ bfe(threadIdx.x, 0));
    x = swap(x, 0x10, bfe(threadIdx.x, 5) ^ bfe(threadIdx.x, 4)); // 32
    x = swap(x, 0x08, bfe(threadIdx.x, 5) ^ bfe(threadIdx.x, 3));
    x = swap(x, 0x04, bfe(threadIdx.x, 5) ^ bfe(threadIdx.x, 2));
    x = swap(x, 0x02, bfe(threadIdx.x, 5) ^ bfe(threadIdx.x, 1));
    x = swap(x, 0x01, bfe(threadIdx.x, 5) ^ bfe(threadIdx.x, 0));

    d_input[idx] = x;
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
                order = 0; break;
            }
        }
        if(!order) {break;}
    }

    printf("\nordered %d\n", order);
}


int main(void){

    srand(0);

    float *h_input;
    float *d_input;

    h_input = (float*) malloc(ARR_SIZE * sizeof(float));
     
    fill_array(h_input);

    /*for(int i = 0; i < 1024; i++){
        if(i % 32 == 0 && i != 0){
            printf("\n");
        }
        printf("%.0f ", h_input[i]);
    }*/

    printf("\n----------\n");
    printf("----------\n");
    printf("----------\n");

    cudaMalloc((void **)&d_input, ARR_SIZE * sizeof(float));
    checkCudaErrors(cudaGetLastError());
    
    cudaMemcpy(d_input, h_input, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());

    //////////////////////////

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    float total_time = 0;

    int *diag_A, *diag_B;
    cudaMalloc((void **)&diag_A, GRIDSIZE * sizeof(float));
    checkCudaErrors(cudaGetLastError());
    cudaMalloc((void **)&diag_B, GRIDSIZE * sizeof(float));
    checkCudaErrors(cudaGetLastError());

    float *ping_pong;
    cudaMalloc((void **)&ping_pong, ARR_SIZE * sizeof(float));
    checkCudaErrors(cudaGetLastError());

    cudaEventRecord(start, 0);
    warpsize_bitonic_sort<<<ARR_SIZE/32, 32>>>(d_input);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    printf( "warpsize bitonic sort: %.8f ms\n", elapsed_time);

    cudaEventRecord(start, 0);
    // 32 -> 512 (inclusive)
    merge_sort_small<<<GRIDSIZE, BLOCKSIZE>>>((float4 *) d_input, (float4 *) ping_pong);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    printf( "merge sort using merge path: %.8f ms\n", elapsed_time);
    printf("total time:%f\n", total_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /////////////////////////

    cudaMemcpy(h_input, ping_pong,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());

    /*for(int i = 0; i < 1024; i++){
        if(i % 32 == 0 && i != 0){
            printf("\n");
        }
        printf("%.0f ", h_input[i]);
    }*/

    check(h_input, 1024);

    cudaFree(d_input);
    free(h_input);

    return 0;
}