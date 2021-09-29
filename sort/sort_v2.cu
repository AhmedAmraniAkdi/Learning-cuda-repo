// since pairs are independent, we can use streams... for later!

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>


#define ARR_SIZE (1 << 20)
#define LOGARRSIZE 20
#define BLOCKSIZE 32
#define GRIDSIZE 1024 // ARR_SIZE/SMEM
#define SMEM 1024
#define LOGSMEM 10
// diff 2 with sort_v1: if itemsperblock == smem, no need for medium merge kernel nor multiple loads, can load everything a block processes on smem
#define ITEMSPERBLOCK 1024 
#define ITEMSPERTHREAD 32 // SMEM / 32
#define FLOAT4LOADS 1// ITEMSPERBLOCK/SMEM


/*************************************************************************
 * **********************************************************************/
/*
__device__ void swap_smem_ptr(float **r, float **s){
    float *pSwap = *r;
    *r = *s;
    *s = pSwap;
}

__global__ void merge_sort_small(float4 *d_input, float4 *d_output){
    
    //__shared__ int offset_A;
    //__shared__ int offset_B;

    __shared__ float4 A_B[SMEM/4];
    __shared__ float4 Out[SMEM/4]; //ping pong
    // this is wierd but works! - https://stackoverflow.com/questions/3393518/swap-arrays-by-using-pointers-in-c
    float *A_Bptr = (float *) A_B;
    float *Outptr = (float *) Out;

    __shared__ float temp_A[32];
    __shared__ float temp_B[32];

    int idx = blockIdx.x * ITEMSPERBLOCK / 4;
    d_input += idx;
    d_output += idx;

    // diff 1 with sort_v1: merge up to size of smem, instead of multiple times reading from gmem, pinpong with smems
    #pragma unroll
    for(int length = 32; length < SMEM; length = length * 2){
        if(length == 32){ // first time read from gmem, then from out
            #pragma unroll
            for(int j = threadIdx.x; j < SMEM / 4; j += 32){
                A_B[j] = d_input[j];
            }
        } else {
            swap_smem_ptr(&A_Bptr, &Outptr);
        }

        // diff 3 : no seq merge, merge 32 items concurrently steps times using binary search
        // no more slick warp shuffle :(
        #pragma unroll
        for(int pairs = 0; pairs < SMEM/length/2; pairs++){

            int offset_A = pairs * length * 2;
            int offset_B = offset_A + length;

            int end_A = offset_B;
            int end_B = offset_B + length;

            int x = 0;
            int y = 0;

            #pragma unroll
            for(int step = 0; step < length * 2 / 32; step++){

                offset_A += x;
                offset_B += y;
                
                int As = end_A - offset_A;
                int Bs = end_B - offset_B;

                if (As < 0)  As = 0;
                if (As > 32) As = 32;
                if (Bs < 0)  Bs = 0;
                if (Bs > 32) Bs = 32;

                int diag = threadIdx.x;
                int atop = diag;
                int btop = 0;
                int abot = btop;

                temp_A[threadIdx.x] = FLT_MAX; // easier to reason with
                temp_B[threadIdx.x] = FLT_MAX;

                if(threadIdx.x < As) temp_A[threadIdx.x] = A_Bptr[offset_A + threadIdx.x]; 
                if(threadIdx.x < Bs) temp_B[threadIdx.x] = A_Bptr[offset_B + threadIdx.x];
                

                while(1){
                    // a crazy idea would be to use shuffle functions
                    int offset = (atop - abot)/2;
                    int ai = atop - offset;
                    int bi = btop + offset;

                    if (ai >= 32 || bi == 0 || temp_A[ai] > temp_B[bi - 1]){
                            
                        if(ai == 0 || bi >= 32 ||  temp_A[ai - 1] <=  temp_B[bi]){

                            if (ai < 32 && (bi == 32 || temp_A[ai] <= temp_B[bi])){

                                Outptr[pairs * length * 2 + step * 32 + threadIdx.x] = temp_A[ai];
                                ai++;

                            } else {
                                
                                Outptr[pairs * length * 2 + step * 32 + threadIdx.x] = temp_B[bi]; // we merge 32 ateach step
                                bi++;

                            }

                            x = __shfl_sync(0xffffffff, ai, 31); 
                            y = __shfl_sync(0xffffffff, bi, 31); 

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
        }

    }
    #pragma unroll
    for(int j = threadIdx.x; j < SMEM / 4; j += 32){
        d_output[j] = Out[j]; // at the end the result is in out
    }
}

*/
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
        // diff 1 with sort_v1: merge up to size of smem, instead of multiple times reading from gmem, pinpong with smems
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

/*************************************************************************
 * **********************************************************************/


__global__ void merge_sort_bigger_than_smem(float *d_input, int length, float *d_output, int *diag_A, int *diag_B, int blocksperpair){


    __shared__ float A[SMEM]; // we can put A and B together since max items per block is SMEM, for later
    __shared__ float B[SMEM];
    __shared__ float Out[SMEM];

    #pragma unroll
    for(int k = threadIdx.x; k < SMEM; k+=32){
        A[k] = FLT_MAX;
        B[k] = FLT_MAX;
    }

    int threadIdx_perblocksperpair = threadIdx.x & (blocksperpair - 1);

    int offset_A2 = diag_A[blockIdx.x];
    int offset_B2 = diag_B[blockIdx.x];

    int offset_A1 = threadIdx_perblocksperpair == 0? 0 : diag_A[blockIdx.x - 1];
    int offset_B1 = threadIdx_perblocksperpair == 0? 0 : diag_B[blockIdx.x - 1];

    d_input += threadIdx.x/blocksperpair * 2 * length;
    d_output += threadIdx.x/blocksperpair * 2 * length;

    int countA = offset_A2 - offset_A1;
    int countB = offset_B2 - offset_B1;

    #pragma unroll
    for(int k = threadIdx.x; k < countA; k+=32){
        A[k] = d_input[offset_A1 + k];
    }

    #pragma unroll
    for(int k = threadIdx.x; k < countB; k+=32){
        B[k] = d_input[offset_B1 + length + k];
    }

    int x1 = 0, y1 = 0;
    int x2 = 0, y2 = 0;
        
    int diag = (threadIdx.x + 1) * SMEM / BLOCKSIZE;
    int atop = diag > length ? length : diag;
    int btop = diag > length ? diag - length : 0;
    int abot = btop;
    
    int ai, bi;
    int offset;

    while(1){
        offset = (atop - abot)/2;
        ai = atop - offset;
        bi = btop + offset;

        if (ai >= length || bi == 0 || A[ai] > B[bi - 1]){
            if(ai == 0 || bi >= length || A[ai] <= B[bi]){
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

    int x1 = __shfl_sync(0xffffffff, x2, threadIdx.x - 1);
    int y1 = __shfl_sync(0xffffffff, y2, threadIdx.x - 1);
    x1 = threadIdx.x == 0 ? 0 : x1;
    y1 = threadIdx.x == 0 ? 0 : y1;

    seq_merge(Out + threadIdx.x * ITEMSPERTHREAD, A, x1, x2, B, y1, y2, ITEMSPERTHREAD);


    #pragma unroll
    for(int k = threadIdx.x; k < SMEM; k += 32){
        d_output[threadIdx_perblocksperpair * SMEM + k] = Out[k]; 
    }

}

__global__ void grid_partition_path(float *d_input, int length, int *diag_A, int *diag_B, int blocksperpair){

    int threadIdx_perblocksperpair = threadIdx.x & (blocksperpair - 1);
    float *A = d_input + threadIdx.x/blocksperpair * 2 * length;
    float *B = A + length;
    
    // blocksperarray blocks process the array
    // so each blocksperarray_i block starts at 0
    if(threadIdx_perblocksperpair == blocksperpair - 1){
        diag_A[threadIdx.x] = length;
        diag_B[threadIdx.x] = length;
    } else {
    
        int diag = (threadIdx_perblocksperpair + 1) * length * 2 / blocksperpair; // where it starts instead of where it ends
        int atop = diag > length ? length : diag;
        int btop = diag > length ? diag - length : 0;
        int abot = btop;

        int ai, bi;
        int offset;

        while(1){

            offset = (atop - abot)/2;
            ai = atop - offset;
            bi = btop + offset;

            if (ai >= length || bi == 0 || A[ai] > B[bi - 1]){
                if(ai == 0 || bi >= length || A[ai - 1] <= B[bi]){
                    diag_A[threadIdx.x] = ai;
                    diag_B[threadIdx.x] = bi;
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

}


/*************************************************************************
 * **********************************************************************/

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

    /*for(int i = 0; i < 64; i++){
        if(i % 32 == 0 && i != 0){
            printf("\n");
        }
        printf("%.0f ", h_input[1024+i]);
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
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkCudaErrors(cudaGetLastError());
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    printf( "warpsize bitonic sort: %.8f ms\n", elapsed_time);

    cudaEventRecord(start, 0);
    // 32 -> 512 (inclusive)
    merge_sort_small<<<GRIDSIZE, BLOCKSIZE>>>((float4 *) d_input, (float4 *) ping_pong);

    for(int step = LOGSMEM; step < LOGARRSIZE; step++){
        int blocksperpair = (1 << (step + 1))/SMEM;
        if((step - LOGSMEM) & 1){ // easier than swapping
            grid_partition_path<<<1, GRIDSIZE>>>(d_input, (1 << step), diag_A, diag_B, blocksperpair);
            merge_sort_bigger_than_smem<<<GRIDSIZE, BLOCKSIZE>>>(d_input, (1 << step), ping_pong, diag_A, diag_B, blocksperpair);
        } else {
            grid_partition_path<<<1, GRIDSIZE>>>(ping_pong, (1 << step), diag_A, diag_B, blocksperpair);
            merge_sort_bigger_than_smem<<<GRIDSIZE, BLOCKSIZE>>>(ping_pong, (1 << step), d_input, diag_A, diag_B, blocksperpair);
        }
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    checkCudaErrors(cudaGetLastError());
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    printf( "merge sort using merge path: %.8f ms\n", elapsed_time);
    printf("total time:%f\n", total_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /////////////////////////
    
    if((LOGARRSIZE - LOGSMEM - 1) & 1){
        cudaMemcpy(h_input, ping_pong,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());
    } else {
        cudaMemcpy(h_input, d_input,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaGetLastError());
    }

    /*for(int i = 0; i < 1024; i++){
        if(i % 32 == 0 && i != 0){
            printf("\n");
        }
        printf("%.0f ", h_input[1024+i]);
    }*/

    check(h_input, ARR_SIZE);

    cudaFree(d_input);
    cudaFree(ping_pong);
    cudaFree(diag_A);
    cudaFree(diag_B);
    free(h_input);

    return 0;
}