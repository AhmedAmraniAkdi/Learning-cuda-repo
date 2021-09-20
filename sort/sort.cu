#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>


#define ARR_SIZE (1 << 20)
#define BLOCKSIZE1 (1 << 5)
#define GRIDSIZE1 (1 << 15)
#define LOG2BLOCKSIZE1 5
#define LOG2ARR_SIZE 20

#define BLOCKSIZE2 (1 << 5)
#define GRIDSIZE2 (1 << 8)
#define SMEM_SIZE 512
#define Stride 16

/*
    merge path configuration:

    ok, hear me out on this one :

    256 blocks of 32 threads
    
    64Kb of smem per sm, 5 sm, 
    we will take 512 smem elems per block making it 256 elements of A, and 256 elements of B

    we will take a stride of 16 elements per thread

    0) first iteration
        we start merging arrays of length 32 , that makes it 2^14 pairs to sort
        2^6 pairs per block
        2^11 (x32) elements of A's and 2^11 elements of B's - at 256 elements smem's 
        -> 8 loads of A's and B's and 8 pairs fit per iter (256/32)

        512 total elements on 1 load / 32 threads = 16 elements per thread = stride -> all smem consumed on 1 iteration

        64 total elements(A+B) / 16 elements per thread = 4 threads per pair
        -> 8 pairs total (32 threads)

    1) second iteration
        length 64, 2^13 pairs
        2^5 pairs per block
        2^11 elements of A's and 2^11 B's, 8 loads

        128 total elements(A+B) / 16 elements per thread = 8 threads per pair
        -> 4 pairs total

    at 8) we start will need 2 blocks for a pair -> will need grid_partition_path

*/

__device__ void seq_merge(float *dest, float *A, int start_a, int end_a, float *B, int start_b, int end_b){


    
}

__global__ void merge_sort(float *d_input, int length, float *diag_A, float *diag_B){



}


/*
    ok, so what's the problem... imagine we are merging 2 sorted arrays A and B...
    and what a single block processes is less than the length of each array...
    we will need for example 2 blocks to merge the 2 arrays...
    block 1 will start from the top left corner finding each intersection of the diagonals with the path
    but block 2 does start where? what's the x and y offsets? we can't have communication between blocks

    the solution: make a gridsize partition when merging, that way each block has it own x, y offset

    inconvenient: we can't use shared memory: the number of elements is too large for it
    convenient: A diag and B diag are gridsize arrays, so small
*/

// gets called onyl when more than 1 block is needed to process the arrays
// 1 block 256 threads
__global__ void grid_partition_path(float *d_input, float *diag_A, float *diag_B, int length, int blocksperarray){

    // get where in d_input we are
    //d_input += something;
    
    float *A = d_input;
    float *B = d_input + length;
    
    // blocksperarray blocks process the array
    // so each blocksperarray_i block starts at 0
    if(threadIdx.x & (blocksperarray - 1)){
        diag_A[threadIdx.x] = 0;
        diag_B[threadIdx.x] = 0;
    }
    
    int diag = (threadIdx.x + 1) * length * 2 / blockDim.x;
    int atop = diag > length ? length : diag;
    int btop = diag > length ? diag - length : 0;
    int abot = btop;

    int ai, bi;
    int offset;

    while(1){

        offset = (atop - abot)/2;
        ai = atop - offset;
        bi = btop + offset;

        if (ai >= 0 && bi <= length && (A[ai] > B[bi - 1] || ai == length || bi == 0)){
            if((A[ai - 1] <= B[bi] || ai == 0 || bi == length)){
                diag_A[threadIdx.x] = ai;
                diag_B[threadIdx.x] = bi;
            } else {
                atop = ai - 1;
                btop = bi + 1; 
            }
        } else {
            abot = ai + 1;
        }
    }

}

/*
__global__ void odd_even_merge_sort(float *d_input){

    __shared__ float s_data[BLOCKSIZE1];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    s_data[threadIdx.x] = d_input[idx];

    __syncthreads();

    int temp;

    #pragma unroll
    for(int p = 1 << (LOG2BLOCKSIZE1 - 1); p > 0; p /= 2) {
        
        int q = 1 << LOG2BLOCKSIZE1;
        int r = 0;

        #pragma unroll
        for (int d = p ; d > 0 ; d = q - p) {

            if(threadIdx.x < BLOCKSIZE1 - d){
            
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
*/



/*
    starting with the merge path from length 1 arrays is a bit overkill...
    what we do is a bitonic sort getting a collection of 32 size sorted arrays, we start merging these
    we will need log N - log 32 merging steps
    why 32? fits nicely with the warpsize - no synchronisation needed and gives us ability to use warp shuffle functions

*/

__device__ float swap(int x, int mask, int dir){
    float y = __shfl_xor_sync(0xffffffff, x, mask);
    return x < y == dir ? y : x;
}

__device__ unsigned int bfe(unsigned int x, unsigned int bit, unsigned int num_bits=1){
    return (x >> bit) & 1;
}

// x0 > x1
// thread 0;  x0; x0 = swap(x0, 1, 0)  ; y = get(xi from 0^1=1) = x1 ; return x0 < x1 == 0 ? x1 : x0 -> x1
// thread 1;  x1; x1 = swap(x1, 1, 1)  ; y = get(x1 from 1^1=0) = x0 ; return x1 < x0 == 1 ? x0 : x1 -> x0

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

// main + interface
void cuda_interface_sort(float* d_input){

    dim3 dimBlock(BLOCKSIZE1);
    dim3 dimGrid(GRIDSIZE1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    float total_time = 0;

    cudaEventRecord(start, 0);
    //odd_even_merge_sort<<<dimGrid, dimBlock>>>(d_input);
    warpsize_bitonic_sort<<<dimGrid, dimBlock>>>(d_input);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    printf( "warpsize bitonic sort: %.8f ms\n", elapsed_time);


    float *diag_A, *diag_B;
    cudaMalloc((void **)&diag_A, GRIDSIZE2 * sizeof(float));
    cudaMalloc((void **)&diag_B, GRIDSIZE2 * sizeof(float));

    cudaEventRecord(start, 0);

    for(int i = LOG2BLOCKSIZE1; i <= LOG2ARR_SIZE; i++){
        // do some if elses
        // grid partition
        // merge

    }
        //merge_sort<<<GRIDSIZE1, BLOCKSIZE1>>>(d_input, (1 << i));
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    printf( "merge sort using merge path: %.8f ms\n", elapsed_time);

    printf("total time:%f\n", total_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}   

void fill_array(float *h_input){

    for(int i = 0; i <  ARR_SIZE; i++){
        h_input[i] = (float) (rand() & 255);
    }
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

    cuda_interface_sort(d_input);

    cudaMemcpy(h_input, d_input,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
      
    checkCudaErrors(cudaGetLastError());

    for(int i = 0; i < 64; i++){
        printf("%f ", h_input[i]);
        if(i == 31){
            printf("\n");
        }
    }
    cudaFree(d_input);
    free(h_input);

    return 0;
}