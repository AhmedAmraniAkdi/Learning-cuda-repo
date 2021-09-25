#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>


#define ARR_SIZE (1 << 20)
#define WARPSIZE (1 << 5)
#define GRIDSIZE1 (1 << 15)
#define LOG2WARPSIZE 5
#define LOG2ARR_SIZE 20

#define BLOCKSIZE (1 << 5)
#define GRIDSIZE2 (1 << 8)
#define STRIDE_THREAD 16 // each thread processes 16 elements
#define STRIDE_BLOCK (1 << 12) // each block processes N/gridsize elements/2 elements of A and elements of B (arrs to merge)
#define LOG2STRIDEBLOCK 12
#define SMEM_SIZE 256
#define LOG2SMEM 8
#define REFILL_LOADS 8 // times we have to refill SMEM = 2^11 elements / 256
#define SMEM_LOADS 8 // 32 threads, 256 smem, need to read 8 times

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
        -> 8 loads of A's and B's and 8 pairs fit  (256/32)

        512 total elements on 1 load / 32 threads = 16 elements per thread = stride -> all smem consumed on 1 iteration

        64 total elements(A+B) / 16 elements per thread = 4 threads per pair
        -> 8 pairs total (32 threads)

    1) second iteration
        length 64, 2^13 pairs
        2^5 pairs per block
        2^11 elements of A's and 2^11 B's, 8 loads

        128 total elements(A+B) / 16 elements per thread = 8 threads per pair
        -> 4 pairs total

    ...

    at 8) we start will need 2 blocks for a pair -> will need grid_partition_path

*/


/*

    Problem: we don't know which 512 elements the block is processing, could be 512 from A and 0 from B, 256/256, 100/412 ...
    we can't just load 256/256 each time... 

    solution: have 1 array of smem with 512 size, we fill it, keep track where A ends and B starts and keep track for the next offsets x, y for the next iteration
    we could have both A, B 512 size smem arrays but we waste smem + could possibly go over the smem limit

    this doesn't change the calculations done in the previous comment.

    But!!!! yes, we have where to start loading but ... when to stop?

    what we will do is: 
    
    1-for L sized segment we need at most L of A and at most B L, but only L elements will be consumed, yes?
    we load L from A, L from B, the last thread gives us the new starting point and we might read the same data again.

    2-we can also use grid_partition_path but we increase A_diag and B_diag and find diagonal point for each smem size

    which one is better?

    1- L sized path needs L elements, we are reading 2L elements, so we will be rereading L elements every time but on coalesced manner!
    + only starts becoming an issue when the arrays to merge don't fit completly on smem, so size 512 A and 512 B
    + possibility to just move the unused items to the start of the arrays instead of rereading (*)
    + check for when either A or B are fully consumed and just start filling with the unfinished array?
    let's say we don't do (*) for 512 sized array, we have 2^10 total pairs, for each pair we read 2*512 elements again, 
    which means we reread the whole array again so N reads - it's 4 am, maybe there is a mistake, but the logic is correct... i believe

    2- A_diag and B_diag become N/SMEM = 2^20/2^9 from 2^8, that's 8 times more and we have to call this kernel many times more
    reads are reaaaly uncoalesced, which means they are serialised -> x32 the number of each warpsize read
    + only starts becoming an issue when the arrays to merge don't fit completly on smem, so size 512 A and 512 B
    paper said we need at most log (size Array) to find partition point ... so total 2^11 * 20 * 32 .. bigger than (1)

    we will stick to 1 and move elements instead of refilling.

*/

// (**) reminder, check for when either A or B are fully consumed and then just directly merge the remaining items of the other array

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

/*

    we will have 3 similar merge sorts :
    1- for when arrays A, B fit into smem so size <256
    2- arrays A and B don't fit so size >=512
    3- when size is > 2^11, 2 or more blocks needed for the A and B

    why? keep code cleaner

    take (maybe i'm wrong): there won't always be enough shared memory, 
    and always have the case of more than 1 block processing 1 pair

    we can however play around the sizes to have more of a case than another

*/

/*

    each new kernell call, the number of threads per pair x2 and the number of pairs /2

*/

__global__ void merge_sort_small(float *d_input, int length){

    __shared__ float A[SMEM_SIZE]; 
    __shared__ float B[SMEM_SIZE]; 
    __shared__ float Out[SMEM_SIZE * 2]; // for coalesced writing 

    // occupancy calculator says no problem with these amounts of smem and registers

    int idx = blockIdx.x * STRIDE_BLOCK + threadIdx.x;

    int swap = length/BLOCKSIZE;

    int offset_A, offset_B;

    int threadsperpair = length * 2 / STRIDE_THREAD; // 4, 8, 16

    int threadIdx_perpair = threadIdx.x & (threadsperpair - 1);

    #pragma unroll
    for(int i = 0; i < REFILL_LOADS; i++){
        offset_A = 0, offset_B = 0;
        #pragma unroll
        for(int j = 0; j < SMEM_LOADS * 2; j++){ // 8 for A, 8 for B
            if(swap & j){ // every other swap elements we switch to B
                B[offset_B + threadIdx.x] = d_input[idx + i * SMEM_SIZE * 2 + j * BLOCKSIZE];
                offset_B += BLOCKSIZE;

            } else {
                A[offset_A + threadIdx.x] = d_input[idx + i * SMEM_SIZE * 2 + j * BLOCKSIZE];
                offset_A += BLOCKSIZE;
            }

        }


        /*if(i == 0 && threadIdx.x == 0 && blockIdx.x == 0){
            for(int j = 0; j < 256; j++){
                printf("%f %f\n", A[j], B[j]);
            }
        }*/

        int x2 = length, y2 = length;
        offset_A = threadIdx.x / threadsperpair * length; // 4 first threads first block, 4 second threads second block etc.
        offset_B = threadIdx.x / threadsperpair * length;

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

                if (ai >= length || bi == 0 || A[offset_A + ai] > B[offset_B + bi - 1]){
                    if(ai == 0 || bi >= length || A[offset_A + ai - 1] <= B[offset_B + bi]){
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

        // we found the points, now we merge
        // look at that slick warp shuffle
        int x1 = __shfl_sync(0xffffffff, x2, threadIdx.x - 1);
        int y1 = __shfl_sync(0xffffffff, y2, threadIdx.x - 1);
        x1 = threadIdx_perpair == 0 ? 0 : x1;
        y1 = threadIdx_perpair == 0 ? 0 : y1;
        
        /*if(i == 0 && threadIdx.x < 4 && blockIdx.x == 0)
            printf("%d %d %d %d %d %d %d %d\n", threadIdx.x, x1, x2, y1, y2, offset_A, offset_B, threadIdx.x * STRIDE_THREAD);*/

        seq_merge(Out + threadIdx.x * STRIDE_THREAD, A + offset_A, x1, x2, B + offset_B, y1, y2, STRIDE_THREAD);
        
        /*if(i == 0 && threadIdx.x == 0 && blockIdx.x == 0)
            printf("**************");

        if(i == 0 && threadIdx.x == 0 && blockIdx.x == 0)
            for(int j = 0; j < 512; j++){
                if(j % 32 == 0)
                    printf("\n");
                printf("%.0f ", Out[j]);
            }*/

        #pragma unroll
        for(int j = 0; j < SMEM_LOADS * 2; j++){
            d_input[idx + i * SMEM_SIZE * 2 + j * BLOCKSIZE] = Out[j * BLOCKSIZE + threadIdx.x];
        }

    }

}

__device__ void move_and_refill(float *d_input, int i_pair, int j_L_Load, float *A, float *B, int offset_smem_A, int offset_smem_B, int &offset_A_sums, int &offset_B_sums, int length){

    // don't know if the pragmas unroll work here

    // move 
    #pragma unroll
    for(int i = threadIdx.x; i < SMEM_SIZE - offset_smem_A; i += BLOCKSIZE){
        A[i] = A[offset_smem_A + i];
    }
    #pragma unroll
    for(int i = threadIdx.x; i < SMEM_SIZE - offset_smem_B; i += BLOCKSIZE){
        B[i] = B[offset_smem_B + i];
    }
    
    int gmem_offset = blockIdx.x * STRIDE_BLOCK + i_pair * length;
    offset_A_sums = (offset_A_sums >= length / 2) ? (length / 2) : (offset_A_sums + offset_smem_A);
    offset_B_sums = (offset_B_sums >= length / 2) ? (length / 2) : (offset_B_sums + offset_smem_B);


    // refill
    #pragma unroll
    for(int i = threadIdx.x + SMEM_SIZE - offset_smem_A; i < SMEM_SIZE; i += BLOCKSIZE){
        A[i] = (offset_A_sums + i >= length / 2)? FLT_MAX : d_input[gmem_offset + offset_A_sums + i];
    }

    #pragma unroll
    for(int i = threadIdx.x + SMEM_SIZE - offset_smem_B; i < SMEM_SIZE; i += BLOCKSIZE){
        B[i] = (offset_B_sums + i >= length / 2)? FLT_MAX : d_input[gmem_offset + length / 2 + offset_B_sums + i];
    }


}


// we add refilling
// first iter: length 512, 1024 pairs, 4 pairs per block
// as we said, L A and L B gives L Out, so 1024/256 = 4 iters
// each thread processes STRIDE_THREAD/2 bcs now we have half Out per iter

/*
    why dynamic smem:
    case: 512 length pairs

    .____0______.______1_____.______2_____._____3______.

    merge 0 and 2, write whole 2 since all 2 elements are smaller than 0

    .____2______.______1_____.______2_____._____3______.

    0 is still on A(moving) compare with 3, write 3

    .____2______.______3_____.______2_____._____3______.
    
    we write 0 now since on A(moving) and B is all FLT_MAX

    .____2______.______3_____.______0_____._____3______.

    we never read 1 and we overwrote it

    OUT will hold the whole merged pair and then we copy it to gmem

*/
__global__ void merge_sort_medium(float *d_input, int length){

    extern __shared__ float SMEM[];

    float *A = SMEM;
    float *B = SMEM + SMEM_SIZE;
    float *Out = SMEM + 2 * SMEM_SIZE;

    /*__shared__ float A[SMEM_SIZE + 1];
    __shared__ float B[SMEM_SIZE + 1]; 
    __shared__ float Out[SMEM_SIZE + 1];*/


    int idx = blockIdx.x * STRIDE_BLOCK;

    int offset_smem_A, offset_smem_B; // offset due to non consumed items on smem
    int offset_A_sums, offset_B_sums;
    int pairs_per_block = STRIDE_BLOCK / length / 2;

    #pragma unroll
    for(int i = 0; i < pairs_per_block; i++){
        offset_smem_A = SMEM_SIZE, offset_smem_B = SMEM_SIZE; // initialisation
        offset_A_sums = -SMEM_SIZE, offset_B_sums = -SMEM_SIZE;
        
        #pragma unroll
        for(int j = 0; j < length * 2 / SMEM_SIZE; j++){
            // move unconsumed items and refill from global mem
            move_and_refill(d_input, i, j, A, B, offset_smem_A, offset_smem_B, offset_A_sums, offset_B_sums, length * 2);

            int diag = (threadIdx.x + 1) * SMEM_SIZE / BLOCKSIZE; // diagonal relative L A, L B loaded on smem
            int atop = diag;
            int btop = 0;
            int abot = btop;
            
            int ai, bi;
            int offset;

            int x2, y2;

            while(1){
                offset = (atop - abot)/2;
                ai = atop - offset;
                bi = btop + offset;

                if (ai >= length || bi == 0 || A[ai] > B[bi - 1]){
                    if(ai == 0 || bi >= length || A[ai - 1] <= B[bi]){
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

            // we found the points, now we merge
            // look at that slick warp shuffle
            int x1 = __shfl_sync(0xffffffff, x2, threadIdx.x - 1);
            int y1 = __shfl_sync(0xffffffff, y2, threadIdx.x - 1);
            x1 = threadIdx.x == 0 ? 0 : x1;
            y1 = threadIdx.x == 0 ? 0 : y1;

            offset_smem_A = __shfl_sync(0xffffffff, x2, 31);
            offset_smem_B = __shfl_sync(0xffffffff, y2, 31);

            seq_merge(Out + threadIdx.x * STRIDE_THREAD / 2 + j * SMEM_SIZE, A, x1, x2, B, y1, y2, STRIDE_THREAD / 2);

        }

        float4 *temp_d_input = (float4*) (d_input + idx + i * length * 2);
        float4 *temp_Out = (float4*) Out;

        #pragma unroll
        for(int k = 0; k < length * 2 / 4 / BLOCKSIZE; k++){
            temp_d_input[k * BLOCKSIZE + threadIdx.x] = temp_Out[k * BLOCKSIZE + threadIdx.x];
        }

    }

}

/*
__global__ void merge_sort_large(float *d_input, int length, int *diag_A, int *diag_B){

    __shared__ float A[SMEM_SIZE + 1]; 
    __shared__ float B[SMEM_SIZE + 1]; 
    __shared__ float Out[SMEM_SIZE + 1]; 

    int idx = blockIdx.x * STRIDE_BLOCK + threadIdx.x;

}*/

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
/*
__global__ void grid_partition_path(float *d_input, int length, int *diag_A, int *diag_B, int blocksperarray){

    // get where in d_input we are
    d_input += threadIdx.x * STRIDE_BLOCK;
    
    float *A = d_input;
    float *B = d_input + length;
    
    // blocksperarray blocks process the array
    // so each blocksperarray_i block starts at 0
    if(threadIdx.x & (blocksperarray - 1)){
        diag_A[threadIdx.x] = 0;
        diag_B[threadIdx.x] = 0;
    } else {
    
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

}
*/
/*
__global__ void odd_even_merge_sort(float *d_input){

    __shared__ float s_data[WARPSIZE];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    s_data[threadIdx.x] = d_input[idx];

    __syncthreads();

    int temp;

    #pragma unroll
    for(int p = 1 << (LOG2WARPSIZE - 1); p > 0; p /= 2) {
        
        int q = 1 << LOG2WARPSIZE;
        int r = 0;

        #pragma unroll
        for (int d = p ; d > 0 ; d = q - p) {

            if(threadIdx.x < WARPSIZE - d){
            
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

    dim3 dimBlock(WARPSIZE);
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


    /*int *diag_A, *diag_B;
    cudaMalloc((void **)&diag_A, GRIDSIZE2 * sizeof(float));
    checkCudaErrors(cudaGetLastError());
    cudaMalloc((void **)&diag_B, GRIDSIZE2 * sizeof(float));
    checkCudaErrors(cudaGetLastError());*/

    cudaEventRecord(start, 0);
    // 32 -> 256 (inclusive)
    for(int i = LOG2WARPSIZE; i <= LOG2SMEM; i++){
        merge_sort_small<<<GRIDSIZE2, BLOCKSIZE>>>(d_input, 1 << i);
        checkCudaErrors(cudaGetLastError());
    }
  
    // 512 -> 2048 (inclusive)
    for(int i = LOG2SMEM + 1; i < LOG2STRIDEBLOCK; i++){
        merge_sort_medium<<<GRIDSIZE2, BLOCKSIZE, ((1 << i) * 2 + 256 + 256) * sizeof(float)>>>(d_input, (1 << i));
        checkCudaErrors(cudaGetLastError());
    }

    // 2048 -> N/2
    /*int blocksperarray = 2;
    for(int i = LOG2STRIDEBLOCK; i <= LOG2ARR_SIZE - 1; i++){
        grid_partition_path<<<1, GRIDSIZE2>>>(d_input, 1 << i, diag_A, diag_B, blocksperarray);
        merge_sort_large<<<GRIDSIZE2, BLOCKSIZE>>>(d_input, 1 << i, diag_A, diag_B);
        blocksperarray <<= 1;
    }*/

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

    cuda_interface_sort(d_input);

    cudaMemcpy(h_input, d_input,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
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