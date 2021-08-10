/*  Let's do better

    Efficient Parallel Scan Algorithms for GPUs
        Shubhabrata Sengupta Davis Mark Harris Michael Garland

    Parallel Scan for Stream Architectures1
        Duane Merrill Andrew Grimshaw

    Inclusive scan

    We need to beat the 12ms mark on 2^25 elements (~33m elements)

*/
 
// for now everything is power of 2, normally this won't be the case -> padding + if elses

#define ARR_SIZE (1 << 25)
#define BLOCKSIZE 512
#define LOG2_BLOCKSIZE 9
#define REDUCTION_STEPS 2 // each thread thread loads 2 float4, 128B
#define SCAN_STEPS REDUCTION_STEPS // each block thread loads 2 float4, 128B
#define SCAN_SMEM_WIDTH (BLOCKSIZE/32)
#define LOG2_SCAN_SMEM_WIDTH 4
#define MIDDLE_SCAN_STEP 16 // 2^(25 - 3 - 9 - 9) // -3 (2 float4 loads) - 9 (blocksize) - 9 (each thread of middle scan block)


#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>

// SIMT Kogge-Stone scan kernel
__device__ __inline__ void scan_warp(volatile float* input, int indx = threadIdx.x){
    int lane = indx & 31;
    
    if (lane >= 1)  input[indx] = input[indx - 1] + input[indx];
    if (lane >= 2)  input[indx] = input[indx - 2] + input[indx];
    if (lane >= 4)  input[indx] = input[indx - 4] + input[indx];
    if (lane >= 8)  input[indx] = input[indx - 8] + input[indx];
    if (lane >= 16) input[indx] = input[indx - 16] + input[indx];
}

// SIMT Brent-Kung scan kernel - same as the merrill_srts reduction kernel but since it's the same as the warp size -> no need for __syncthreads()
// BUT BUT!!!!! since this is SIMT -> there is actually 0 gain from reducing the number of operations , so the scan-warp will be used.
__device__ __inline__ void reduce_warp(float* input, int indx = threadIdx.x){
    scan_warp(input, indx);
}

// merrill_srts reduce kernel
__global__ void reduce(float4 *d_input, float *d_output){

    __shared__ float s_data[BLOCKSIZE * 2];//1 cell per thread + another blockdim for easier indx management - will have 2 way bank conflicts though
    
    int idx = blockDim.x * blockIdx.x * REDUCTION_STEPS + threadIdx.x;
    
    d_input += idx;
    d_output += blockIdx.x;

    float4 item;
    float sum = 0;

    #pragma unroll
    for(int i = 0; i < REDUCTION_STEPS; i++){
        item = d_input[i * BLOCKSIZE];
        sum += item.w + item.x + item.y + item.z;
    }

    s_data[threadIdx.x] = sum;

    __syncthreads();

    // we reduce and put the result on the second half of shared memory
    // why no serial reduce like the scan? 
    float *a = s_data;

    #pragma unroll
    for(int d = LOG2_BLOCKSIZE; d > 5; d--){ // 9 -> 5
        
        if( threadIdx.x < (1 << (d - 1)) ){
            a[(1 << d) + threadIdx.x] = a[2 * threadIdx.x] + a[2 * threadIdx.x + 1];
        }

        a = &a[(1 << d)];
        __syncthreads();

    }

    if((threadIdx.x >> 5) == 0){ // warp 0
        reduce_warp(a); // sum will be at idx 31
    }

    // output the sum
    if(threadIdx.x == 31){
        d_output[0] = a[31];
    }

}

// the only change is how smem is handled after the serial scan
__device__ __inline__ void scan_warp_merrill_srts(volatile float (*s_data)[SCAN_SMEM_WIDTH + 1 + 1], int indx = threadIdx.x){
    int lane = indx & 31;

    if (lane >= 1)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 1][SCAN_SMEM_WIDTH] + s_data[indx][SCAN_SMEM_WIDTH];
    if (lane >= 2)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 2][SCAN_SMEM_WIDTH] + s_data[indx][SCAN_SMEM_WIDTH];
    if (lane >= 4)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 4][SCAN_SMEM_WIDTH] + s_data[indx][SCAN_SMEM_WIDTH];
    if (lane >= 8)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 8][SCAN_SMEM_WIDTH] + s_data[indx][SCAN_SMEM_WIDTH];
    if (lane >= 16) s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 16][SCAN_SMEM_WIDTH] + s_data[indx][SCAN_SMEM_WIDTH];

}

// merrill_srts scan kernel
__global__ void scan(float4 *d_input, float *seeds, float4 *d_output){

    __shared__ float s_data[32 * SCAN_STEPS][SCAN_SMEM_WIDTH + 1 + 1]; // 1 for no bank conflict and another one for the result of the warp scan

    int idx = blockDim.x * blockIdx.x * SCAN_STEPS + threadIdx.x;

    d_input += idx;
    d_output += idx;

    int row = threadIdx.x >> LOG2_SCAN_SMEM_WIDTH;
    int col = threadIdx.x & (SCAN_SMEM_WIDTH - 1);

    float4 item[SCAN_STEPS];

    #pragma unroll
    for(int i = 0; i < SCAN_STEPS; i++){
        item[i] = d_input[i * BLOCKSIZE];
        if(threadIdx.x == 0 && i==0 && blockIdx.x > 0){
            item[i].x += seeds[blockIdx.x - 1];
        }
        item[i].y += item[i].x;
        item[i].z += item[i].y;
        item[i].w += item[i].z;
        s_data[32 * i + row][col] = item[i].w;
    }

    __syncthreads();

    // serial reduce
    // each warp going to do 32 rows of smem
    if((threadIdx.x >> 5) < SCAN_STEPS){
        #pragma unroll
        for(int i = 1; i < SCAN_SMEM_WIDTH; i++){
            s_data[threadIdx.x][i] += s_data[threadIdx.x][i - 1];
        }

        scan_warp_merrill_srts(s_data);
    }

    __syncthreads();

    // add the SIMT scan seeds

    #pragma unroll
    for(int i = 0; i < SCAN_STEPS; i++){
        // sum last column of simt scan
        if(threadIdx.x >= SCAN_SMEM_WIDTH){
            item[i].x += s_data[32 * i + row - 1][SCAN_SMEM_WIDTH];
            item[i].y += s_data[32 * i + row - 1][SCAN_SMEM_WIDTH];
            item[i].z += s_data[32 * i + row - 1][SCAN_SMEM_WIDTH];
            item[i].w += s_data[32 * i + row - 1][SCAN_SMEM_WIDTH];
        }
        // sum element before in row, serial scan
        if(threadIdx.x > 0){
            item[i].x += s_data[32 * i + row][col - 1];
            item[i].y += s_data[32 * i + row][col - 1];
            item[i].z += s_data[32 * i + row][col - 1];
            item[i].w += s_data[32 * i + row][col - 1];
        }
        // sum last element of previous simt scan
        if(i > 0){
            item[i].x += s_data[32 * (i - 1) + 31][SCAN_SMEM_WIDTH];
            item[i].y += s_data[32 * (i - 1) + 31][SCAN_SMEM_WIDTH];
            item[i].z += s_data[32 * (i - 1) + 31][SCAN_SMEM_WIDTH];
            item[i].w += s_data[32 * (i - 1) + 31][SCAN_SMEM_WIDTH];
        }
        d_output[i * BLOCKSIZE] = item[i];
    }

}

// two level reduce then scan - middle scan kernel
__global__ void middle_scan(float *seeds){

    __shared__ float s_data[32][SCAN_SMEM_WIDTH + 1 + 1]; // 1 for no bank conflict and another one for the result of the warp scan

    int row = threadIdx.x >> LOG2_SCAN_SMEM_WIDTH;
    int col = threadIdx.x & (SCAN_SMEM_WIDTH - 1);
    
    float seed = 0;
    seeds += threadIdx.x;
    
    // cyclically scan the reduced sums
    #pragma unroll
    for(int i = 0; i < MIDDLE_SCAN_STEP; i++){
        s_data[row][col] = seeds[i * BLOCKSIZE];

        if(threadIdx.x == 0){
            s_data[0][0] += seed;
        }

        __syncthreads();


        if((threadIdx.x >> 5) == 0){
            #pragma unroll
            for(int j = 1; j < SCAN_SMEM_WIDTH; j++){
                s_data[threadIdx.x][j] += s_data[threadIdx.x][j - 1];
            }

            __syncthreads();

            scan_warp_merrill_srts(s_data);
        }

        if(threadIdx.x == 0){
            seed = s_data[31][SCAN_SMEM_WIDTH];
        }

        __syncthreads();

        if(threadIdx.x >= SCAN_SMEM_WIDTH){
            seeds[i * BLOCKSIZE] = s_data[row][col] + s_data[row - 1][SCAN_SMEM_WIDTH];
        } else {
            seeds[i * BLOCKSIZE] = s_data[0][threadIdx.x];
        }
    }
}

// main + interface

void cuda_interface_scan(float4* d_input, float4* d_output){

    int temp = (ARR_SIZE >> 3)/BLOCKSIZE; // each thread processes 2 float4
    dim3 dimBlock(BLOCKSIZE);
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
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "reduce: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;

    /*std::cout<<"------------------\n";
    float *h_scan = (float*)malloc(temp * sizeof(float));
    cudaMemcpy(h_scan, d_scan,  temp * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=510; i < 515; i++)
        std::cout<<h_scan[i]<<"\n";*/

    cudaEventRecord(start, 0);
    middle_scan<<<1, dimBlock>>>(d_scan);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "middle scan: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;

    /*std::cout<<"--------------------------middle scan\n";
    cudaMemcpy(h_scan, d_scan,  temp * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=510; i < 515; i++)
        std::cout<<h_scan[i]<<"\n";*/

    cudaEventRecord(start, 0);
    scan<<<dimGrid, dimBlock>>>(d_input, d_scan, d_output);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "final scan: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;

    printf("total time GPU %.8fms\n", total_time);

    cudaFree(d_scan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}   


void fill_array(float4 *h_input){

    float *temp = (float*) h_input;
    for(int i = 0; i <  ARR_SIZE; i++){
        temp[i] = (float) rand() / RAND_MAX;
    }
}

int main(void){

    srand(0);

    float4 *h_input, *h_output;
    float4 *d_input, *d_output;

    h_input = (float4*) malloc(ARR_SIZE * sizeof(float));
    h_output = (float4*) malloc(ARR_SIZE * sizeof(float));
     
    fill_array(h_input);

    for(int i = 0; i < 5; i++){
        std::cout<<h_input[i].x<<" "<<h_input[i].y<<" "<<h_input[i].z<<" "<<h_input[i].w<<"\n";
    }

    std::cout<<"----------------------\n";

    cudaMalloc((void **)&d_input, ARR_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output, ARR_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cuda_interface_scan(d_input, d_output);

    cudaMemcpy(h_output, d_output,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout<<"--------GPU----------\n";

    for(int i = 0; i < 5; i++){
        std::cout<<h_output[i].x<<" "<<h_output[i].y<<" "<<h_output[i].z<<" "<<h_output[i].w<<"\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}