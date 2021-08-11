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
#define BLOCKSIZE 128
#define LOG2_BLOCKSIZE 7
#define SCAN_SMEM_WIDTH (BLOCKSIZE/32)
#define LOG2_SCAN_SMEM_WIDTH (LOG2_BLOCKSIZE - 5)
#define WORK_PER_THREAD 8 // 8 float4 each thread on scanning and reducing
#define LOG2_WORK_PER_THREAD 5 // 32 elements (8 float4)
#define MIDDLE_SCAN_STEP 64 // 2^(25 - 5 - 7 - 7) // -5 (8 float4 loads) - 7 (blocksize) - 7 (each thread of middle scan block)

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_math.h>

// SIMT Kogge-Stone scan kernel
/*__device__ __inline__ void scan_warp(volatile float* input, int indx = threadIdx.x){
    int lane = indx & 31;
    
    if (lane >= 1)  input[indx] = input[indx - 1] + input[indx];
    if (lane >= 2)  input[indx] = input[indx - 2] + input[indx];
    if (lane >= 4)  input[indx] = input[indx - 4] + input[indx];
    if (lane >= 8)  input[indx] = input[indx - 8] + input[indx];
    if (lane >= 16) input[indx] = input[indx - 16] + input[indx];
}*/

// SIMT Brent-Kung scan kernel - same as the merrill_srts reduction kernel but since it's the same as the warp size -> no need for __syncthreads()
// BUT BUT!!!!! since this is SIMT -> there is actually 0 gain from reducing the number of operations , so the scan-warp will be used.
/*__device__ __inline__ void reduce_warp(float* input, int indx = threadIdx.x){
    scan_warp(input, indx);
}*/

// the only change is how smem is handled after the serial scan
__device__ __inline__ void scan_warp_merrill_srts(volatile float (*s_data)[SCAN_SMEM_WIDTH + 1], int indx = threadIdx.x){
    int lane = indx & 31;

    if (lane >= 1)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 1][SCAN_SMEM_WIDTH - 1] + s_data[indx][SCAN_SMEM_WIDTH - 1];
    if (lane >= 2)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 2][SCAN_SMEM_WIDTH - 1] + s_data[indx][SCAN_SMEM_WIDTH - 1];
    if (lane >= 4)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 4][SCAN_SMEM_WIDTH - 1] + s_data[indx][SCAN_SMEM_WIDTH - 1];
    if (lane >= 8)  s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 8][SCAN_SMEM_WIDTH - 1] + s_data[indx][SCAN_SMEM_WIDTH - 1];
    if (lane >= 16) s_data[indx][SCAN_SMEM_WIDTH] = s_data[indx - 16][SCAN_SMEM_WIDTH - 1] + s_data[indx][SCAN_SMEM_WIDTH - 1];

}

// merrill_srts reduce kernel
__global__ void reduce(float4 *d_input, float *d_output){

    __shared__ float s_data[32][SCAN_SMEM_WIDTH + 1]; // 1 for the result of the warp scan

    int idx = blockDim.x * blockIdx.x * WORK_PER_THREAD + threadIdx.x;

    d_input += idx;
    d_output += blockIdx.x;

    int row = threadIdx.x >> LOG2_SCAN_SMEM_WIDTH;
    int col = threadIdx.x & (SCAN_SMEM_WIDTH - 1);

    float4 item;
    float sum = 0;
    float total_sum = 0;

    #pragma unroll
    for(int i = 0; i < WORK_PER_THREAD; i++){
        item = d_input[i * BLOCKSIZE];
        sum += item.x + item.y + item.z + item.w;

        s_data[row][col] = sum;
        __syncthreads();

        if((threadIdx.x >> 5) == 0){
            #pragma unroll
            for(int j = 1; j < SCAN_SMEM_WIDTH; j++){
                s_data[threadIdx.x][j] += s_data[threadIdx.x][j - 1];
            }
            scan_warp_merrill_srts(s_data);
        }

        // after simt scan we work on first warp, so no need for sync
        if(threadIdx.x == 0){
            total_sum += s_data[31][SCAN_SMEM_WIDTH]; // total sum is at this cell
        }

    }

    if(threadIdx.x == 0){
        d_output[0] = total_sum;
    }

    
}

// merrill_srts scan kernel
__global__ void scan(float4 *d_input, float *seeds, float4 *d_output){

    __shared__ float s_data[32][SCAN_SMEM_WIDTH + 1]; // 1 for no bank conflict and another one for the result of the warp scan

    int idx = blockDim.x * blockIdx.x * WORK_PER_THREAD+ threadIdx.x;

    d_input += idx;
    d_output += idx;

    int row = threadIdx.x >> LOG2_SCAN_SMEM_WIDTH;
    int col = threadIdx.x & (SCAN_SMEM_WIDTH - 1);

    float4 item;

    float seed = 0;

    for(int i = 0 ; i < WORK_PER_THREAD; i++){
        item = d_input[i * BLOCKSIZE];

        if(threadIdx.x == 0 && blockIdx.x > 0 && i==0){
            item.x += seeds[blockIdx.x - 1];
        }
        item.x += seed; // only thread 0 adds seed here
        item.y += item.x;
        item.z += item.y;
        item.w += item.z;
        s_data[row][col] = item.w;

        __syncthreads();

        // serial reduce
        if((threadIdx.x >> 5) == 0){
            #pragma unroll
            for(int j = 1; j < SCAN_SMEM_WIDTH; j++){
                s_data[threadIdx.x][j] += s_data[threadIdx.x][j - 1];
            }
            scan_warp_merrill_srts(s_data);
        }

        __syncthreads();

        // add the SIMT scan seeds
        // sum last column of simt scan
        if(threadIdx.x >= SCAN_SMEM_WIDTH){
            item += s_data[row - 1][SCAN_SMEM_WIDTH];
        }
        // sum element before in row, serial scan
        if(threadIdx.x > 0){
            item += s_data[row][col - 1];
        }

        if(threadIdx.x == 0){
            seed = s_data[31][SCAN_SMEM_WIDTH];
        }

        d_output[0] = item;

        __syncthreads();
    }
}

// two level reduce then scan - middle scan kernel
__global__ void middle_scan(float *seeds){

    __shared__ float s_data[32][SCAN_SMEM_WIDTH + 1]; // 1 for no bank conflict and another one for the result of the warp scan

    int row = threadIdx.x >> LOG2_SCAN_SMEM_WIDTH;
    int col = threadIdx.x & (SCAN_SMEM_WIDTH - 1);
    
    float seed = 0;
    seeds += threadIdx.x;

    // cyclically scan the reduced sums
    #pragma unroll
    for(int i = 0; i < MIDDLE_SCAN_STEP; i++){
        s_data[row][col] = seeds[i * BLOCKSIZE] + seed; // only thread 0 adds seed here

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

    int temp = (ARR_SIZE >> LOG2_WORK_PER_THREAD)/BLOCKSIZE; // each thread processes 8 float4
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


void check(float4 *h_input, float4 *h_output){
    float *temp1 = (float*) h_input;
    float *temp2 = (float*) h_output;
    float *temp3 = (float*) malloc(ARR_SIZE * sizeof(float));

    temp3[0] = temp1[0];
    for(int i = 1; i < ARR_SIZE; i++){
        temp3[i] = temp1[i] + temp3[i - 1];
    }

    std::cout<<"first 50 elements:\n";
    std::cout<<"element"<<"\tcpu"<<"\tgpu\n";

    for(int i = 0; i < 1050; i++){
        std::cout<<i<<"\t"<<temp1[i] << "\t" << temp3[i] << "\t" << temp2[i] <<"\n";
    }

    free(temp3);
}



int main(void){

    srand(0);

    float4 *h_input, *h_output;
    float4 *d_input, *d_output;

    h_input = (float4*) malloc(ARR_SIZE * sizeof(float));
    h_output = (float4*) malloc(ARR_SIZE * sizeof(float));
     
    fill_array(h_input);

    cudaMalloc((void **)&d_input, ARR_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output, ARR_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cuda_interface_scan(d_input, d_output);

    cudaMemcpy(h_output, d_output,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    check(h_input, h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}