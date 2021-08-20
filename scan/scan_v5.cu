/*

High on copium


    LightScan: Faster Scan Primitive on CUDA
    Yongchao Liu, Srinivas Aluru
  
  
 */

#define ARR_SIZE (1 << 25)
#define MAXNUM_SM_960M 5 // we will saturate the SM with 1024 threads consuming all ressources and grid size == num of sm that way we prevent deadlock
#define BLOCK_SIZE 1024
#define WARPS_NUM 32
#define WORK_PER_THREAD 40 // 10 float4 = 40 elements
#define BLOCK_STEP 164 // 2^25 / 5 / 1024 / 40 ~ 164
//round up to 164 we can either round up to the closes multiple of the arr size, or put guards, we're padding easier going to add 32kb of zeros

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_math.h>

__device__ int block_count = 0;

__global__ void scan(float *d_input, float* seeds, float *d_output){

    __shared__ float s_data[32];
    int lane = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    int idx = blockDim.x * blockIdx.x * WORK_PER_THREAD + warpid * 32 * WORK_PER_THREAD + threadIdx.x; // where we start = add offset du to blockidx + offset of warp + threadidx

    float items[WORK_PER_THREAD];
    d_input += idx;
    d_output += idx;
    seeds += blockIdx.x;

    float shift_variable = 0;
    float offset = 0;

    #pragma unroll
    for(int i = 0; i < BLOCK_STEP; i++){

        // load elements
        #pragma unroll
        for(int j = 0; j < WORK_PER_THREAD; j++){
            items[j] = d_input[32 * j];
        }

        // warp level scan
        #pragma unroll
        for(int j = 0; j < WORK_PER_THREAD; j++){
            shift_variable = __shfl_sync(0xffffffff, items[j], threadIdx.x - 1);
            if (lane >= 1) items[j] += shift_variable;

            shift_variable = __shfl_sync(0xffffffff, items[j], threadIdx.x - 2);
            if (lane >= 2)  items[j] += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, items[j], threadIdx.x - 4);
            if (lane >= 4)  items[j] += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, items[j], threadIdx.x - 8);
            if (lane >= 8)  items[j] += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, items[j], threadIdx.x - 16);
            if (lane >= 16) items[j] += shift_variable;
        }

        // load the 31th item to each item - this is the offset between the items loaded by same warp
        #pragma unroll
        for(int j = 1; j < WORK_PER_THREAD; j++){
            items[j] += __shfl_sync(0xffffffff, items[j - 1], 31);
        } 

        // we load the last items value of the 31th thread
        if(lane == 31){
            s_data[warpid] = items[WORK_PER_THREAD - 1];
        }

        __syncthreads();
        // load on registers and do intra warp scan on warp 1
        if(warpid == 0){
            float s_data_item = s_data[threadIdx.x];
            shift_variable = __shfl_sync(0xffffffff, s_data_item, threadIdx.x - 1);
            if (lane >= 1) s_data_item += shift_variable;

            shift_variable = __shfl_sync(0xffffffff, s_data_item, threadIdx.x - 2);
            if (lane >= 2)  s_data_item += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, s_data_item, threadIdx.x - 4);
            if (lane >= 4)  s_data_item += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, s_data_item, threadIdx.x - 8);
            if (lane >= 8)  s_data_item += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, s_data_item, threadIdx.x - 16);
            if (lane >= 16) s_data_item += shift_variable;

            s_data[threadIdx.x] = s_data_item;
        }

        __syncthreads();

        // add the inter warp offsets
        if(warpid > 0){
            #pragma unroll
            for(int j = 0; j < WORK_PER_THREAD; j++){
                items[j] += s_data[warpid - 1];
            }
        }

        // wait for the offset to be ready
        if(!(blockIdx.x == 0 && i == 0)){
            while(atomicAdd(&block_count, 0) < (MAXNUM_SM_960M * i + blockIdx.x)){}
            offset = *(seeds - 1);
        }

        // add block offset and store
        #pragma unroll
        for(int j = 0; j < WORK_PER_THREAD; j++){
            items[j] += offset;
        }

        #pragma unroll
        for(int j = 0; j < WORK_PER_THREAD; j++){
            d_output[j * 32] = items[j];
        }

        if(threadIdx.x == 1023){
            seeds[0] = items[WORK_PER_THREAD - 1];
        }
        // makes sure the data is there when reading with atomic
        __threadfence();

        if(threadIdx.x == 0){
            atomicAdd(&block_count, 1);
        }

        d_input += MAXNUM_SM_960M * blockDim.x * WORK_PER_THREAD;
        d_output += MAXNUM_SM_960M * blockDim.x * WORK_PER_THREAD;
        seeds += MAXNUM_SM_960M;
    }

}

// main + interface
void cuda_interface_scan(float* d_input, float* d_output){

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(MAXNUM_SM_960M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    float *offsets;
    cudaMalloc((void **)&offsets, MAXNUM_SM_960M * BLOCK_STEP * sizeof(float));

    cudaEventRecord(start, 0);
    scan<<<dimGrid, dimBlock>>>(d_input, offsets, d_output);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "scan: %.8f ms\n", elapsed_time);

    cudaFree(offsets);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}   

void fill_array(float *h_input, int padded_length){

    for(int i = 0; i <  ARR_SIZE; i++){
        h_input[i] = (float) rand() / RAND_MAX;
    }
    for(int i = ARR_SIZE; i < padded_length; i++){
        h_input[i] = 0;
    }
}

void check_result(float *h_input, float *h_output){

    float *temp = (float*) malloc(ARR_SIZE * sizeof(float));

    temp[0] = h_input[0];
    for(int i = 1; i < ARR_SIZE; i++){
        temp[i] = h_input[i] + temp[i - 1];
    }

    std::cout<<"first 1050 elements:\n";
    std::cout<<"element"<<"\tcpu"<<"\tgpu\n";

    for(int i = 0; i < 1050; i++){
        std::cout<<i<<"\t"<<h_input[i] << "\t" << temp[i] << "\t" << h_output[i] <<"\n";
    }

    float diff = 0;
    for(int i = 0; i < ARR_SIZE; i++){
        diff += h_output[i] - temp[i];
    }   

    std::cout<<"diff"<< diff << "\n";

    free(temp);
}

int main(void){

    srand(0);

    float *h_input, *h_output;
    float *d_input, *d_output;

    int padded_length = MAXNUM_SM_960M * BLOCK_STEP * BLOCK_SIZE * WORK_PER_THREAD;

    h_input = (float*) malloc(padded_length * sizeof(float));
    h_output = (float*) malloc(ARR_SIZE * sizeof(float));
     
    fill_array(h_input, padded_length);

    cudaMalloc((void **)&d_input, padded_length * sizeof(float));
    cudaMalloc((void **)&d_output, padded_length * sizeof(float));

    cudaMemcpy(d_input, h_input, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cuda_interface_scan(d_input, d_output);

    cudaMemcpy(h_output, d_output,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    check_result(h_input, h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}