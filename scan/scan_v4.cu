// we continue the optimisation of scan_v4, playing around grid sizes/threads/work per thread, etc.

/*

    max block per sm 32
    max num of warps per sm 64
    64k 32bit registers per sm
    64k shared mem per sm

*/


#define ARR_SIZE (1 << 25)
#define GRIDSIZE 256
#define BLOCKSIZE 128
#define LOG2_BLOCKSIZE 7
#define WARPS_NUM (BLOCKSIZE/32)
#define WORK_PER_THREAD 256 // 2^(25 - 8 - 7 - 2) = 256 float4 = 1024 elements
#define LOG2_WORK_PER_THREAD 10 // 1024 elements (256 float4)

#define MIDDLE_SCAN_WORK_PER_THREAD (GRIDSIZE/BLOCKSIZE) // 2

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_math.h>

__device__ __inline__ void warp_smem_scan(volatile float *s_data, int indx = threadIdx.x){
    int lane = indx & 31;
    if (lane >= 1)  s_data[indx] = s_data[indx - 1] + s_data[indx];
    if (lane >= 2)  s_data[indx] = s_data[indx - 2] + s_data[indx];
    if (lane >= 4)  s_data[indx] = s_data[indx - 4] + s_data[indx];
    if (lane >= 8)  s_data[indx] = s_data[indx - 8] + s_data[indx];
    if (lane >= 16) s_data[indx] = s_data[indx - 16] + s_data[indx];
}


__global__ void reduce(float4 *d_input, float *d_output){

    __shared__ float s_data[32];
    int idx = blockDim.x * blockIdx.x * WORK_PER_THREAD + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;

    d_input += idx;
    d_output += blockIdx.x;

    float sum[WORK_PER_THREAD];

    #pragma unroll
    for(int i = 0; i < WORK_PER_THREAD; i++){
        sum[i] = d_input[i * BLOCKSIZE].x + d_input[i * BLOCKSIZE].y + d_input[i * BLOCKSIZE].z + d_input[i * BLOCKSIZE].w; 
    }

    #pragma unroll
    for(int i = 1; i < WORK_PER_THREAD; i++){
        sum[0] += sum[i];
    }

    // at this point we have 1 sum per thread, so 128, we reduce them to 4

    sum[0] += __shfl_sync(0xffffffff, sum[0], threadIdx.x - 1);
    sum[0] += __shfl_sync(0xffffffff, sum[0], threadIdx.x - 2);
    sum[0] += __shfl_sync(0xffffffff, sum[0], threadIdx.x - 4);
    sum[0] += __shfl_sync(0xffffffff, sum[0], threadIdx.x - 8);
    sum[0] += __shfl_sync(0xffffffff, sum[0], threadIdx.x - 16);

    if(lane == 31){
        s_data[warpid] = sum[0];
    }

    __syncthreads();

    if(threadIdx.x == 31){

        #pragma unroll
        for(int i = 1; i < WARPS_NUM; i++){
            s_data[i] += s_data[i - 1];
        }

        d_output[0] = s_data[WARPS_NUM - 1];

    }

    /*
    //the 31th threads will have the sum - no need for smem

    if(lane == 31){
        #pragma unroll
        for(int i = 0; i < WORK_PER_THREAD; i++){
            s_data[warpid + i * WARPS_NUM] = sum[i];
        }
    }

    __syncthreads();

    if(warpid == 0){
        warp_smem_scan(s_data);
    }

    if(threadIdx.x == 0){
        d_output[0] = s_data[31];
    }*/
}
/*
__global__ void scan(float4 *d_input, float *seeds, float4 *d_output){

    __shared__ float s_data[32];
    int idx = blockDim.x * blockIdx.x * WORK_PER_THREAD + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;

    d_input += idx;
    d_output += idx;

    float4 item[WORK_PER_THREAD];
    float shift_variable;

    item[0] = d_input[0];
    item[0].x += (threadIdx.x == 0 && blockIdx.x > 0) ? seeds[blockIdx.x - 1] : 0;
    item[0].y += item[0].x;
    item[0].z += item[0].y;
    item[0].w += item[0].z;
    
    #pragma unroll
    for(int i = 1; i < WORK_PER_THREAD; i++){
        item[i] = d_input[i * BLOCKSIZE];
        item[i].y += item[i].x;
        item[i].z += item[i].y;
        item[i].w += item[i].z;
    }
    
    #pragma unroll
    for(int i = 0; i < WORK_PER_THREAD; i++){
        shift_variable = __shfl_sync(0xffffffff, item[i].w, threadIdx.x - 1);
        if (lane >= 1) item[i] += shift_variable;

        shift_variable = __shfl_sync(0xffffffff, item[i].w, threadIdx.x - 2);
        if (lane >= 2)  item[i] += shift_variable;
        
        shift_variable = __shfl_sync(0xffffffff, item[i].w, threadIdx.x - 4);
        if (lane >= 4)  item[i] += shift_variable;
        
        shift_variable = __shfl_sync(0xffffffff, item[i].w, threadIdx.x - 8);
        if (lane >= 8)  item[i] += shift_variable;
        
        shift_variable = __shfl_sync(0xffffffff, item[i].w, threadIdx.x - 16);
        if (lane >= 16) item[i] += shift_variable;
    }

    if(lane == 31){
        #pragma unroll
        for(int i = 0; i < WORK_PER_THREAD; i++){
            s_data[warpid + i * WARPS_NUM] = item[i].w;
        }
    }

    __syncthreads();

    if(warpid == 0){
        warp_smem_scan(s_data);
    }

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < WORK_PER_THREAD; i++){
        if (!(warpid == 0 && i == 0)){
                item[i] += s_data[warpid - 1 + i * WARPS_NUM];
        }
    }

    #pragma unroll
    for(int i = 0; i < WORK_PER_THREAD; i++){
        d_output[i * BLOCKSIZE] = item[i];
    }
    
}

__global__ void middle_scan(float *seeds){

    __shared__ float s_data[32];
    int lane = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    
    float seed = 0;
    seeds += threadIdx.x;

    float item[MIDDLE_SCAN_WORK_PER_WARP]; // 4 warps, need 8 to fill 32

    float shift_variable = 0;

    // cyclically scan the reduced sums
    #pragma unroll
    for(int k = 0; k < MIDDLE_SCAN_STEP_PER_WARP; k++){

        item[0] = seeds[0] + seed; //only thread 0 adds seed here
        
        #pragma unroll
        for(int i = 1; i < MIDDLE_SCAN_WORK_PER_WARP; i++){
            item[i] = seeds[i * BLOCKSIZE];
        }

        #pragma unroll
        for(int i = 0; i < MIDDLE_SCAN_WORK_PER_WARP; i++){
            shift_variable = __shfl_sync(0xffffffff, item[i], threadIdx.x - 1);
            if (lane >= 1) item[i] += shift_variable;

            shift_variable = __shfl_sync(0xffffffff, item[i], threadIdx.x - 2);
            if (lane >= 2)  item[i] += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, item[i], threadIdx.x - 4);
            if (lane >= 4)  item[i] += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, item[i], threadIdx.x - 8);
            if (lane >= 8)  item[i] += shift_variable;
            
            shift_variable = __shfl_sync(0xffffffff, item[i], threadIdx.x - 16);
            if (lane >= 16) item[i] += shift_variable;

        }

        if(lane == 31){
            #pragma unroll
            for(int i = 0; i < MIDDLE_SCAN_WORK_PER_WARP; i++){
                s_data[warpid + i * WARPS_NUM] = item[i];
            }
        }

        __syncthreads();

        if(warpid == 0){
            warp_smem_scan(s_data);
        }

        if(threadIdx.x == 0){
            seed = s_data[31];
        }

        __syncthreads();

        #pragma unroll
        for(int i = 0; i < MIDDLE_SCAN_WORK_PER_WARP; i++){
            if (!(warpid == 0 && i == 0)){
                item[i] += s_data[warpid - 1 + i * WARPS_NUM];
            }
        }

        #pragma unroll
        for(int i = 0; i < MIDDLE_SCAN_WORK_PER_WARP; i++){
            seeds[i * BLOCKSIZE] = item[i];
        }

        seeds += MIDDLE_SCAN_WORK_PER_WARP * BLOCKSIZE;
    }
}
*/
// main + interface
void cuda_interface_scan(float4* d_input, float4* d_output, float4 *h_input){

    //int temp = ARR_SIZE >> (LOG2_WORK_PER_THREAD + LOG2_BLOCKSIZE); // each thread processes 8 float4
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(GRIDSIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0;
    float elapsed_time;

    float *d_scan;
    cudaMalloc((void **)&d_scan, GRIDSIZE * sizeof(float));

    cudaEventRecord(start, 0);
    reduce<<<dimGrid, dimBlock>>>(d_input, d_scan);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "reduce: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;


    float *cpu_out = (float*)malloc(GRIDSIZE * sizeof(float));
    /*float *temp = (float*) h_input;
    for(int i = 0; i < GRIDSIZE; i++){
        cpu_out[i] = 0;
        for(int j = 0; j < (1 << 17); i++){
            cpu_out[i] += temp[j]; 
        }
        temp += (1 << 17);
    }*/

    float *res = (float*)malloc(GRIDSIZE * sizeof(float));
    cudaMemcpy(res, d_scan,  GRIDSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < GRIDSIZE; i++){
        std::cout<<res[i]<<" "<<cpu_out[i]<< "\n";
    }
    /*cudaEventRecord(start, 0);
    middle_scan<<<1, dimBlock>>>(d_scan);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "middle scan: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;


    cudaEventRecord(start, 0);
    scan<<<dimGrid, dimBlock>>>(d_input, d_scan, d_output);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "final scan: %.8f ms\n", elapsed_time);
    total_time += elapsed_time;

    printf("total time GPU %.8fms\n", total_time);*/

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

    std::cout<<"first 1050 elements:\n";
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

    cuda_interface_scan(d_input, d_output, h_input);

    /*cudaMemcpy(h_output, d_output,  ARR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    check(h_input, h_output);*/

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}