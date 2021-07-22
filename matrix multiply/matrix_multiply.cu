#include <iostream>
#include <stdio.h>
#include <cuda_profiler_api.h>

// C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include
#include <helper_timer.h>

#define BLOCK_DIM 16


// C = alpha*A*B + beta*C -- A= MxK -- B= KxN -- C= MxN
__global__ void matrix_multiply(float* A, float* B, float*C, int M, int N, int K, float alpha, float beta){
    
    int bid_x = blockIdx.x * blockDim.x;
    int bid_y = blockIdx.y * blockDim.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float element_c = 0.f;

    // seen by whole block 
    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    for(int k=0; k < K; k+= BLOCK_DIM){
        // we put entire tile A and tile B on shared mem
        s_tile_A[tid_y][tid_x] = A[(bid_y + tid_y)*K + tid_x + k]; // tid_x element of each block of that row of matrix A
        s_tile_B[tid_y][tid_x] = B[N*k + bid_x + tid_x]; // tid_x elemnt of each block of that col of matrix B
        
        //sync
        __syncthreads();

        // multiply 
        for (int e = 0; e < BLOCK_DIM; e++)
            element_c += s_tile_A[tid_y][e] * s_tile_B[e][tid_x];
	    
        // sync
	    __syncthreads();

        // we update tiles and multiply again and accumulate.
    }
    
    C[(bid_y + tid_y) * N + (bid_x + tid_x)] = alpha * element_c + beta * C[(bid_y + tid_y) * N + (bid_x + tid_x)];

}

void random_init(float *data, int length)
{
    for (int i = 0; i < length; i++) {
        data[i] = (rand() & 0xFFFF) / (float)RAND_MAX;
    }
}

int main(int c, char *argv[])
{
    float *A, *B, *C_host, *C_gpu;
    float *d_A, *d_B, *d_C;
    int M, N, K;
    float alpha = 2.f;
    float beta = 1.f;
    int n_iter = 1;
    N = M = K = 2048;

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocation of linear memory space
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C_host = (float *)malloc(M * N * sizeof(float));
    C_gpu = (float *)malloc(M * N * sizeof(float));

    // allocation of gpu linear memory space
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // initialize randomized values for memory space
    random_init(A, M * K);
    random_init(B, K * N);

    // profiler will focus from this point
    sdkStartTimer(&timer);

    // copy initial value for gpu memory
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, A, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // do operation
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    cudaProfilerStart();

    for (int i = 0; i < n_iter; i++) {
        matrix_multiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    // profiler will stop its focus
    cudaProfilerStop();
    
    // measuring the performance
    cudaDeviceSynchronize();
    sdkStopTimer(&timer); // this profiler should be behined of device synchronization

    
    cudaMemcpy(C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);


    // terminates allocated gpu memory space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // terminates allocated memory space
    free(A);
    free(B);
    free(C_host);
    free(C_gpu);

    return 0;
}