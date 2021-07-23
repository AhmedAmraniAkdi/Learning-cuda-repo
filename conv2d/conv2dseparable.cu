#include <iostream>
#include <stdio.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>

#define IMROWS 1204
#define IMCOLS 1880 
#define IMCHANNELS 1
#define KERNELRADIUS 8 // 8x2 + 1

#define BLOCKDIM_1 16
#define BLOCKDIM_2 4
#define STEP 4

__constant__ float c_kernel[KERNELRADIUS * 2 + 1];

__global__ void conv2d_row(float *d_input, float *d_output){

    // fits nicely with 32 // gives 10x32
    __shared__ float s_data[((KERNELRADIUS*2 + STEP * BLOCKDIM_1) * BLOCKDIM_2 / 32)][32];

    /*
            8                64                   8
           __________________________________________
        8 | ________|_____________________|___________|
            halo           main data        halo

            16 first threads of the warp
            the first element of main data will read the leftest element of the left halo
            the second... the second and so on - 16 threads, 16 reads on total

            then we shift one, so the first threads reads the second element of the left halo and so on

            16 next threads same but with second line

            on shared memory the layout is:

            1 -> 16 (first line) |   17 -> 32 (second line)
            1|2|3|4|5........|16 |1|2|3|4|5........|16|    first read, the whole warp reads the first line of shared mem, no bank conflicts
            17|...               |17...
            
            then we shift as we said, so the first half reads from 2 to 17, same for the second - no bank conflicts with this shifting strat.

            and yes, hopefully it works.
            
            ---

            since we have a tile bigger than thread block size by 4 - each thread does 5 (80/16) iterations
            each iteration a warp fills entire line of shared mem

            --

            on constant memory, each read will correspond to the same element of the kernel

            --

            what is dislike is that this is dependent of the sizes, what happens when they are not divisible nicely? -- if elses to the rescue?

            --

            my head is becoming bigger
    */


   int half_warp_lane = threadIdx.y & 1;
   int warp_id = (threadIdx.y * blockDim.x + threadIdx.x)/32;
   int temp = BLOCKDIM_1 * STEP + 2 * KERNELRADIUS / BLOCKDIM_1;
   int temp_warp_id = temp * warp_id;

   // load data
    for(int i = 0; i < temp; i++){
        s_data[BLOCKDIM_1 * i + threadIdx.x + temp_warp_id][threadIdx.x + BLOCKDIM_1 * half_warp_lane] = 0;
    }

};

__global__ void conv2d_col(float *d_input, float *d_output){



};

// add err checking
void processing(float* h_input, float *h_output, float *h_kernel){
    // variables for device
    float *d_input, *d_intermediate_output, *d_output, *d_kernel;
    int buf_size = IMROWS * IMCOLS * sizeof(float);
    int kernel_size = (KERNELRADIUS*2+1) * sizeof(float);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocate device mem
    cudaMalloc((void **)&d_input, buf_size);
    cudaMalloc((void **)&d_intermediate_output, buf_size);
    cudaMalloc((void **)&d_output, buf_size);

    // send data
    cudaMemcpy(d_input, h_input, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_size);

    // dimensions
    int temp = STEP * BLOCKDIM_1;
    dim3 dimBlock_row(BLOCKDIM_1, BLOCKDIM_2);
    dim3 dimGrid_row((IMCOLS + temp - 1)/temp, (IMROWS + BLOCKDIM_2 - 1)/BLOCKDIM_2);

    // where magic happens
    // row
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_row<<<dimGrid_row, dimBlock_row>>>(d_input, d_intermediate_output);
    cudaProfilerStop();
    sdkStopTimer(&timer);
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));

    dim3 dimBlock_col(BLOCKDIM_2, BLOCKDIM_1);
    dim3 dimGrid_col((IMCOLS + BLOCKDIM_2 - 1)/BLOCKDIM_2, (IMROWS + temp - 1)/temp);
    
    //col
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    cudaProfilerStart();
    conv2d_col<<<dimGrid_col, dimBlock_col>>>(d_intermediate_output, d_output);
    cudaProfilerStop();
    sdkStopTimer(&timer);
    printf("Processing Time: %.2f ms\n", sdkGetTimerValue(&timer));
    
    cudaDeviceSynchronize();

    // return data
    cudaMemcpy(h_output, d_output, buf_size, cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_intermediate_output);

}
