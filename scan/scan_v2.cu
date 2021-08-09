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
#define LOG2_STEPSBEFOREWARPSIZE (LOG2_BLOCKSIZE - 5)
#define REDUCTION_STEPS 4 // each block thread loads 4 float4, 64B


// SIMT Kogge-Stone scan kernel
__device__ float scan_warp(float* input, int indx = threadIdx.x){
    int lane = indx & 31;
    
    if (lane >= 1) input[indx] = input[indx - 1] + input[indx];
    if (lane >= 2)  input[indx] = input[indx - 2] + input[indx];
    if (lane >= 4)  input[indx] = input[indx - 4] + input[indx];
    if (lane >= 8)  input[indx] = input[indx - 8] + input[indx];
    if (lane >= 16) input[indx] = input[indx - 16] + input[indx];
    
    return input[indx];
}

// SIMT Brent-Kung scan kernel - same as the merrill_srts reduction kernel but since it's the same as the warp size -> no need for __syncthreads()
// BUT BUT!!!!! since this is SIMT -> there is actually 0 gain from reducing the number of operations , so the scan-warp will be used.
__device__ __inline__ float reduce_warp(float* input, int indx = threadIdx.x){
    return scan_warp(input, indx);
}

// merrill_srts reduce kernel

__global__ void reduce(float4 *d_input, float *d_output){

    __shared__ float s_data[BLOCKSIZE * 2];//1 cell per thread + another blockdim for easier indx management
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    d_input += idx;
    d_output += blockIdx.x;

    float4 item = *d_input;
    float sum = 0;

    #pragma unroll
    for(int i = 0; i < REDUCTION_STEPS; i++){
        item = d_input[i * BLOCKSIZE];
        sum += item.w + item.x + item.y + item.z;
    }
    
    s_data[threadIdx.x] = sum;

    __syncthreads();

    // we reduce and put the result on the second half of shared memory

    float *a = s_data;

    #pragma unroll
    for(int d = LOG2_BLOCKSIZE; d > LOG2_STEPSBEFOREWARPSIZE; d--){ // 9 -> 5

        if( threadIdx.x < (1 << (d - 1)) ){
            a[(1 << d) + threadIdx.x] = a[2 * threadIdx.x] + a[2 * threadIdx.x + 1];
        }

        a = &a[(1 << d)];
        __syncthreads();

    }

    float val = 0;
    if((idx >> 5) == 0){ // warp 0
        val = reduce_warp(a); // sum will be at idx 31
    }

    // output the sum
    if(threadIdx.x == 31){
        d_output[0] = val;
    }
}


// merrill_srts scan kernel

// two level reduce then scan - middle scan kernel

// main