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
#define REDUCTION_STEPS 4 // each block thread loads 4 float4, 256B
#define SCAN_STEPS 2 // each block thread loads 2 float4, 128B
#define SCAN_SMEM_WIDTH (BLOCKSIZE/32)


// SIMT Kogge-Stone scan kernel
__device__ __inline__ void scan_warp(float* input, int indx = threadIdx.x){
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
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
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

    float *a = s_data;

    #pragma unroll
    for(int d = LOG2_BLOCKSIZE; d > LOG2_STEPSBEFOREWARPSIZE; d--){ // 9 -> 5

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
__device__ __inline__ void scan_warp_merrill_srts(float s_data[32 * SCAN_STEPS][SCAN_SMEM_WIDTH + 1 + 1], int indx = threadIdx.x){
    int lane = indx & 31;
    
    if (lane >= 1)  s_data[indx][SCAN_SMEM_WIDTH + 1] = s_data[indx - 1][SCAN_SMEM_WIDTH + 1] + s_data[indx][SCAN_SMEM_WIDTH + 1];
    if (lane >= 2)  s_data[indx][SCAN_SMEM_WIDTH + 1] = s_data[indx - 2][SCAN_SMEM_WIDTH + 1] + s_data[indx][SCAN_SMEM_WIDTH + 1];
    if (lane >= 4)  s_data[indx][SCAN_SMEM_WIDTH + 1] = s_data[indx - 4][SCAN_SMEM_WIDTH + 1] + s_data[indx][SCAN_SMEM_WIDTH + 1];
    if (lane >= 8)  s_data[indx][SCAN_SMEM_WIDTH + 1] = s_data[indx - 8][SCAN_SMEM_WIDTH + 1] + s_data[indx][SCAN_SMEM_WIDTH + 1];
    if (lane >= 16) s_data[indx][SCAN_SMEM_WIDTH + 1] = s_data[indx - 16][SCAN_SMEM_WIDTH + 1] + s_data[indx][SCAN_SMEM_WIDTH + 1];
}

// merrill_srts scan kernel
__global__ void scan(float4 *d_input, float *seeds, float4 *d_output){

    __shared__ float s_data[32 * SCAN_STEPS][SCAN_SMEM_WIDTH + 1 + 1]; // 1 for no bank conflict and another one for the result of the warp scan

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    d_input += idx;
    float seed = seeds[blockIdx.x];

    float4 item[SCAN_STEPS];

    #pragma unroll
    for(int i = 0; i < SCAN_STEPS; i++){
        item[i] = d_input[i * BLOCKSIZE];
        item[i].y += item[i].x + seed;
        item[i].z += item[i].y;
        item[i].w += item[i].z;
        s_data[32 * i + threadIdx.x & 31][threadIdx.x & (SCAN_SMEM_WIDTH - 1)] = item[i].w;
    }

    __syncthreads();

    // serial reduce
    // each warp going to do 32 rows of smem
    // this is funny, we could do each warp doing 32, ending up on 32*32*16 elements on shared mem
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
        item[i].x += s_data[32 * i + threadIdx.x & 31][SCAN_SMEM_WIDTH + 1];
        item[i].y += s_data[32 * i + threadIdx.x & 31][SCAN_SMEM_WIDTH + 1];
        item[i].z += s_data[32 * i + threadIdx.x & 31][SCAN_SMEM_WIDTH + 1];
        item[i].w += s_data[32 * i + threadIdx.x & 31][SCAN_SMEM_WIDTH + 1];
        d_output[idx + i * BLOCKSIZE] = item[i];
    }

}

// two level reduce then scan - middle scan kernel



// main