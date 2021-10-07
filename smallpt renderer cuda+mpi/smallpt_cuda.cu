// https://github.com/vchizhov/smallpt-explained/blob/master/smallpt_explained.cpp

#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <math.h>

#define w 1024
#define h 768
#define samps 64

#define BLOCKDIM 128

////

struct Ray{
    float3 origin, dir;
    Ray(float3 origin, float3 dir) : origin(origin), dir(dir) {}
};

////





int main(){

    // https://en.wikipedia.org/wiki/Ray_tracing_(graphics)#Calculate_rays_for_rectangular_viewport

    Ray cam(make_float3(50, 52 , 295.6), normalize(make_float3(0, -0.042612, -1)));

    float aspectRatio = w/h;

    float vfov = 0.502643;
    float fovScale = 2 * tan(0.5*vfov);

    float3 cx = make_float3(aspectRatio, 0, 0) * fovScale;
    float3 cy = normalize(cross(cx, cam.dir)) * fovScale;

    float3 r;
    float3 h_img[w*h];
    memset(h_img, 0, sizeof(float3) * h * w);

    // cuda variables

    curandState_t *devStates;
    cudaMalloc((void **)&devStates, sizeof(curandState) * h * w);

    float3 *d_img;
    cudaMalloc((void **)&d_img, sizeof(float3) * h * w);

    dim3 dimBlock(BLOCKDIM, BLOCKDIM);
    dim3 dimGrid((w + BLOCKDIM - 1)/BLOCKDIM, (h + BLOCKDIM - 1)/BLOCKDIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    cudaEventRecord(start, 0);
    smallpt_kernel<<<dimGrid, dimBlock>>>(d_img, devStates);
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "Ray Tracing time: %.8f ms\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_img);
    cudaFree(devStates);

    return 0;
}