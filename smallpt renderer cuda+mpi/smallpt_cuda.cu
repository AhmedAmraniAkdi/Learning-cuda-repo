// https://github.com/vchizhov/smallpt-explained/blob/master/smallpt_explained.cpp

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include <curand_kernel.h>
#include <curand.h>

#include <stdio.h>

#include "spheres_rays.cuh"


#define W 1024
#define H 768
#define samps 64 // samples per subpixel

#define BLOCKDIMX 32
#define BLOCKDIMY 4


__global__ void smallpt_kernel(float3 *d_img, curandStatePhilox4_32_10_t *state, float3 cx, float3 cy, Ray cam){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int id = idy * W + idx;

    if(idx >= W || idy >= H) return;

    int i = (H - idy - 1)*W + idx; // img comes reversed

    d_img += id;

    curand_init(id, id, 0, &state[id]);

    float3 r = make_float3(0);

    float3 acum = make_float3(0);

    acum.x = curand_uniform (&state[id]);

    /*#pragma unroll
    for(int sy = 0; sy < 2; sy++){

        #pragma unroll
        for(int sx = 0; sx < 2; sx++, r = make_float3(0)){

            #pragma unroll
            for(int s = 0; s < samps ; s++){// each sample is independent, can have another grid doing samps/2 and then atomic sum

                float r1 = 2 * curand_uniform (&state[id]);
                float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                float r2 = 2 * curand_uniform (&state[id]);
                float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                float3 d = cx * (((sx + .5 + dx) / 2 + idx) / W - .5) +
                    cy * (((sy + .5 + dy) / 2 + idy) / H - .5) + cam.dir;

                //r = r + radiance(Ray(cam.origin + d * 140, normalize(d)), 0, state, id)*(1. / samps);
                r = make_float3(0);

            }

            acum = acum + clamp(r, 0, 1) *.25;
        }
    }*/

    d_img[0] = acum;
}


int main(){

    // https://en.wikipedia.org/wiki/Ray_tracing_(graphics)#Calculate_rays_for_rectangular_viewport

    Ray cam(make_float3(50, 52 , 295.6), normalize(make_float3(0, -0.042612, -1)));

    float aspectRatio = W/H;

    float vfov = 0.502643;
    float fovScale = 2 * tan(0.5*vfov);

    float3 cx = make_float3(aspectRatio, 0, 0) * fovScale;
    float3 cy = normalize(cross(cx, cam.dir)) * fovScale;

    float3 *h_img = (float3 *)malloc(sizeof(float3) * H * W);

	printf("hi there\n");
    
	// cuda variables

    curandStatePhilox4_32_10_t *devStates;
    cudaMalloc((void **)&devStates, sizeof(curandStatePhilox4_32_10_t) * W * H);
	checkCudaErrors(cudaGetLastError());

    float3 *d_img;
    cudaMalloc((void **)&d_img, sizeof(float3) * H * W);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpyToSymbol(spheres, &spheres_cpu, sizeof(spheres_cpu));
	checkCudaErrors(cudaGetLastError());

    dim3 dimBlock(BLOCKDIMX, BLOCKDIMY);
    dim3 dimGrid((W + BLOCKDIMX - 1)/BLOCKDIMX, (H + BLOCKDIMY - 1)/BLOCKDIMY);

	printf("hi there\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    cudaEventRecord(start, 0);
    smallpt_kernel<<<dimGrid, dimBlock>>>(d_img, devStates, cx, cy, cam);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf( "Ray Tracing time: %.8f ms\n", elapsed_time);
    checkCudaErrors(cudaGetLastError());

	cudaMemcpy(h_img, d_img,  H * W * sizeof(float3), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());

	/*FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", W, H, 255);
	for (int i = 0; i < W*H; i++)
		fprintf(f, "%d %d %d ", toInt(h_img[i].x), toInt(h_img[i].y), toInt(h_img[i].z));
*/
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_img);
    cudaFree(d_img);
    cudaFree(devStates);

    return 0;
}