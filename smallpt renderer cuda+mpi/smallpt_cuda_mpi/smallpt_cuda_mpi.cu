// idea is to divide the samples between gpus using mpi
// on 1 gpu, we can take advante of multi process service

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include <curand_kernel.h>
#include <curand.h>

#include <stdio.h>

#include "spheres_rays.cuh"
#include "radiance.cuh"

#define num_processes_mpi 4

#define W 1024
#define H 768
#define samps 1024 // samples per subpixel

#define BLOCKDIMX 32
#define BLOCKDIMY 2
#define XSTEP 1

//https://forums.developer.nvidia.com/t/curand-init-sequence-number-problem/56573 however xorwow is half the time of philox
__global__ void smallpt_kernel(float3 *d_img, /*curandStatePhilox4_32_10_t*/ curandState_t *state, float3 cx, float3 cy, Ray cam, int my_rank){

    #pragma unroll
    for(int step = 0; step < XSTEP; step++){

        int idx = blockIdx.x * blockDim.x * XSTEP + threadIdx.x + step * BLOCKDIMX;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        int id = idy * W + idx;

        if(idx >= W || idy >= H) return;

        int i = (H - idy - 1 ) * W + idx; // img comes reversed

        if(step == 0) {
            curand_init(id + W * H * my_rank, 0, 0, &state[id]); // diff seed per process
        }

        float3 r = make_float3(0);

        float3 acum = make_float3(0);

        #pragma unroll
        for(int sy = 0; sy < 2; sy++){

            #pragma unroll
            for(int sx = 0; sx < 2; sx++, r = make_float3(0)){

                #pragma unroll
                for(int s = 0; s < samps/num_processes_mpi ; s++){// each sample is independent, can have another grid doing samps/2 and then atomic sum

                    float r1 = 2 * curand_uniform (&state[id]);
                    float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                    float r2 = 2 * curand_uniform (&state[id]);
                    float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                    float3 d = cx * (((sx + .5 + dx) / 2 + idx) / W - .5) +
                        cy * (((sy + .5 + dy) / 2 + idy) / H - .5) + cam.dir;

                    r = r + radiance(Ray(cam.origin + d * 130, normalize(d)), state, id) * (1./samps);

                }
                acum = acum + r * 0.25;
            }
        }

        d_img[i] = acum ;//clamp(acum, 0, 1);

    }

}


extern "C" void smallpt_main(int my_rank, float *h_img_output){

    // https://en.wikipedia.org/wiki/Ray_tracing_(graphics)#Calculate_rays_for_rectangular_viewport

    Ray cam(make_float3(50, 52 , 295.6), normalize(make_float3(0, -0.042612, -1)));

    float aspectRatio = W/H;

    float vfov = 0.502643;
    float fovScale = 2 * tan(0.5*vfov);

    float3 cx = make_float3(aspectRatio, 0, 0) * fovScale;
    float3 cy = normalize(cross(cx, cam.dir)) * fovScale;

    //float3 *h_img = (float3 *)malloc(sizeof(float3) * H * W);
    float3 *h_img = (float3 *) h_img_output;
    
	// cuda variables

    /*curandStatePhilox4_32_10_t*/ curandState_t *devStates;
    cudaMalloc((void **)&devStates, sizeof(/*curandStatePhilox4_32_10_t*/ curandState_t) * W * H );
	checkCudaErrors(cudaGetLastError());

    float3 *d_img;
    cudaMalloc((void **)&d_img, sizeof(float3) * H * W);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpyToSymbol(spheres, &spheres_cpu, sizeof(spheres_cpu));
	checkCudaErrors(cudaGetLastError());

    dim3 dimBlock(BLOCKDIMX, BLOCKDIMY);
    dim3 dimGrid((W + BLOCKDIMX * XSTEP - 1)/BLOCKDIMX/XSTEP, (H + BLOCKDIMY - 1)/BLOCKDIMY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    cudaEventRecord(start, 0);
    smallpt_kernel<<<dimGrid, dimBlock>>>(d_img, devStates, cx, cy, cam, my_rank);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("rank:%d Ray Tracing time: %.8f ms\n",my_rank, elapsed_time);
    checkCudaErrors(cudaGetLastError());

	cudaMemcpy(h_img, d_img,  H * W * sizeof(float3), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());

    /*for (int i = 0; i < W*H; i++){
        h_img[i].x = toInt(h_img[i].x);
        h_img[i].y = toInt(h_img[i].y);
        h_img[i].z = toInt(h_img[i].z);
    }*/

	/*FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", W, H, 255);
	for (int i = 0; i < W*H; i++)
		fprintf(f, "%d %d %d ", toInt(h_img[i].x), toInt(h_img[i].y), toInt(h_img[i].z));*/

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //free(h_img);
    cudaFree(d_img);
    cudaFree(devStates);

    //return 0;
}