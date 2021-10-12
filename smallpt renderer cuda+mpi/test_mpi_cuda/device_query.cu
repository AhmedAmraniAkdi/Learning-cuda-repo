#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

void device_query(int my_rank){  
  
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("%d - Detected %d CUDA Capable device(s)\n", my_rank, deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("%d -  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", my_rank, 
                driverVersion / 1000, (driverVersion % 100) / 10,
                runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("%d -  CUDA Capability Major/Minor version number:    %d.%d\n", my_rank,
                deviceProp.major, deviceProp.minor);
    }
}