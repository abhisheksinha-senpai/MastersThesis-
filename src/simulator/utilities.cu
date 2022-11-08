#include "utilities.cuh"

__host__ void getDeviceInfo()
{
    double bytesPerMiB = 1024.0*1024.0;
    double bytesPerGiB = 1024.0*1024.0*1024.0;
    
    checkCudaErrors(cudaSetDevice(0));
    int deviceId = 0;
    checkCudaErrors(cudaGetDevice(&deviceId));
    
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
    
    size_t gpu_free_mem, gpu_total_mem;
    checkCudaErrors(cudaMemGetInfo(&gpu_free_mem,&gpu_total_mem));

    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n",deviceProp.name);
    printf("    multiprocessors: %d\n",deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n",deviceProp.major,deviceProp.minor);
    printf("      global memory: %.1f MiB\n",deviceProp.totalGlobalMem/bytesPerMiB);
    printf("        free memory: %.1f MiB\n",gpu_free_mem/bytesPerMiB);
    return;
}