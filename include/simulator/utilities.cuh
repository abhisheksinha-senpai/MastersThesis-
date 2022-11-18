#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <string.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#define _USE_MATH_DEFINES
#include <math.h>

const dim3 nThreads(64, 1, 1);

#define checkCudaErrors(err)  __checkCudaErrors(err,#err,__FILE__,__LINE__)
#define getLastCudaError(msg)  __getLastCudaError(msg,__FILE__,__LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line )
{
    if(err != cudaSuccess)
    {
        fprintf(stderr, "\nCUDA error at %s(%d)\"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "\nCUDA error at %s(%d): [%d] %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}
 
__device__ __forceinline__ unsigned int gpu_scalar_index(unsigned int x, unsigned int y, unsigned int z,int *lb_sim_domain)
{
    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    return (z*(NX*NY) + NX*y + x);
}

__device__ __forceinline__ unsigned int gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int z, unsigned int d, int *lb_sim_domain)
{
    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];
    return ((d)*NX*NY*NZ + z*(NX*NY) + NX*y + x);
}

__host__ void getDeviceInfo();
#endif