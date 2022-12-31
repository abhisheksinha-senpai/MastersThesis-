#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <string.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#define _USE_MATH_DEFINES
#include <math.h>

const dim3 nThreads(64, 1, 1);

extern float *Fx_gpu, *Fy_gpu, *Fz_gpu;
extern float *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;
extern float *f1_gpu, *f2_gpu, *feq_gpu, *source_term_gpu;
extern float *mass_gpu, *strain_rate_gpu, *delMass;
extern float *empty_filled_cell;
//cell Type: 0-fluid, 1-interface, 2-gas, 3-obstacle
extern unsigned int *cell_type_gpu;
extern float *temp_cell_type_gpu;
extern float *count_loc;

__device__ unsigned int FLUID = (unsigned int)(1 << 0);
__device__ unsigned int  INTERFACE  = (unsigned int)(1 << 1);
__device__ unsigned int  EMPTY =  (unsigned int)(1 << 2);
__device__ unsigned int  OBSTACLE =  (unsigned int)(1 << 3);
__device__ unsigned int  NO_FLUID_NEIGH =  (unsigned int)(1 << 4);
__device__ unsigned int  NO_EMPTY_NEIGH =  (unsigned int)(1 << 5);
__device__ unsigned int  NO_IFACE_NEIGH =  (unsigned int)(1 << 6);
__device__ unsigned int  IF_TO_FLUID = ((unsigned int)(1 << 1)|(unsigned int)(1 << 0));
__device__ unsigned int  IF_TO_EMPTY = ((unsigned int)(1 << 1)|(unsigned int)(1 << 2));
__device__ unsigned int  EMPTY_TO_IF = (unsigned int)(1 << 0)|((unsigned int)(1 << 1)|(unsigned int)(1 << 2));

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