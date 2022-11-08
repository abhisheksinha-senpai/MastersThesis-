#ifndef LBM_H_
#define LBM_H__

#include "utilities.cuh"

struct Vertex;

 __host__ void LB_simulate(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, 
                           float *f1_gpu, float* f2_gpu, float *feq_gpu, float *source_term_gpu, 
                           float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, 
                           int NX, int NY, int NZ, void (*cal_force_spread)(Vertex**, int*, int, int, float*, float*, float*, cudaStream_t *),
                           void (*advection_force)(Vertex **, int *, int, int,float *, float *, float *, cudaStream_t *),
                           Vertex **nodeLists, int *vertex_size_per_mesh, int num_threads, int num_mesh, cudaStream_t *streams);

__host__ void LB_init(int NX, int NY, int NZ, float Reynolds, float mu,
                            float **f1_gpu, float **f2_gpu, float **feq_gpu, float **source_term_gpu, 
                            float **rho_gpu, float **ux_gpu, float **uy_gpu, float **uz_gpu, 
                            float **rho, float **ux, float **uy, float **uz, 
                            float **Fx_gpu, float **Fy_gpu, float **Fz_gpu,
                            cudaStream_t *streams);

__host__ void LB_cleanup(float *f1_gpu, float* f2_gpu, float *feq_gpu, float *source_term_gpu, 
                         float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu);

#endif