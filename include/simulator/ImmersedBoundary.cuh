#ifndef IMMERSED_BOUNDARY_H_
#define IMMERSED_BOUNDARY_H_

#include "utilities.cuh"
#include "Definitions.hpp"

struct Vertex;

__host__ void IBM_init(int NX, int NY, int NZ,
                       float **Fx_gpu, float **Fy_gpu, float **Fz_gpu, 
                       int num_mesh, Vertex **nodeLists, 
                       int *vertex_size_per_mesh, Vertex** nodeData, 
                       cudaStream_t *streams, float s_c);

__host__ void IBM_force_spread(Vertex **nodeLists, int *nodeList_size, int num_threads, int num_mesh,
                               float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, cudaStream_t *streams);

__host__ void IBM_cleanup(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu,
                          int num_mesh, Vertex **nodeLists);

__host__ void IBM_advect_bound(Vertex **nodeLists, int *nodeList_size, int num_threads, int num_mesh,
                                float *ux_gpu, float *uy_gpu, float *uz_gpu, cudaStream_t *streams);

__host__ void IBM_force_spread_RB(Vertex **nodeLists, int *nodeList_size, int num_threads, int num_mesh,
                                  float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float Ct,
                                  float *rho_gpu,float *ux_gpu, float *uy_gpu, float *uz_gpu, cudaStream_t *streams);

__host__ void update_IB_params(Vertex **nodeLists, int *nodeList_size, int num_threads, int num_mesh, float Ct,
                               glm::f32vec3 Velocity_RB, float *ux_gpu, float *uy_gpu, float *uz_gpu, cudaStream_t *streams);
#endif