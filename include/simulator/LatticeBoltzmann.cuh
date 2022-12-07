#ifndef LBM_H_
#define LBM_H__

#include "glm/glm.hpp"
#include "Definitions.hpp"
#include "utilities.cuh"


struct Vertex;

__host__ float LB_init(int NX, int NY, int NZ, float Reynolds, float mu,
                      float **rho, float **ux, float **uy, float **uz,
                      cudaStream_t *streams);

__host__ void LB_cleanup();

__host__ void LB_simulate_RB(int NX, int NY, int NZ, float Ct,
                             void (*cal_force_spread_RB)(int, int, float, cudaStream_t *), 
                             void (*advect_velocity)(int, int, cudaStream_t *),
                             int num_threads, int num_mesh, cudaStream_t *streams);
#endif