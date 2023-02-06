#ifndef LBM_H_
#define LBM_H__

#include "glm/glm.hpp"
#include "Definitions.hpp"
#include "utilities.cuh"


struct Vertex;

__host__ float LB_init(int NX, int NY, int NZ, float Reynolds, float mu, float **rho, float **ux, float **uy, float **uz,  dim3 num_threads, int dim);

__host__ void LB_cleanup();

__host__ void LB_simulate_RB(int NX, int NY, int NZ, float Ct, void (*cal_force_spread_RB)(dim3, int, float), 
                             void (*advect_velocity)(dim3, int), dim3 num_threads, int num_mesh, bool flag_FSI);
#endif