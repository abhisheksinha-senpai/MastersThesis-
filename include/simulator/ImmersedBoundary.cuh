#ifndef IMMERSED_BOUNDARY_H_
#define IMMERSED_BOUNDARY_H_

#include "utilities.cuh"
#include "Definitions.hpp"

struct Vertex;

__host__ float IBM_init(int NX, int NY, int NZ, int num_mesh, Vertex** nodeData, cudaStream_t *streams, float spring_const);

__host__ void IBM_force_spread(int num_threads, int num_mesh,cudaStream_t *streams);

__host__ void IBM_cleanup(int num_mesh);

__host__ void IBM_advect_bound(int num_threads, int num_mesh, cudaStream_t *streams);

__host__ void IBM_force_spread_RB(int num_threads, int num_mesh, float Ct, cudaStream_t *streams);

__host__ void update_IB_params(int num_threads, int num_mesh, float Ct, float Cl, glm::f32vec3 Velocity_RB, cudaStream_t *streams);
#endif