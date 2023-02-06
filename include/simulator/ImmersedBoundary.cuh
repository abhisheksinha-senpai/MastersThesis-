#ifndef IMMERSED_BOUNDARY_H_
#define IMMERSED_BOUNDARY_H_

#include "utilities.cuh"
#include "Definitions.hpp"

struct Vertex;

__host__ float IBM_init(int NX, int NY, int NZ, int num_mesh, Vertex** nodeData, float spring_const);

__host__ void IBM_cleanup(int num_mesh);

__host__ void IBM_vel_spread_RB(dim3 num_threads, int num_mesh);

__host__ void IBM_force_spread_RB(dim3 num_threads, int num_mesh, float Ct);
#endif