#ifndef HELPER_H_
#define HELPER_H_

#include "Definitions.hpp"
#include "ResourceManager.hpp"
#include "Shader.hpp"
#include "Model.hpp"
#include "ParticleSystem.hpp"
#include "Domain.hpp"

#include "utilities.cuh"
#include "ImmersedBoundary.cuh"
#include "LatticeBoltzmann.cuh"
#include "PBD.cuh"

extern float GRAV_CONST;

__host__ void display_init(GLFWwindow** window);

__host__ void domain_init(glm::f32vec3 simDimention, float **rho, float **ux, float **uy,float **uz, int dim);

__host__ void scene_init(float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu,
                         float *rho, float *ux, float *uy, float *uz, 
                         int NX, int NY, int NZ);

__host__ void model_init(ResourceManager &r_manager, Shader &ourShader, Model &ourModel, int NX, int NY, int NZ,
                         glm::f32vec3 scale, glm::f32vec3 origin);

__host__ void preDisplay();

__host__ void postDisplay(GLFWwindow** window);

__host__ void displayFluid(float *rho, float*ux, float *uy, float *uz,
                            float *rho_gpu, float *ux_gpu, float*uy_gpu, float* uz_gpu,
                            int NX, int NY, int NZ, ParticleSystem &fluid, glm::f32vec3 dis_scale);

__host__ void displayModel( GLFWwindow** window, Shader& shader, Model& objmodel, glm::f32vec3 scale);

__host__ void displayDomain(Geometry &fluidDomain);

__host__ void scene_cleanup(Vertex **nodeLists, Vertex **nodeData, int *vertex_size_per_mesh,
                            float *rho, float *ux, float *uy, float *uz);
#endif