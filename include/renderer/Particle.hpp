#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "Definitions.hpp"

struct Drop
{
    glm::f32vec3 Position;
    glm::f32vec4 Color;
};

GLvoid marching_cube(int idx, int idy, int idz, int NX, int NY, int NZ, float *mass, std::vector<Drop> &fluid);

#endif // !PARTICLE_H_