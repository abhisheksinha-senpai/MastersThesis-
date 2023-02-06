#ifndef PARTICLE_SYSTEM_H_
#define PARTICLE_SYSTEM_H_

#include "Definitions.hpp"
#include "Particle.hpp"
#include "Shader.hpp"
#include "ResourceManager.hpp"

class ParticleSystem
{
    std::vector<Drop> fluid;
    Shader fluidShader;
    unsigned int VAO_1, VBO_1;
public:
    ParticleSystem(int NX, int NY, int NZ, glm::f32vec3 model_scale, float *mass);
    void update_particles(int NX, int NY, int NZ, float *mass, float *ux, float *uy, float *uz);
    void draw_particles(int SCR_WIDTH, int SCR_HEIGHT, glm::vec3 cameraPos, glm::vec3 cameraFront, glm::vec3 cameraUp, glm::f32vec3 dis_scale);
    ~ParticleSystem();
};

#endif // !PARTICLE_SYSTEM_H_