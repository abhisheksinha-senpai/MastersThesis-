#pragma once

#include "Definitions.hpp"
#include "Shader.hpp"
#include "ResourceManager.hpp"

class Geometry
{
private:
	glm::f32vec3 br;

	float body[252] = {
        // positions         // colors
         1.0f, -1.0f, -1.0f,  0.0f, 0.0f, 1.0f, 1.0f,// bottom
         1.0f,  1.0f, -1.0f,  0.0f, 0.0f, 1.0f,1.0f,
        -1.0f,  1.0f, -1.0f,  0.0f, 0.0f, 1.0f,1.0f,
        -1.0f,  1.0f, -1.0f,  0.0f, 0.0f, 1.0f,1.0f,
        -1.0f, -1.0f, -1.0f,  0.0f, 0.0f, 1.0f,1.0f,
         1.0f, -1.0f, -1.0f,  0.0f, 0.0f, 1.0f,1.0f,
         1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 1.0f, 1.0f,// top
         1.0f, -1.0f,  1.0f,  0.0f, 1.0f, 1.0f,1.0f,
        -1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 1.0f,1.0f,
         1.0f, -1.0f,  1.0f,  0.0f, 1.0f, 1.0f,1.0f,
        -1.0f, -1.0f,  1.0f,  0.0f, 1.0f, 1.0f,1.0f,
        -1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 1.0f,1.0f,
         1.0f, -1.0f, -1.0f,  1.0f, 0.0f, 1.0f,1.0f, // left
        -1.0f, -1.0f,  1.0f,  1.0f, 0.0f, 1.0f,1.0f,
         1.0f, -1.0f,  1.0f,  1.0f, 0.0f, 1.0f,1.0f,
        -1.0f, -1.0f, -1.0f,  1.0f, 0.0f, 1.0f,1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f, 0.0f, 1.0f,1.0f,
         1.0f, -1.0f, -1.0f,  1.0f, 0.0f, 1.0f,1.0f,
         1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 0.0f,1.0f, // right
        -1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 0.0f,1.0f,
        -1.0f,  1.0f, -1.0f,  0.0f, 1.0f, 0.0f,1.0f,
         1.0f,  1.0f, -1.0f,  0.0f, 1.0f, 0.0f,1.0f,
         1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 0.0f,1.0f,
        -1.0f,  1.0f, -1.0f,  0.0f, 1.0f, 0.0f,1.0f,
        -1.0f,  1.0f,  1.0f,  1.0f, 0.0f, 0.0f,1.0f, // back
        -1.0f, -1.0f,  1.0f,  1.0f, 0.0f, 0.0f,1.0f,
        -1.0f,  1.0f, -1.0f,  1.0f, 0.0f, 0.0f,1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f, 0.0f, 0.0f,1.0f,
        -1.0f, -1.0f, -1.0f,  1.0f, 0.0f, 0.0f,1.0f,
        -1.0f,  1.0f, -1.0f,  1.0f, 0.0f, 0.0f,1.0f,
         1.0f, -1.0f,  1.0f,  1.0f, 1.0f, 0.0f,1.0f, // front
         1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 0.0f,1.0f,
         1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 0.0f,1.0f,
         1.0f,  1.0f, -1.0f,  1.0f, 1.0f, 0.0f,1.0f,
         1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 0.0f,1.0f,
         1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 0.0f, 1.0f
    };
    unsigned int VAO, VBO;
    Shader domainShader;
    glm::f32vec3 position;
public:
    Geometry(glm::f32vec3 scale);
	void draw_geometry(int SCR_WIDTH, int SCR_HEIGHT, glm::vec3 cameraPos, glm::vec3 cameraFront, glm::vec3 cameraUp);
    ~Geometry();
};