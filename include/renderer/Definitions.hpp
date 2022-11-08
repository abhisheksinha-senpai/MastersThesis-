#ifndef DEFINITIONS_H
#define  DEFINITIONS_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <iterator>
#include <string>
#include <exception>
#include <system_error>
#include <cerrno>
#include <unordered_map>
#include <utility>

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "stb/stb_image.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

struct Vertex 
{
    glm::f32vec3 Position;
    glm::f32vec3 Normal;
    glm::f32vec2 TexCoords;
    glm::f32vec3 Base;
    // float mass;
    glm::f32vec3 Velocity;
};

// struct Edge
// {
//     glm::f32vec3 v1;
//     glm::f32vec3 v2;
//     float k;
//     float L0;
// };

#endif