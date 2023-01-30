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
#include <ctime>

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
    glm::f32vec3 Base_Velocity;
    glm::f32vec3 Prev_Position;
    glm::f32vec3 Velocity;
    glm::f32vec3 Force;
    float Area = 0.0f;
    float invMass = 0.0f;
};

struct Edge
{
    int vertID[2];
    float k;
    float L0;
};

struct Tetrahedral
{
    glm::i32vec3 face[4] = {glm::i32vec3{1,3,2}, glm::i32vec3{0,2,3}, glm::i32vec3{0,3,1}, glm::i32vec3{0,1,2}};
    int vertID[4];
    float V0;
    float k;
};

#endif