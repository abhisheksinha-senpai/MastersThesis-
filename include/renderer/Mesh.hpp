#ifndef MESH_H
#define MESH_H

#include "Definitions.hpp"
#include "Shader.hpp"

struct Texture {
    unsigned int id;
    std::string type;
    std::string path; // store path of texture to compare with other textures
};


class Mesh 
{
    public:
    // mesh data
    glm::f32vec3 origin_pos;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    //std::vector<Edge> edges;

    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures);//, std::vector<Edge> edges);
    void Draw(Shader &shader, glm::f32vec3 scale);
    unsigned int VAO_mesh, VBO_mesh, EBO_mesh;
    void setupMesh();
    ~Mesh();
};

#endif // !MESH_H
