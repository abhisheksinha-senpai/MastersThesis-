#ifndef MODEL_H
#define MODEL_H

#include "Definitions.hpp"
#include "Shader.hpp"
#include "Mesh.hpp"

class Model
{
    public:
    Model(){};
    Model(char *path, glm::f32vec3 scale, glm::f32vec3 origin);
    void Draw(Shader &shader, glm::f32vec3 scale);
    std::vector<Mesh> meshes;
    float minX= 99999, maxX=-99999, minY=99999, maxY=-99999, minZ=99999, maxZ=-99999;
    ~Model();
    private:
    // model data
    
    std::vector<Texture> textures_loaded;
    std::string directory;
    void loadModel(std::string path, glm::f32vec3 scale, glm::f32vec3 origin);
    void processNode(aiNode *node, const aiScene *scene);
    Mesh processMesh(aiMesh *mesh, const aiScene *scene);
    std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName);
};

#endif