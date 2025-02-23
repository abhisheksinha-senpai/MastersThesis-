#include "Model.hpp"
#include <glm/gtx/string_cast.hpp>

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        h1 ^= (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        return h1;
    }
};

unsigned int TextureFromFile(const char *path, const std::string &directory, bool gamma);


Model::Model(char *path,  glm::f32vec3 scale, glm::f32vec3 origin)
{
    loadModel(path, scale, origin);
}

void Model::Draw(Shader &shader, glm::f32vec3 scale, glm::f32vec3 origin)
{
    for(unsigned int i = 0; i < meshes.size(); i++)
        meshes[i].Draw(shader, scale, origin);
}

void Model::loadModel(std::string path, glm::f32vec3 scale, glm::f32vec3 origin)
{
    
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate|aiProcess_FlipUVs);

    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return;
    }
    directory = path.substr(0, path.find_last_of("/"));
    processNode(scene->mRootNode, scene);
    
    for(int i=0;i<meshes.size();i++)
    {
        for(int j=0;j<meshes[i].vertices.size();j++)
        {
            meshes[i].vertices[j].Position.x = (meshes[i].vertices[j].Position.x - minX)*scale.x/(maxX-minX) + origin.x;
            meshes[i].vertices[j].Position.y = (meshes[i].vertices[j].Position.y - minY)*scale.y/(maxY-minY) + origin.y;
            meshes[i].vertices[j].Position.z = (meshes[i].vertices[j].Position.z - minZ)*scale.z/(maxZ-minZ) + origin.z;

            if((maxX-minX) == 0)
                printf("Same X\n");
            if((maxY-minY) == 0)
                printf("Same Z\n");
            if((maxZ-minZ) == 0)
                printf("Same Z\n");
            meshes[i].vertices[j].Base = meshes[i].vertices[j].Position;
        }
    }
    printf("%f %f %f\n",  scale.x,  scale.y,  scale.z);
    printf("%s, %s\n", glm::to_string(glm::f32vec3((scale.x/(maxX-minX)), (scale.y/(maxY-minY)), (scale.z/(maxZ-minZ)))).c_str(), glm::to_string(origin).c_str());
    printf("Modeling done\n");
    printf("Total mesh %ld\n", meshes.size());
}

void Model::processNode(aiNode *node, const aiScene *scene)
{
    // process all the node’s meshes (if any)
    for(unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }
    // then do the same for each of its children
    for(unsigned int i = 0; i < node->mNumChildren; i++)
        processNode(node->mChildren[i], scene);
}

Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene)
{
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    // std::unordered_map<std::pair<float, float>, bool, pair_hash> edgeList;
    // std::vector<Edge> edges;
    for(unsigned int i = 0; i < mesh->mNumVertices; i++)
    {
        Vertex vertex;
        // process vertex positions, normals and texture coordinates
        glm::vec3 vector;
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.Position = vector;
        vertex.Base = vector;
        vertex.Velocity = glm::f32vec3(0.0f);

        minX = glm::min(minX, vector.x);
        maxX = glm::max(maxX, vector.x);

        minY = glm::min(minY, vector.y);
        maxY = glm::max(maxY, vector.y);

        minZ = glm::min(minZ, vector.z);
        maxZ = glm::max(maxZ, vector.z);

        if (mesh->HasNormals())
        {
            vector.x = mesh->mNormals[i].x;
            vector.y = mesh->mNormals[i].y;
            vector.z = mesh->mNormals[i].z;
            vertex.Normal = vector;
        }

        if(mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
        {
            glm::vec2 vec;
            vec.x = mesh->mTextureCoords[0][i].x; 
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex.TexCoords = vec;
        }
        else
            vertex.TexCoords = glm::vec2(0.0f, 0.0f);

        vertices.push_back(vertex);
    }
    // process indices
    for(unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        for(unsigned int j = 0; j < face.mNumIndices; j++)
            indices.push_back(face.mIndices[j]);
    }

    // process material

    aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
    std::vector<Texture> diffuseMaps = loadMaterialTextures(material,
    aiTextureType_DIFFUSE, "texture_diffuse");
    textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
    std::vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
    textures.insert(textures.end(), specularMaps.begin(),specularMaps.end());

    return Mesh(vertices, indices, textures);//, edges);
}

std::vector<Texture> Model::loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName)
{
    std::vector<Texture> textures;
    for(unsigned int i = 0; i < mat->GetTextureCount(type); i++)
    {
        aiString str;
        mat->GetTexture(type, i, &str);
        bool skip = false;
        for(unsigned int j = 0; j < textures_loaded.size(); j++)
        {
            if(std::strcmp(textures_loaded[j].path.data(),str.C_Str()) == 0)
            {
                textures.push_back(textures_loaded[j]); 
                skip = true;
                break;
            }
        }
        if(!skip)
        { // if texture hasn’t been loaded already, load it
            Texture texture;
            texture.id = TextureFromFile(str.C_Str(), directory, false);
            texture.type = typeName;
            texture.path = str.C_Str();
            textures.push_back(texture);
            textures_loaded.push_back(texture); // add to loaded textures
        }
    }
    return textures;
}

unsigned int TextureFromFile(const char *path, const std::string &directory, bool gamma)
{
    std::string filename = std::string(path);
    filename = directory + '/' + filename;

    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

Model::~Model()
{
    printf("Model deletion done\n");
}