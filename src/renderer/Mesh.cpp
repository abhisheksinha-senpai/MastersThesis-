#include "Mesh.hpp"

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures)//, std::vector<Edge> edges)
{
    this->vertices = vertices;
    this->indices = indices;
    this->textures = textures;
    // this->edges = edges;
    setupMesh();
}

void Mesh::setupMesh()
{
    glGenVertexArrays(1, &VAO_mesh);
    glGenBuffers(1, &VBO_mesh);
    glGenBuffers(1, &EBO_mesh);
    glBindVertexArray(VAO_mesh);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_mesh);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), &vertices[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_mesh);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
    // vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Position));
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    glBindVertexArray(0);
}

void Mesh::Draw(Shader &shader, glm::f32vec3 scale, glm::f32vec3 origin)
{
    unsigned int diffuseNr = 1;
    unsigned int specularNr = 1;
    unsigned int VBO_1, EBO_1, VAO_1;
    for(unsigned int i = 0; i < textures.size(); i++)
    {
        glActiveTexture(GL_TEXTURE0 + i); // activate texture unit first
        // retrieve texture number (the N in diffuse_textureN)
        std::string number;
        std::string name = textures[i].type;
        if(name == "texture_diffuse")
            number = std::to_string(diffuseNr++);
        else if(name == "texture_specular")
            number = std::to_string(specularNr++);
        std::string temp_str = "material." + name + number;
        glUniform1f(glGetUniformLocation(shader.get_shader_pgm(), temp_str.c_str()),i); 
        glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }
    glActiveTexture(GL_TEXTURE0);
    glm::mat4 model = glm::mat4(1.0f);

    model = glm::scale(model,scale);

    glUniformMatrix4fv(glGetUniformLocation(shader.get_shader_pgm(), "model"), 1, GL_FALSE, glm::value_ptr(model));
    // draw mesh
    glGenVertexArrays(1, &VAO_mesh);
    glGenBuffers(1, &VBO_mesh);
    glGenBuffers(1, &EBO_mesh);
    glBindVertexArray(VAO_mesh);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_mesh);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_mesh);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_mesh);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), &vertices[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_mesh);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
    // vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),  (void*)offsetof(Vertex, Position));
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    glDrawElements(GL_TRIANGLES, this->indices.size(), GL_UNSIGNED_INT, 0);
    model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader.get_shader_pgm(), "model"), 1, GL_FALSE, glm::value_ptr(model));
    glBindVertexArray(0);
    glDeleteBuffers(1, &VBO_mesh);
    glDeleteBuffers(1, &EBO_mesh);
    glDeleteVertexArrays(1, &VAO_mesh);
}

Mesh::~Mesh()
{
    glDeleteBuffers(1, &VBO_mesh);
    glDeleteBuffers(1, &EBO_mesh);
    glDeleteVertexArrays(1, &VAO_mesh);
}