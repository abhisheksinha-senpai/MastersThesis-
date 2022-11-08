#include "ResourceManager.hpp"

void ResourceManager::load_shader(std::string file_name, std::string shader_type, std::string &location)
{
    try
    {
        std::ifstream shader_file;
        shader_file.clear(std::istream::eofbit | std::istream::failbit);
        try
        {
            shader_file.open(file_name);
        }
        catch (std::ios_base::failure& e) {
            std::cout << "Error: " << strerror(errno);
            throw;
        }

        std::stringstream shader;
        shader << shader_file.rdbuf();
        shader_file.close();
        location = shader.str();
    }
    catch(std::exception e)
    {
        throw std::runtime_error(std::string("ERROR::SHADER: Failed to read shader files"));
    }
}