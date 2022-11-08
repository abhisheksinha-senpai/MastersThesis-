#include "Shader.hpp"

void Shader::use()
{
    glUseProgram(this->shader_pgm);
}

void Shader::create_vs_shader(const char* src_ver)
{
    try
    {
        this->src_ver_id = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(src_ver_id, 1, &src_ver, NULL);
        glCompileShader(src_ver_id);
        this->check_errors(src_ver_id, "VERTEX");
    }
    catch(std::exception e)
    {
        std::cout<<"Unable to create VERTEX shader"<<"\n";
        throw;
    }
}

void Shader::create_fs_shader(const char* src_frag)
{
    try
    {
        this->src_frag_id = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(src_frag_id, 1, &src_frag, NULL);
        glCompileShader(src_frag_id);
        check_errors(src_frag_id, "FRAGMENT");
    }
    catch(std::exception e)
    {
        std::cout<<"Unable to create FRAGMENT shader"<<"\n";
        throw;
    }
}

void Shader::compile()
{
    try
    {
    // shader program
        this->shader_pgm = glCreateProgram();
        glAttachShader(this->shader_pgm, src_ver_id);
        glAttachShader(this->shader_pgm, src_frag_id);

        glLinkProgram(this->shader_pgm);
        check_errors(this->shader_pgm, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(src_ver_id);
        glDeleteShader(src_frag_id);
    }
    catch(std::exception e)
    {
        std::cout<<"Unable to compile shader profram"<<"\n";
        throw;
    }
}

void Shader::check_errors(unsigned int object, std::string type)
{
    int success;
    char infoLog[1024];
    if (type != "PROGRAM")
    {
        glGetShaderiv(object, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(object, 1024, NULL, infoLog);
            
            std::cout << "| ERROR::SHADER: Compile-time error: Type: " << type << "\n";
            {throw std::runtime_error(std::string(infoLog));}
        }
    }
    else
    {
        glGetProgramiv(object, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(object, 1024, NULL, infoLog);
            std::cout << "| ERROR::Shader: Link-time error: Type: " << type << "\n";
            {throw std::runtime_error(std::string(infoLog));}
        }
    }
}

int Shader::get_shader_pgm()
{
    return this->shader_pgm;
}