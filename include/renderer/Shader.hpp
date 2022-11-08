#ifndef SHADER_H
#define SHADER_H

#include "Definitions.hpp"

class Shader
{
    private:
    unsigned int shader_pgm;
    unsigned int src_ver_id;
    unsigned int src_frag_id;
    public:
    std::string vertex_shader;
    std::string fragment_shader;
    Shader(){};
    void create_vs_shader(const char* src_ver);
    void create_fs_shader(const char* src_frag);
    void compile();
    void use();
    void check_errors(unsigned int object, std::string type);
    int get_shader_pgm();
    ~Shader(){};
};

#endif