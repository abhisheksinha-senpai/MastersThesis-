#ifndef RM_H
#define RM_H

#include "Definitions.hpp"

class ResourceManager
{
    public:
    ResourceManager(){};
    ~ResourceManager(){printf("Resource Manager deleted\n");};
    void load_shader(std::string file_name, std::string shader_type, std::string &location);
};

#endif // !RM_H