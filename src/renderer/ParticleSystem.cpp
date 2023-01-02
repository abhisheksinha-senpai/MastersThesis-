#include "ParticleSystem.hpp"

unsigned int FLUID = (unsigned int)(1 << 0);
unsigned int  INTERFACE  = (unsigned int)(1 << 1);
unsigned int  EMPTY =  (unsigned int)(1 << 2);
unsigned int  OBSTACLE =  (unsigned int)(1 << 3);
unsigned int  NO_FLUID_NEIGH =  (unsigned int)(1 << 4);
unsigned int  NO_EMPTY_NEIGH =  (unsigned int)(1 << 5);
unsigned int  NO_IFACE_NEIGH =  (unsigned int)(1 << 6);
unsigned int  IF_TO_FLUID = ((unsigned int)(1 << 1)|(unsigned int)(1 << 0));
unsigned int  IF_TO_EMPTY = ((unsigned int)(1 << 1)|(unsigned int)(1 << 2));
unsigned int  EMPTY_TO_IF = (unsigned int)(1 << 0)|((unsigned int)(1 << 1)|(unsigned int)(1 << 2));

ParticleSystem::ParticleSystem(int NX, int NY, int NZ, glm::f32vec3 model_scale, float *mass)
{
    ResourceManager r_manager = ResourceManager();
    r_manager.load_shader("resources/shaders/vertex/domain_shader.vs", "VERTEX", fluidShader.vertex_shader);
    r_manager.load_shader("resources/shaders/fragment/domain_shader.fs", "FRAGMENT", fluidShader.fragment_shader);
    fluidShader.create_vs_shader(fluidShader.vertex_shader.c_str());
    fluidShader.create_fs_shader(fluidShader.fragment_shader.c_str());
    fluidShader.compile();
    printf("Fluid shader loaded.....\n");
    for(int i=1;i<NX-1;i++)
    {
        for(int j=1;j<NY-1;j++)
        {
            for(int k=1;k<NZ-1;k++)
            {
                // marching_cube(i, j, k, NX, NY, NZ, mass, fluid);
                glm::f32vec3 position = glm::f32vec3((float)i,(float)j,(float)k);
                int loc = i+j*NX+k*NX*NY;
                glm::f32vec4 color = glm::vec4(1.0f, 1.0f, 1.0f, mass[loc]);
                Drop d{position, color};
                fluid.push_back(d);
            }
        }
    }
    glGenVertexArrays(1, &VAO_1);
    glGenBuffers(1, &VBO_1);
    printf("Fluids initiliazed.....\n");
}

void ParticleSystem::update_particles(int NX, int NY, int NZ, float *mass, float *ux, float *uy, float *uz, glm::f32vec3 model_scale)
{
    // std::vector<Drop>().swap(fluid);
    // for(int i=1;i<NX-1;i++)
    // {
    //     for(int j=1;j<NY-1;j++)
    //     {
    //         for(int k=1;k<NZ-1;k++)
    //         {
    //             marching_cube(i, j, k, NX, NY, NZ, mass, fluid);
    //         }
    //     }
    // }
    for(int i=0;i<fluid.size();i++)
    {
        float x = (fluid[i].Position.x);
        float y = (fluid[i].Position.y);
        float z = (fluid[i].Position.z);
        int loc = (int)(x+y*NX+z*NX*NY);
        float vel = glm::length(glm::f32vec3(ux[loc], uy[loc], uz[loc]));
        fluid[i].Color = glm::vec4(((int)mass[loc] == INTERFACE), ((int)mass[loc] == FLUID), ((int)mass[loc] == EMPTY), (((int)mass[loc] & (INTERFACE|FLUID))));
        // fluid[i].Color = glm::vec4(30.0f*abs(ux[loc]), 30.0f*abs(uy[loc]), 30.0f*abs(uz[loc]), ((mass[loc])));
        fluid[i].Color = glm::vec4(mass[loc]);
        // fluid[i].Color = glm::vec4(ux[loc]/uy[loc], ux[loc], uy[loc], ((int)mass[loc] & (FLUID|INTERFACE)));
        // if((int)mass[loc] & INTERFACE && ((int)y == 3*NY/4) && ((int)x == 4*NX/8))
        //     fluid[i].Color = glm::vec4(0,ux[loc]/uy[loc], 0, ((int)mass[loc] & (FLUID|INTERFACE)));
    }
} 

void ParticleSystem::draw_particles(int SCR_WIDTH, int SCR_HEIGHT, glm::vec3 cameraPos, glm::vec3 cameraFront, glm::vec3 cameraUp, glm::f32vec3 dis_scale)
{
    fluidShader.use();
    glm::mat4 model = glm::mat4(1);
    model = glm::translate(model, glm::f32vec3(-1, -1, -1));
    model = glm::scale(model, dis_scale);
    
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos+cameraFront, cameraUp); 
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);
    glUniformMatrix4fv(glGetUniformLocation(fluidShader.get_shader_pgm(), "view_domain"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(fluidShader.get_shader_pgm(), "model_domain"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(fluidShader.get_shader_pgm(), "projection_domain"), 1, GL_FALSE, glm::value_ptr(proj));
    
    glBindVertexArray(VAO_1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_1);
    glBufferData(GL_ARRAY_BUFFER, fluid.size()*sizeof(Drop), &fluid[0], GL_DYNAMIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Drop), (void*)0);
    // vertex texture coords
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Drop), (void*)offsetof(Drop, Color));
    glPointSize(5);
    glDrawArrays(GL_POINTS , 0, fluid.size());
    // glDrawArrays(GL_TRIANGLES , 0, fluid.size());
    glPointSize(1);
    glBindVertexArray(0);
}

ParticleSystem::~ParticleSystem()
{
    glDeleteBuffers(1, &VBO_1);
    glDeleteVertexArrays(1, &VAO_1);
}