#include "ParticleSystem.hpp"

ParticleSystem::ParticleSystem(int NX, int NY, int NZ, glm::f32vec3 model_scale, float *rho)
{
    ResourceManager r_manager = ResourceManager();
    r_manager.load_shader("resources/shaders/vertex/domain_shader.vs", "VERTEX", fluidShader.vertex_shader);
    r_manager.load_shader("resources/shaders/fragment/domain_shader.fs", "FRAGMENT", fluidShader.fragment_shader);
    fluidShader.create_vs_shader(fluidShader.vertex_shader.c_str());
    fluidShader.create_fs_shader(fluidShader.fragment_shader.c_str());
    fluidShader.compile();
    printf("Fluid shader loaded.....\n");
    for(int i=0;i<NX;i++)
    {
        for(int j=0;j<NY;j++)
        {
            for(int k=0;k<NZ;k++)
            {
                //glm::f32vec3 position = glm::f32vec3((float)i*(model_scale.x/NX)-1.0f,(float)j*(model_scale.y/NY)-1.0f,(float)k*(model_scale.z/NZ)-1.0f);
                glm::f32vec3 position = glm::f32vec3((float)i,(float)j,(float)k);
                int loc = i+j*NX+k*NX*NY;
                glm::f32vec4 color = glm::vec4(1.0f, 1.0f, 1.0f, rho[loc]);
                Drop d{position, color};
                fluid.push_back(d);
            }
        }
    }
    glGenVertexArrays(1, &VAO_1);
    glGenBuffers(1, &VBO_1);
    printf("Fluids initiliazed.....\n");
}

void ParticleSystem::update_particles(int NX, int NY, int NZ, float *rho, float *ux, float *uy, float *uz, glm::f32vec3 model_scale)
{
    for(int i=0;i<fluid.size();i++)
    {
        float x = (fluid[i].Position.x);
        float y = (fluid[i].Position.y);
        float z = (fluid[i].Position.z);
        int loc = (int)(x+y*NX+z*NX*NY);
        float val = rho[loc]/((int)(abs(rho[loc])));
        fluid[i].Color = glm::vec4( 100*ux[loc],100*uy[loc],100*uz[loc], rho[loc]>0.5f);
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
    glBindVertexArray(0);
    glPointSize(1);
    // glDeleteBuffers(1, &VBO_1);
    // glDeleteVertexArrays(1, &VAO_1);
}

ParticleSystem::~ParticleSystem()
{
    glDeleteBuffers(1, &VBO_1);
    glDeleteVertexArrays(1, &VAO_1);
}