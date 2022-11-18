#include "Helper.cuh"
#include <ctime>
extern unsigned int SCR_WIDTH;
extern unsigned int SCR_HEIGHT;

extern float Ct;
extern float Cl;
float *count_loc;

int main(int argc, char* argv[])
{
    cudaDeviceReset();
    cudaSetDevice(0);
    getDeviceInfo();

    int NX = 64;
    int NY = 64;
    int NZ = 64;

    float Re_lattice = 1000.0f;
    float viscosity =1.48e-3f;
    float spring_constant = 0.005f;
    float tau_star = 0.55f;

    float *rho, *ux, *uy, *uz;
    float *Fx_gpu, *Fy_gpu, *Fz_gpu;
    float *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;
    float *f1_gpu, *f2_gpu, *feq_gpu, *source_term_gpu;
    Vertex **nodeLists, **nodeData;

    int *vertex_size_per_mesh;

    // glm::f32vec3 mod_origin = glm::f32vec3(NX/2, NY/2, NZ/2);
    // glm::f32vec3 mod_scale = glm::f32vec3(2, 2, 2);
    glm::f32vec3 mod_scale = glm::f32vec3(4, 4, 4);
    glm::f32vec3 mod_origin = glm::f32vec3(NX/2, NY/2, NZ/2);

    glm::f32vec3 dis_scale = glm::f32vec3(2.0f/NX, 2.0f/NY, 2.0f/NZ);
    ResourceManager r_manager;
    Shader ourShader;
    Model ourModel;
    GLFWwindow* window;

    display_init(&window);
    model_init(r_manager, ourShader, ourModel, NX, NY, NZ, mod_scale, mod_origin);
    domain_init(NX, NY, NZ, &rho, &ux, &uy, &uz);
    ParticleSystem myfluid(NX, NY, NZ, mod_scale, &rho[0]);
    Geometry fluidDomain = Geometry(1.0f);

    vertex_size_per_mesh = (int *)malloc(ourModel.meshes.size()*sizeof(int));
    nodeLists = (Vertex**)malloc(ourModel.meshes.size()*sizeof(Vertex *));
    nodeData = (Vertex**)malloc(ourModel.meshes.size()*sizeof(Vertex *));

    int num_mesh = ourModel.meshes.size();
    cudaStream_t streams[num_mesh];
    for(int i=0;i<num_mesh;i++)
    {
        nodeData[i] = ourModel.meshes[i].vertices.data();
        vertex_size_per_mesh[i] = ourModel.meshes[i].vertices.size();
        cudaStreamCreate(&streams[i]);
    }

    IBM_init(NX, NY, NZ,
             &Fx_gpu, &Fy_gpu, &Fz_gpu,
             num_mesh, nodeLists, 
             vertex_size_per_mesh, nodeData, streams, spring_constant);

    LB_init(NX, NY, NZ, Re_lattice, viscosity,
            &f1_gpu, &f2_gpu, &feq_gpu, &source_term_gpu, 
            &rho_gpu, &ux_gpu, &uy_gpu, &uz_gpu, 
            &rho, &ux, &uy, &uz, 
            &Fx_gpu, &Fy_gpu, &Fz_gpu,
            streams, tau_star);

    printf("Starting simulation .....\n");
    time_t cur_time1 = clock();
    time_t cur_time2 = clock();
    int KK = 0;
    glm::f32vec3 Velocity_RB;
    float delta_angle = 0;
    float current_angle = 0;
    float angular_vel = M_PI/6.0f;
    float Uc = (Re_lattice*viscosity/2.0f);
    printf("Characteristic velocity %f\n", Uc);
    while(!glfwWindowShouldClose(window))
    {
        if(((float)(clock() - cur_time2))/CLOCKS_PER_SEC>1/30.0f)// && KK++<50)
        // if(KK++<40)
        {
            delta_angle = ((clock() - (float)cur_time2)/CLOCKS_PER_SEC);
            current_angle += delta_angle;
            if(current_angle>=2*M_PI)
                current_angle = -2.0f*M_PI;
            Velocity_RB = (Uc)*glm::f32vec3(angular_vel*cosf(angular_vel*current_angle), angular_vel*sinf(angular_vel*current_angle), angular_vel*sinf(angular_vel*current_angle));
            update_IB_params(nodeLists, vertex_size_per_mesh, 128, num_mesh, Ct, Velocity_RB, ux_gpu, uy_gpu,uz_gpu, streams);
            // LB_simulate(Fx_gpu, Fy_gpu, Fz_gpu, 
            //         f1_gpu, f2_gpu, feq_gpu, source_term_gpu, 
            //         rho_gpu, ux_gpu, uy_gpu, uz_gpu, 
            //         NX, NY, NZ, IBM_force_spread, IBM_advect_bound,
            //         nodeLists, vertex_size_per_mesh, 128, num_mesh, streams);

            LB_simulate_RB(Fx_gpu, Fy_gpu, Fz_gpu, 
                        f1_gpu, f2_gpu, feq_gpu, source_term_gpu, 
                        rho_gpu, ux_gpu, uy_gpu, uz_gpu, 
                        NX, NY, NZ, Ct, IBM_force_spread_RB, IBM_advect_bound,
                        nodeLists, vertex_size_per_mesh, 128, num_mesh, streams);

           

            cur_time2 = clock();
            
        }
        if(((float)(clock() - cur_time1))/CLOCKS_PER_SEC>1/30.0f)
        {
            // display( rho, ux, uy, uz,
            //         rho_gpu, count_loc, count_loc, count_loc,
            //         NX, NY, NZ, 
            //         myfluid, mod_scale, dis_scale,
            //         &window, ourShader, ourModel, fluidDomain,
            //         num_mesh, nodeLists, vertex_size_per_mesh, streams);
            // display( rho, ux, uy, uz,
            //     rho_gpu, Fx_gpu, Fy_gpu, Fz_gpu,
            //     NX, NY, NZ, 
            //     myfluid, mod_scale, dis_scale,
            //     &window, ourShader, ourModel, fluidDomain,
            //     num_mesh, nodeLists, vertex_size_per_mesh, streams);
            display( rho, ux, uy, uz,
                    rho_gpu, ux_gpu, uy_gpu, uz_gpu,
                    NX, NY, NZ, 
                    myfluid, mod_scale, dis_scale,
                    &window, ourShader, ourModel, fluidDomain, 
                    num_mesh, nodeLists, vertex_size_per_mesh, streams);
            cur_time1 = clock();
        }
       
    }
    
    IBM_cleanup(Fx_gpu, Fy_gpu, Fz_gpu, num_mesh, nodeLists);

    LB_cleanup(f1_gpu, f2_gpu, feq_gpu, source_term_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);

    scene_cleanup(nodeLists, nodeData, vertex_size_per_mesh, rho, ux, uy, uz);

    for(int i=0;i<num_mesh; i++)
        cudaStreamDestroy(streams[i]);

    return 0;
}