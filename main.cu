#include "Helper.cuh"

extern unsigned int SCR_WIDTH;
extern unsigned int SCR_HEIGHT;

extern float Ct;
extern float Cl;

extern Vertex **nodeLists;
extern int *vertex_size_per_mesh;
extern float *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;

int main(int argc, char* argv[])
{
    cudaDeviceReset();
    cudaSetDevice(0);
    getDeviceInfo();

    int NX = 64;
    int NY = 64;
    int NZ = 64;

    float Re_lattice = 10000.0f;
    float viscosity =1.48e-3f;
    float spring_constant = 0.005f;

    float *rho, *ux, *uy, *uz;
    Vertex **nodeData;

    // glm::f32vec3 mod_origin = glm::f32vec3(NX/2, NY/2, NZ/2);
    // glm::f32vec3 mod_scale = glm::f32vec3(2, 2, 2);
    glm::f32vec3 mod_scale = glm::f32vec3(16, 16, 16);
    glm::f32vec3 mod_origin = glm::f32vec3(NX/4, 8, NZ/4);

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


    float total_size_allocated = 0;
    total_size_allocated += LB_init(NX, NY, NZ, Re_lattice, viscosity, &rho, &ux, &uy, &uz, streams);
    total_size_allocated+= IBM_init(NX, NY, NZ, num_mesh, nodeData, streams, spring_constant);
    float byte_per_GB = powf(1024.0f, 3);
    float Uc = (Re_lattice*viscosity/2.0f);
    printf("Characteristic velocity %f\n", Uc);
    printf("Total memory allocated in GPU: %f GB\n",total_size_allocated/byte_per_GB );
    printf("...............................................................................\n");
    printf("Starting simulation .....\n");
    time_t cur_time1 = clock();
    time_t cur_time2 = clock();
    time_t start_time = clock();
    int KK = 0;
    glm::f32vec3 Velocity_RB;
    float delta_angle = 0;
    float current_angle = 0;
    float angular_vel = M_PI/6.0f;
    printf("\t Constants: %f %f \n", Ct, Cl);
    float time_elapsed = 0.0f;
    while(!glfwWindowShouldClose(window))
    {
        if((((float)(clock() - cur_time2))/CLOCKS_PER_SEC>0.02f) && (((float)(clock() - start_time))/CLOCKS_PER_SEC>5.0f )&& KK++<1000)
        //if(KK++<2)
        {
            float del_time = ((clock() - (float)cur_time2)/CLOCKS_PER_SEC);
            printf("Current Simulation time: %f \n", time_elapsed );
            delta_angle = del_time;
            current_angle += delta_angle;
            if(current_angle>=2*M_PI)
                current_angle = -2.0f*M_PI;
            
            Velocity_RB = glm::f32vec3(0.0f);//(Uc)*glm::f32vec3(angular_vel*cosf(angular_vel*current_angle), 0, angular_vel*sinf(angular_vel*current_angle));
            update_IB_params(128, num_mesh, Ct, Cl, Velocity_RB, streams);
            LB_simulate_RB(NX, NY, NZ, Ct, IBM_force_spread_RB, IBM_advect_bound, 128, num_mesh, streams);
            cur_time2 = clock();
            time_elapsed += Ct;
            
        }
        if(((float)(clock() - cur_time1))/CLOCKS_PER_SEC>1/30.0f)
        {
            display( rho, ux, uy, uz,
                    rho_gpu, ux_gpu, uy_gpu, uz_gpu,
                    NX, NY, NZ, 
                    myfluid, mod_scale, dis_scale,
                    &window, ourShader, ourModel, fluidDomain, 
                    num_mesh, nodeLists, vertex_size_per_mesh, streams);
            cur_time1 = clock();
        }
       
    }
    
    IBM_cleanup(num_mesh);

    LB_cleanup();

    scene_cleanup(nodeLists, nodeData, vertex_size_per_mesh, rho, ux, uy, uz);

    for(int i=0;i<num_mesh; i++)
        cudaStreamDestroy(streams[i]);

    return 0;
}