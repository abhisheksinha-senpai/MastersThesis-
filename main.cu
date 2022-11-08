#include "Helper.cuh"
#include <ctime>
extern unsigned int SCR_WIDTH;
extern unsigned int SCR_HEIGHT;

int main(int argc, char* argv[])
{
    cudaDeviceReset();
    cudaSetDevice(0);
    getDeviceInfo();

    int NX = 64;
    int NY = 64;
    int NZ = 64;

    float Re_lattice = 10.0f;
    float viscosity =1.48e-5f;
    float spring_constant = 0.005f;

    float *rho, *ux, *uy, *uz;
    float *Fx_gpu, *Fy_gpu, *Fz_gpu;
    float *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;
    float *f1_gpu, *f2_gpu, *feq_gpu, *source_term_gpu;
    Vertex **nodeLists, **nodeData;
    int *vertex_size_per_mesh;

    glm::f32vec3 origin = glm::f32vec3(NX/2, NY/2, NZ/2);
    glm::f32vec3 mod_scale = glm::f32vec3(NX/8, NY/8, NZ/8);

    glm::f32vec3 dis_scale = glm::f32vec3(1.0f/16.0f, 1.0f/16.0f, 1.0f/16.0f);

    ResourceManager r_manager;
    Shader ourShader;
    Model ourModel;
    GLFWwindow* window;

    display_init(&window);
    model_init(r_manager, ourShader, ourModel, NX, NY, NZ, mod_scale, origin);
    domain_init(NX, NY, NZ, &rho, &ux, &uy, &uz);
    ParticleSystem myfluid(NX, NY, NZ, mod_scale, &rho[0]);

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
            streams);

    printf("Starting simulation .....\n");
    time_t cur_time1 = clock();
    time_t cur_time2 = clock();
    int KK = 0;
    while(!glfwWindowShouldClose(window))
    {
        // if(((float)(clock() - cur_time2))/CLOCKS_PER_SEC>1/5.0f)// && KK++<10)
        // {
            LB_simulate(Fx_gpu, Fy_gpu, Fz_gpu, 
                    f1_gpu, f2_gpu, feq_gpu, source_term_gpu, 
                    rho_gpu, ux_gpu, uy_gpu, uz_gpu, 
                    NX, NY, NZ, IBM_force_spread, IBM_advect_bound,
                    nodeLists, vertex_size_per_mesh, 128, num_mesh, streams);
            cur_time2 = clock();
        // }
        if(((float)(clock() - cur_time1))/CLOCKS_PER_SEC>1/30.0f)
        {
            display( rho, ux, uy, uz,
                    rho_gpu, Fx_gpu, Fy_gpu, Fz_gpu,
                    NX, NY, NZ, 
                    myfluid, mod_scale, origin, dis_scale,
                    &window, ourShader, ourModel, 
                    num_mesh, nodeLists, vertex_size_per_mesh, streams);
            // display( rho, ux, uy, uz,
            //         rho_gpu, ux_gpu, uy_gpu, uz_gpu,
            //         NX, NY, NZ, 
            //         myfluid, mod_scale, origin, dis_scale,
            //         &window, ourShader, ourModel);
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