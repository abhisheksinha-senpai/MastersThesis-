#include "Helper.cuh"

extern unsigned int SCR_WIDTH;
extern unsigned int SCR_HEIGHT;

extern float Ct;
extern float Cl;

extern Vertex **nodeLists;
extern int *vertex_size_per_mesh;
extern float *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;
extern float *Fx_gpu, *Fy_gpu, *Fz_gpu;

int main(int argc, char* argv[])
{
    cudaDeviceReset();
    cudaSetDevice(0);
    getDeviceInfo();

    float *rho, *ux, *uy, *uz;
    Vertex **nodeData;
    ResourceManager r_manager;
    Shader ourShader;
    Model ourModel;
    GLFWwindow* window;

    dim3 num_threads(32, 32, 1);
    int NX = 64;
    int NY = 64;
    int NZ = 16;
    glm::f32vec3 simDimention = glm::f32vec3(NX, NY, NZ);

    float Re_lattice = 100.0f;
    float viscosity =1.48e-5f;
    float spring_constant = 0.005f;

    glm::f32vec3 mod_scale = glm::f32vec3(8, 8, 8);
    glm::f32vec3 mod_origin = glm::f32vec3(0);
    int sc = max(NX, max(NY, NZ));
    glm::f32vec3 dis_scale = (glm::f32vec3(2.0f/sc, 2.0f/sc, 2.0f/sc));

    display_init(&window);
    model_init(r_manager, ourShader, ourModel, NX, NY, NZ, mod_scale, mod_origin);
    domain_init(simDimention, &rho, &ux, &uy, &uz, 2);
    ParticleSystem myfluid(NX, NY, NZ, mod_scale, &rho[0]);
    Geometry fluidDomain = Geometry(glm::f32vec3((float)NX/sc, (float)NY/sc, (float)NZ/sc));

    vertex_size_per_mesh = (int *)malloc(ourModel.meshes.size()*sizeof(int));
    nodeLists = (Vertex**)malloc(ourModel.meshes.size()*sizeof(Vertex *));
    nodeData = (Vertex**)malloc(ourModel.meshes.size()*sizeof(Vertex *));

    int num_mesh = ourModel.meshes.size();
    for(int i=0;i<num_mesh;i++)
    {
        nodeData[i] = ourModel.meshes[i].vertices.data();
        vertex_size_per_mesh[i] = ourModel.meshes[i].vertices.size();
    }

    float total_size_allocated = 0;
    total_size_allocated += LB_init(NX, NY, NZ, Re_lattice, viscosity, &rho, &ux, &uy, &uz, num_threads, 3);
    total_size_allocated += IBM_init(NX, NY, NZ, num_mesh, nodeData, spring_constant);
    
    Softbody monkey = Softbody(nodeData[0], vertex_size_per_mesh[0], ourModel.meshes[0].edges.data(), ourModel.meshes[0].edges.size(), 0.0f, 0.0f, Cl);
    checkCudaErrors(cudaMemcpy((void *)nodeLists[0], (void *)nodeData[0], vertex_size_per_mesh[0]*sizeof(Vertex), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    float byte_per_GB = powf(1024.0f, 3);
    printf("Total memory allocated in GPU: %f GB\n",total_size_allocated/byte_per_GB );
    printf("...............................................................................\n");
    printf("Starting simulation .....\n");


    time_t cur_time1 = clock();
    time_t cur_time2 = clock();
    time_t start_time = clock();
    int KK = 0;
    float time_elapsed = 0.0f;
    bool flag_FSI = false;
    while(!glfwWindowShouldClose(window))
    {
        if((((float)(clock() - cur_time2))/CLOCKS_PER_SEC>0.08f) && (((float)(clock() - start_time))/CLOCKS_PER_SEC>2.00f )&& KK++<20000)
        {
            float del_time = Ct;//((clock() - (float)cur_time2)/CLOCKS_PER_SEC);
            LB_simulate_RB(NX, NY, NZ, Ct, IBM_force_spread_RB, IBM_vel_spread_RB, num_threads, num_mesh, flag_FSI);

            // checkCudaErrors(cudaMemcpy((void *)nodeData[0], (void *)nodeLists[0], vertex_size_per_mesh[0]*sizeof(Vertex), cudaMemcpyDeviceToHost));
            // checkCudaErrors(cudaDeviceSynchronize());

            // for (int i = 0; i < monkey.numVerts; i++) 
            // {
            //     if (glm::length(monkey.verts[i].Force) > 100.0f)
            //     {
            //         printf(" \n\nForce = %f \n\n", glm::length(monkey.verts[i].Force));
            //         flag_FSI = true;
            //         break;
            //     }
            // }
            // for(int i=0;i<1;i++)
            // {
            //     monkey.preSolve(del_time/1.0f, glm::f32vec3(NX, NY, NZ), Ct, Cl);
            //     monkey.SolveEdges(del_time/1.0f);
            //     monkey.postSolve(del_time/1.0f);
            // }

            // checkCudaErrors(cudaMemcpy((void *)nodeLists[0], (void *)nodeData[0], vertex_size_per_mesh[0]*sizeof(Vertex), cudaMemcpyHostToDevice));
            // checkCudaErrors(cudaDeviceSynchronize());

            cur_time2 = clock();
            time_elapsed += Ct;
            
        }

        if(((float)(clock() - cur_time1))/CLOCKS_PER_SEC>1/30.0f)
        {
            preDisplay();
            displayDomain(fluidDomain);
            // displayModel(&window, ourShader, ourModel, dis_scale);
            displayFluid(rho, ux, uy, uz, rho_gpu, ux_gpu, uy_gpu, uz_gpu, NX, NY, NZ, myfluid, dis_scale);
            postDisplay(&window);
            cur_time1 = clock();
        }
    }
    
    IBM_cleanup(num_mesh);

    LB_cleanup();

    scene_cleanup(nodeLists, nodeData, vertex_size_per_mesh, rho, ux, uy, uz);

    return 0;
}