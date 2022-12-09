#include "ImmersedBoundary.cuh"

__device__ int sim_domain[3];
__device__ float spring_constant;

Vertex **nodeLists;
int *vertex_size_per_mesh;

__device__ float calculate_kernel_phi(float x, int type)
{
    if(type == 1)
    {
        if(abs(x)<=1.0f)
            return 1-abs(x);
        else
            return 0.0f;
    }
    else if(type == 2)
    {
        if (abs(x)<0.5f)
            return (1.0f/3.0f)*(1.0f + sqrt(1.0f - 3.0f*x*x));
        else if(abs(x)>=0.5f*1.0f && abs(x)<3.0f/2.0f)
            return (1.0f/6.0f)*(5.0f-3.0f*abs(x)-sqrt(-2+6.0f*abs(x) - 3.0f*x*x));
        else
            return 0.0f;
    }
    else if(type == 3)
    {
        if (abs(x)<1.0f)
            return (1.0f/8.0f)*(3.0f - 2.0f*abs(x) + sqrt(1.0f + 4.0f*abs(x)-4.0f*x*x));
        else if(abs(x) >= 1.0f && abs(x)<2.0f)
            return (1.0f/8.0f)*(5.0f - 2.0f*abs(x) - sqrt(-7.0f + 12.0f*abs(x)-4.0f*x*x));
        else
            return 0.0f;
    }

    return 0.0f;
}

__global__ void spread_fluid_velocity(Vertex *nodeLists, int vertexList_size, int num_threads,
                                      float *ux_gpu, float *uy_gpu, float *uz_gpu, unsigned int *cell_type_gpu)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float phi = 0.0f;
    glm::f32vec3 fj = glm::f32vec3(0.0f);
    int coord = 0;
    float ux, uy, uz;
    for(int p=id;p<vertexList_size;p+=num_threads)
    {
        float X1 = nodeLists[p].Position.x;
        float Y1 = nodeLists[p].Position.y;
        float Z1 = nodeLists[p].Position.z;

        glm::f32vec3 vel = glm::f32vec3(0.0f);

        for(int i=-2;i<=2;i++)
        {
            for(int j=-2;j<=2;j++)
            {
                for(int k=-2;k<=2;k++)
                {
                    coord = gpu_scalar_index((int)(X1+(float)i), (int)(Y1+(float)j), (int)(Z1+(float)k), sim_domain);
                    if(cell_type_gpu[coord] & (INTERFACE|FLUID))
                    {
                        ux = ux_gpu[coord];
                        uy = uy_gpu[coord];
                        uz = uz_gpu[coord];
                        float deltaX = X1-(X1+(float)i);
                        float deltaY = Y1-(Y1+(float)j);
                        float deltaZ = Z1-(Z1+(float)k);
                        phi = calculate_kernel_phi(abs(deltaX), 3)*calculate_kernel_phi(abs(deltaY), 3)*calculate_kernel_phi(abs(deltaZ), 3);
                        vel.x += ux*phi;
                        vel.y += uy*phi;
                        vel.z += uz*phi;
                        // if(glm::length(glm::f32vec3(ux, uy, uz))>0)
                        //     printf(" (%f %f %f) ", ux, uy, uz);
                    }
                }
            }
        }
        nodeLists[p].Velocity = vel;
        // if(glm::length(vel)>0)
        //     printf(" (%f %f %f) ", vel.x, vel.y, vel.z);
        //nodeLists[p].Position += vel;
    }
}

__global__ void update_position_RB(Vertex *nodeLists, int vertexList_size, int num_threads, float Ct)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for(int p=id;p<vertexList_size;p+=num_threads)
    {
        nodeLists[p].Base_Pos = nodeLists[p].Base_Pos + nodeLists[p].Base_Vel;
        nodeLists[p].Position = nodeLists[p].Base_Pos;
    }
}

__global__ void update_velocity_RB(Vertex *nodeLists, int vertexList_size, int num_threads,
                                   glm::f32vec3 Velocity, float *ux_gpu, float *uy_gpu, float *uz_gpu, 
                                   float Ct, float Cl)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float Cu = Cl/Ct;
    glm::f32vec3 Base_vel = Velocity/Cu;
    //printf("%f %f %f   ", Base_vel.x, Base_vel.y, Base_vel.z);
    for(int p=id;p<vertexList_size;p+=num_threads)
        nodeLists[p].Base_Vel = Base_vel;
}

__global__ void calculate_force_spreading_per_mesh(Vertex *nodeLists, int vertexList_size, int num_threads,
                                                    float *Fx_gpu, float *Fy_gpu, float *Fz_gpu)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float phi = 0.0f;
    glm::f32vec3 fj = glm::f32vec3(0.0f);
    int coord = 0;
    for(int p=id;p<vertexList_size;p+=num_threads)
    {
        float X1 = nodeLists[p].Position.x;
        float Y1 = nodeLists[p].Position.y;
        float Z1 = nodeLists[p].Position.z;
        glm::f32vec3 len = nodeLists[p].Position - nodeLists[p].Base_Pos;
        fj = -spring_constant*len;
           
        for(int i=-2;i<=2;i++)
        {
            for(int j=-2;j<=2;j++)
            {
                for(int k=-2;k<=2;k++)
                {
                    coord = gpu_scalar_index((int)(X1+i), (int)(Y1+j), (int)(Z1+k), sim_domain);
                    float deltaX = X1-(X1+(float)i);
                    float deltaY = Y1-(Y1+(float)j);
                    float deltaZ = Z1-(Z1+(float)k);
                    phi = calculate_kernel_phi(abs(deltaX), 3)*calculate_kernel_phi(abs(deltaY), 3)*calculate_kernel_phi(abs(deltaZ), 3);
                    if(phi>0)
                    {
                        atomicAdd(&(Fx_gpu[coord]),fj.x*phi);
                        atomicAdd(&(Fy_gpu[coord]),fj.y*phi);
                        atomicAdd(&(Fz_gpu[coord]),fj.z*phi);
                    }
                }
            }
        }
    }
}

__global__ void calculate_force_spreading_RB_per_mesh(Vertex *nodeLists, int vertexList_size, int num_threads,
                                                      float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float Ct,
                                                      float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, float *count_loc, 
                                                      unsigned int *cell_type_gpu)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float phi = 0.0f;
    glm::f32vec3 fj = glm::f32vec3(0.0f);
    int coord = 0;
    int start = id;
    int sampling_factor = 1;

    for(int p=sampling_factor*start;p<vertexList_size;p+=sampling_factor*num_threads)
    {
        float X1 = nodeLists[p].Position.x;
        float Y1 = nodeLists[p].Position.y;
        float Z1 = nodeLists[p].Position.z;
        coord = gpu_scalar_index((int)(X1), (int)(Y1), (int)(Z1), sim_domain);
        fj = 2.0f*rho_gpu[coord]*( nodeLists[p].Base_Vel - nodeLists[p].Velocity)*nodeLists[p].Area*(2.0f/powf(sim_domain[0], 1.0f));

        for(int i=-2;i<=2;i++)
        {
            for(int j=-2;j<=2;j++)
            {
                for(int k=-2;k<=2;k++)
                {
                    coord = gpu_scalar_index((int)(X1+(float)i), (int)(Y1+(float)j), (int)(Z1+(float)k), sim_domain);
                    if(cell_type_gpu[coord] & (INTERFACE|FLUID))
                    {
                        float deltaX = X1-(float)((int)(X1+(float)i));
                        float deltaY = Y1-(float)((int)(Y1+(float)j));
                        float deltaZ = Z1-(float)((int)(Z1+(float)k));
                        phi = calculate_kernel_phi(abs(deltaX), 3)*calculate_kernel_phi(abs(deltaY), 3)*calculate_kernel_phi(abs(deltaZ), 3);
                        
                        if(phi>0)
                        {
                            atomicAdd(&(count_loc[coord]),1.0f);
                            {
                                // if(glm::length(fj)>0)
                                //     printf("(%f %f %f)", nodeLists[p].Area, rho_gpu[coord],nodeLists[p].Velocity  );
                                atomicAdd(&(Fx_gpu[coord]),fj.x*phi);
                                atomicAdd(&(Fy_gpu[coord]),fj.y*phi);
                                atomicAdd(&(Fz_gpu[coord]),fj.z*phi);
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void print_force(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, sim_domain);

    if(Fx_gpu[coord] >0 || Fy_gpu[coord]>0 || Fz_gpu[coord]>0)
        printf("( %f %f %f   %d %d %d) ",Fx_gpu[coord], Fy_gpu[coord], Fz_gpu[coord], idx, idy, idz);
}

__host__ float IBM_init(int NX, int NY, int NZ, int num_mesh, Vertex** nodeData, cudaStream_t *streams, float spring_const)
{
    float total_size_allocated = 0;
    for(int i=0;i<num_mesh;i++)
    {
        checkCudaErrors(cudaMalloc((void**)&(nodeLists[i]), vertex_size_per_mesh[i]*sizeof(Vertex)));
        total_size_allocated += vertex_size_per_mesh[i]*sizeof(Vertex);
    }

    for(int i=0;i<num_mesh;i++)
        checkCudaErrors(cudaMemcpyAsync(nodeLists[i], nodeData[i], vertex_size_per_mesh[i]*sizeof(Vertex), cudaMemcpyHostToDevice, streams[i]));
    checkCudaErrors(cudaDeviceSynchronize());

    float k = spring_const;
    checkCudaErrors(cudaMemcpyToSymbolAsync(spring_constant, &k, sizeof(float), 0, cudaMemcpyHostToDevice, streams[0]));
    int nu = NX;
    int nv = NY;
    int nw = NZ;

    checkCudaErrors(cudaMemcpyToSymbol(sim_domain, &nu, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(sim_domain, &nv, sizeof(int), sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(sim_domain, &nw, sizeof(int), 2*sizeof(int), cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpyToSymbolAsync(sim_domain, &nu, sizeof(int), 0, cudaMemcpyHostToDevice, streams[0]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(sim_domain, &nv, sizeof(int), sizeof(int), cudaMemcpyHostToDevice, streams[1]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(sim_domain, &nw, sizeof(int), 2*sizeof(int), cudaMemcpyHostToDevice, streams[2]));
    
    checkCudaErrors(cudaDeviceSynchronize());

    return total_size_allocated;
}

__host__ void IBM_force_spread(int num_threads, int num_mesh, cudaStream_t *streams)
{
    for(int i=0;i<num_mesh;i++)
        calculate_force_spreading_per_mesh<<<1,num_threads,0, streams[i]>>>(nodeLists[i], vertex_size_per_mesh[i], num_threads, Fx_gpu, Fy_gpu, Fz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void IBM_force_spread_RB(int num_threads, int num_mesh, float Ct, cudaStream_t *streams)
{
    for(int i=0;i<num_mesh;i++)
        calculate_force_spreading_RB_per_mesh<<<1,num_threads,0, streams[i]>>>(nodeLists[i], vertex_size_per_mesh[i],
                                                          num_threads, Fx_gpu, Fy_gpu, Fz_gpu, Ct, rho_gpu, ux_gpu, uy_gpu, uz_gpu, count_loc, 
                                                          cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void IBM_advect_bound(int num_threads, int num_mesh, cudaStream_t *streams)
{
    for(int i=0;i<num_mesh;i++)
        spread_fluid_velocity<<<1,num_threads,0, streams[i]>>>(nodeLists[i], vertex_size_per_mesh[i], num_threads, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void update_IB_params(int num_threads, int num_mesh, float Ct, float Cl, glm::f32vec3 Velocity_RB, cudaStream_t *streams)
{
    for(int i=0;i<num_mesh;i++)
    {
        update_velocity_RB<<<1,num_threads,0, streams[i]>>>(nodeLists[i], vertex_size_per_mesh[i], num_threads, Velocity_RB, ux_gpu, uy_gpu, uz_gpu, Ct, Cl);
        update_position_RB<<<1,num_threads,0, streams[i]>>>(nodeLists[i], vertex_size_per_mesh[i], num_threads, Ct);
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void IBM_cleanup(int num_mesh)
{
    for(int i=0;i<num_mesh;i++)
        cudaFree(nodeLists[i]);
    printf("Immersed Boundary object cleaned\n");
}