#include "LatticeBoltzmann.cuh"
#define GRAV_CONST 0.001f

__device__ float viscosity;
__device__ float Re;
__device__ float tau_no_dim;
__device__ float delT=1.0f;
__device__ float cs_inv_sq = 3.0f;
__device__ float Ct;

__device__ float w[19] = {1.0f/3.0f, 1.0f/18.0f,  1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

__device__ int cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0};
__device__ int cy[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1};
__device__ int cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1};

__device__ int lb_sim_domain[3];

__device__ float max_ux = -9999.0f;
__device__ float max_uy = -9999.0f;
__device__ float max_uz = -9999.0f;
__device__ float max_rho = -9999.0f;
__device__ float max_Fx = -9999.0f;
__device__ float max_Fy = -9999.0f;
__device__ float max_Fz = -9999.0f;

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void LB_compute_local_params(float *f1_gpu, float *Fx_gpu, float *Fy_gpu, float* Fz_gpu, 
                                        float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float lat_rho=0.000f, lat_rho_ux=0.000f, lat_rho_uy=0.000f, lat_rho_uz=0.000f;
    int coord = 0;
    float f_val = 0.000f;
    for(int i=0;i<19;i++)
    {
        f_val = f1_gpu[gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain)];
        lat_rho += f_val;
        lat_rho_ux += f_val*cx[i];
        lat_rho_uy += f_val*cy[i];
        lat_rho_uz += f_val*cz[i];
    }
    coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    rho_gpu[coord] = lat_rho;
    float rho_inv = (lat_rho == 0.0f?0.0f:(1.0f/lat_rho));
    ux_gpu[coord] = (lat_rho_ux + 0.5f * delT * Fx_gpu[coord]) * rho_inv;
    uy_gpu[coord] = (lat_rho_uy + 0.5f * delT * Fy_gpu[coord]) * rho_inv;
    uz_gpu[coord] = (lat_rho_uz + 0.5f * delT * Fz_gpu[coord]) * rho_inv;
}

__global__ void LB_compute_equi_distribution(float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu,
                                             float* feq_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float lat_rho, lat_ux, lat_uy, lat_uz;
    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    lat_rho =  rho_gpu[coord];
    lat_ux = ux_gpu[coord];
    lat_uy = uy_gpu[coord];
    lat_uz = uz_gpu[coord];

    float u_dot_c, u_dot_u = lat_ux * lat_ux + lat_uy * lat_uy + lat_uz * lat_uz;
    for(int i=0;i<19;i++)
    {
        u_dot_c= cx[i] * lat_ux + cy[i] * lat_uy + cz[i] * lat_uz;
        feq_gpu[gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain)] = w[i]*lat_rho*(1+cs_inv_sq*(u_dot_c +0.5*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5*u_dot_u));
    }
}

__global__ void LB_compute_source_term(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *source_term_gpu,
                                        float *ux_gpu, float *uy_gpu, float *uz_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float ci_dot_u = 0.0f, ci_dot_F = 0.0f;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    float lat_ux = ux_gpu[coord];
    float lat_uy = uy_gpu[coord];
    float lat_uz = uz_gpu[coord];

    float coef = (1-0.5*delT/tau_no_dim)*delT;
    float Fi = 0;
    float u_dot_F = lat_ux*Fx_gpu[coord] + lat_uy*Fy_gpu[coord] + lat_uz*Fz_gpu[coord];
    for(int i=0;i<19;i++)
    {
        ci_dot_F = cx[i]*Fx_gpu[coord] + cy[i]*Fy_gpu[coord] + cz[i]*Fz_gpu[coord];
        ci_dot_u = cx[i]*lat_ux + cy[i]*lat_uy + cz[i]*lat_uz;
        Fi = cs_inv_sq * ((ci_dot_F - u_dot_F) + cs_inv_sq*(ci_dot_F*ci_dot_u));
        source_term_gpu[gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain)] = w[i] * coef * Fi;
    }
}

__global__ void LB_equi_Initialization( float *f1_gpu, float *feq_gpu, 
                                        float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, 
                                        float *Fx_gpu, float *Fy_gpu, float *Fz_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float lat_rho, lat_ux, lat_uy, lat_uz;
    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    lat_rho =  rho_gpu[coord];
    float rho_inv = (lat_rho == 0.0f)?0.0f:(1.0f/lat_rho);
    lat_ux = ux_gpu[coord] - 0.5f*rho_inv*Fx_gpu[coord]*delT;
    lat_uy = uy_gpu[coord] - 0.5f*rho_inv*Fy_gpu[coord]*delT;
    lat_uz = uz_gpu[coord] - 0.5f*rho_inv*Fz_gpu[coord]*delT;
    ux_gpu[coord] = lat_ux;
    uy_gpu[coord] = lat_uy;
    uz_gpu[coord] = lat_uz;

    float u_dot_c, u_dot_u = lat_ux * lat_ux + lat_uy * lat_uy + lat_uz * lat_uz;
    float val = 0;
    for(int i=0;i<19;i++)
    {
        coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
        u_dot_c= cx[i] * lat_ux + cy[i] * lat_uy + cz[i] * lat_uz;
        val = w[i]*lat_rho*(1.0f + cs_inv_sq*(u_dot_c +0.5*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5*u_dot_u));
        feq_gpu[coord] = val;
        f1_gpu[coord] = val;
    }
}

__global__ void LB_collide(float *f1_gpu, float* f2_gpu, float *feq_gpu, float *source_term_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord;
    float omega, source;
    float tau_bar = tau_no_dim +delT/2.0f;
    float tau_inv = (-1.0f/tau_bar);
    for(int i =0;i<19;i++)
    {
        coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
        omega = tau_inv*(f1_gpu[coord] - feq_gpu[coord]);
        source = source_term_gpu[coord];
        f2_gpu[coord] = f1_gpu[coord]+(omega + source)*delT;
    }
}

__global__ void LB_enforce_boundary_wall(float *f1_gpu, float *f2_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    if(idy==0)
    {
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 4, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 3, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 12, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 11, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 18, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 17, lb_sim_domain)];
        if(idz == NZ-1)
        {
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 5, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 6, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 11, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 12, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 18, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 17, lb_sim_domain)];
            if(idx == 0)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)];
            }
            else if(idx == NX-1)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)];
            }
        }
        else if(idz == 0)
        {
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 6, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 5, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 17, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 18, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 12, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 11, lb_sim_domain)];

            if(idx == 0)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)];
            }
            else if(idx == NX-1)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)];
            }
        }
    }
    else if(idy == NY-1)
    {
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 3, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 4, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 11, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 12, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 17, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 18, lb_sim_domain)];
         if(idz == NZ-1)
        {
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 5, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 6, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 11, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 12, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 18, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 17, lb_sim_domain)];
            if(idx == 0)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)];
            }
            else if(idx == NX-1)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)];
            }
        }
        else if(idz == 0)
        {
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 6, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 5, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 17, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 18, lb_sim_domain)];
            f1_gpu[gpu_fieldn_index(idx, idy, idz, 12, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 11, lb_sim_domain)];

            if(idx == 0)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)];
            }
            else if(idx == NX-1)
            {
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)];
                f1_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)];
            }
        }
    }
}

__global__ void LB_stream(float *f1_gpu, float *f2_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;
    // int coord;

    bool X_left_cond = (idx != 0);
    bool X_right_cond = (idx != lb_sim_domain[0]-1);
    bool Y_left_cond = (idy != 0);
    bool Y_right_cond = (idy != lb_sim_domain[1]-1);
    bool Z_left_cond = (idz != 0);
    bool Z_right_cond = (idz != lb_sim_domain[2]-1);

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    unsigned int xp1 = (idx+1)%NX; //front in X direction
    unsigned int yp1 = (idy+1)%NY; //front in Y direction
    unsigned int zp1 = (idz+1)%NZ; //front in Z direction
    unsigned int xm1 = (NX+idx-1)%NX; //back in X direction
    unsigned int ym1 = (NY+idy-1)%NY; //back in Y direction
    unsigned int zm1 = (NZ+idz-1)%NZ; //back in Z direction
    //if( X_left_cond && X_right_cond && Y_left_cond && Y_right_cond && Z_left_cond && Z_right_cond)
    {
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 1, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xm1, idy, idz, 1, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 2, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xp1, idy, idz, 2, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 3, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, ym1, idz, 3, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 4, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, yp1, idz, 4, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 5, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, zm1, 5, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 6, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, zp1, 6, lb_sim_domain)];

        f1_gpu[gpu_fieldn_index(idx, idy, idz, 7, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xm1, ym1, idz, 7, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 8, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xp1, yp1, idz, 8, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 9, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xm1, idy, zm1, 9, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 10, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xp1, idy, zp1, 10, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 11, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, ym1, zm1, 11, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 12, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, yp1, zp1, 12, lb_sim_domain)];

        f1_gpu[gpu_fieldn_index(idx, idy, idz, 13, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xm1, yp1, idz, 13, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 14, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xp1, ym1, idz, 14, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 15, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xm1, idy, zp1, 15, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 16, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(xp1, idy, zm1, 16, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 17, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, ym1, zp1, 17, lb_sim_domain)];
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 18, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, yp1, zm1, 18, lb_sim_domain)];
    }
    // for(int i =0;i<19;i++)
    // {
    //     coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
    //     f1_gpu[coord] = f2_gpu[coord];
    // }
}

__global__ void check_max_params()
{
    printf("Max values of params = %f %f %f %f\t%f %f %f\n", max_ux, max_uy, max_uz, max_rho, max_Fx, max_Fy, max_Fz);
}

__global__ void update_max_params(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu,
                                  float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    atomicMax(&max_ux, abs(ux_gpu[coord]));
    atomicMax(&max_uy, abs(uy_gpu[coord]));
    atomicMax(&max_uz, abs(uz_gpu[coord]));
    atomicMax(&max_rho, abs(rho_gpu[coord]));
    atomicMax(&max_Fx, abs(Fx_gpu[coord]));
    atomicMax(&max_Fy, abs(Fy_gpu[coord]));
    atomicMax(&max_Fz, abs(Fz_gpu[coord]));
}

__global__ void LB_add_gravity(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *rho_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float domain_size = 2.0f;
    float Cl = domain_size/max(max(lb_sim_domain[0], lb_sim_domain[1]), lb_sim_domain[2]);
    int sidx = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    atomicAdd(&Fy_gpu[sidx], -1.0f* GRAV_CONST*Ct*Ct/Cl);
}

__host__ void LB_clear_Forces(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *source_term_gpu, 
    cudaStream_t *streams, int NX, int NY, int NZ)
{
    int total_lattice_points = NX*NY*NZ;
    checkCudaErrors(cudaMemsetAsync ((void*)(Fx_gpu), 0, total_lattice_points*sizeof(float), streams[0]));
    checkCudaErrors(cudaMemsetAsync ((void*)(Fy_gpu), 0, total_lattice_points*sizeof(float), streams[1]));
    checkCudaErrors(cudaMemsetAsync ((void*)(Fz_gpu), 0, total_lattice_points*sizeof(float), streams[2]));
    checkCudaErrors(cudaMemsetAsync ((void*)(source_term_gpu), 0, 19*total_lattice_points*sizeof(float), streams[3]));
}

__host__ void LB_reset_max(cudaStream_t *streams)
{
    float val_1 = -9999.0f;
    checkCudaErrors(cudaMemcpyToSymbolAsync(max_rho, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[0]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(max_ux, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[1]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(max_uy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[2]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(max_uz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[3]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(max_Fx, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[4]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(max_Fy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[5]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(max_Fz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[6]));
}

__host__ void LB_compute_sim_param(int NX, int NY, int NZ, float viscosity, float Re)
{
    float domain_size = 2.0;
    float cs_inv_sq = 3.0f;
    float cs = 1.0f/sqrt(cs_inv_sq);
    float lat_l_no_dim = max(max(NX, NY), NZ);
    float delX = domain_size/lat_l_no_dim;

    float non_dim_tau = 0.55f;
    float deltaT = (1/cs_inv_sq)*(non_dim_tau-0.5)*((delX*delX)/viscosity);
    checkCudaErrors(cudaMemcpyToSymbol(tau_no_dim, &non_dim_tau, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(Ct, &deltaT, sizeof(float), 0, cudaMemcpyHostToDevice));
    printf("..................Simulation parameters are as follows.........................\n");
    printf("Lattice Reynolds number : %f\n", Re);
    printf("non dimentional del T :%f\n", deltaT);
    printf("non dimentional tau :%f\n", non_dim_tau);
    printf("non dimentional gravity %f\n", GRAV_CONST*deltaT*deltaT/delX);
    printf("Velocity conversion factor %f\n", delX/deltaT);
    printf("...............................................................................\n");
}

__host__ float LB_allocate_memory(int NX, int NY, int NZ,
                                float **f1_gpu, float **f2_gpu, float **feq_gpu, float **source_term_gpu, 
                                float **rho_gpu, float **ux_gpu, float **uy_gpu, float **uz_gpu)
{
    float total_size_allocated = 0;
    int total_lattice_points = NX*NY*NZ;
    unsigned int mem_size_ndir  = sizeof(float)*total_lattice_points*(19);
    unsigned int mem_size_scalar = sizeof(float)*total_lattice_points;
    checkCudaErrors(cudaMalloc((void**)f1_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;
    checkCudaErrors(cudaMalloc((void**)f2_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;
    checkCudaErrors(cudaMalloc((void**)feq_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;
    checkCudaErrors(cudaMalloc((void**)source_term_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;
    checkCudaErrors(cudaMalloc((void**)rho_gpu,mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)ux_gpu,mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)uy_gpu,mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)uz_gpu,mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaDeviceSynchronize());
    return total_size_allocated;
}

__host__ void LB_init_symbols(int NX, int NY, int NZ, float Reynolds, float mu, cudaStream_t *streams)
{
    float Reynold_number = Reynolds;
    float vis = mu;
    int nu = NX;
    int nv = NY;
    int nw = NZ;
    checkCudaErrors(cudaMemcpyToSymbolAsync(lb_sim_domain, &nu, sizeof(int), 0, cudaMemcpyHostToDevice, streams[0]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(lb_sim_domain, &nv, sizeof(int), sizeof(int), cudaMemcpyHostToDevice, streams[1]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(lb_sim_domain, &nw, sizeof(int), 2*sizeof(int), cudaMemcpyHostToDevice, streams[2]));

    checkCudaErrors(cudaMemcpyToSymbolAsync(Re, &Reynold_number, sizeof(float), 0, cudaMemcpyHostToDevice, streams[3]));
    checkCudaErrors(cudaMemcpyToSymbolAsync(viscosity, &vis, sizeof(float), 0, cudaMemcpyHostToDevice, streams[4]));
}

__host__ void LB_init_memory( float **f1_gpu, float **f2_gpu, float **feq_gpu, float **source_term_gpu, 
                            float **rho_gpu, float **ux_gpu, float **uy_gpu, float **uz_gpu, 
                            float **rho, float **ux, float **uy, float **uz,
                            cudaStream_t *streams, int NX, int NY, int NZ)
{
    int total_lattice_points = NX*NY*NZ;
    unsigned int mem_size_ndir  = sizeof(float)*total_lattice_points*(19);
    unsigned int mem_size_scalar = sizeof(float)*total_lattice_points;
    checkCudaErrors(cudaMemcpyAsync((void*)(*rho_gpu), (*rho), mem_size_scalar,  cudaMemcpyHostToDevice, streams[5]));
    checkCudaErrors(cudaMemcpyAsync((void*)(*ux_gpu), (*ux), mem_size_scalar,  cudaMemcpyHostToDevice, streams[6]));
    checkCudaErrors(cudaMemcpyAsync((void*)(*uy_gpu), (*uy), mem_size_scalar,  cudaMemcpyHostToDevice, streams[7]));
    checkCudaErrors(cudaMemcpyAsync((void*)(*uz_gpu), (*uz), mem_size_scalar,  cudaMemcpyHostToDevice, streams[8]));

    checkCudaErrors(cudaMemsetAsync ((void*)(*f1_gpu), 0, mem_size_ndir, streams[9]));
    checkCudaErrors(cudaMemsetAsync ((void*)(*f2_gpu), 0, mem_size_ndir, streams[10]));
    checkCudaErrors(cudaMemsetAsync ((void*)(*feq_gpu), 0, mem_size_ndir, streams[11]));
    checkCudaErrors(cudaMemsetAsync ((void*)(*source_term_gpu), 0, mem_size_ndir, streams[12]));
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void LB_init(int NX, int NY, int NZ, float Reynolds, float mu,
                      float **f1_gpu, float **f2_gpu, float **feq_gpu, float **source_term_gpu, 
                      float **rho_gpu, float **ux_gpu, float **uy_gpu, float **uz_gpu, 
                      float **rho, float **ux, float **uy, float **uz, 
                      float **Fx_gpu, float **Fy_gpu, float **Fz_gpu,
                      cudaStream_t *streams)
{
    float bytesPerGiB = 1024.0f*1024.0f*1024.0f;
    float total_size_allocated = LB_allocate_memory(NX, NY, NZ, f1_gpu, f2_gpu, feq_gpu,
                                                    source_term_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);

    LB_init_symbols(NX, NY, NZ, Reynolds, mu, streams);
    LB_init_memory( f1_gpu, f2_gpu, feq_gpu, source_term_gpu, 
                    rho_gpu, ux_gpu, uy_gpu, uz_gpu,
                    rho, ux, uy, uz, streams, NX, NY, NZ);
    
    printf("Total memory allocated in GPU: %f\n", total_size_allocated/bytesPerGiB);

    dim3 nthx(nThreads.x, nThreads.y, nThreads.z);
    dim3 ngrid(NX/nthx.x, NY/nthx.y, NZ/nthx.z);
    LB_compute_sim_param(NX, NY, NZ, mu, Reynolds);
    checkCudaErrors(cudaDeviceSynchronize());
    LB_add_gravity<<<ngrid, nthx>>>(*Fx_gpu, *Fy_gpu, *Fz_gpu, *rho_gpu);
    LB_equi_Initialization<<<ngrid, nthx>>>(*f1_gpu, *feq_gpu, 
                            *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu, 
                            *Fx_gpu, *Fy_gpu, *Fz_gpu);
    
    checkCudaErrors(cudaDeviceSynchronize());
    printf("At start of simulations.......\n");
    update_max_params<<<ngrid, nthx>>>(*Fx_gpu, *Fy_gpu, *Fz_gpu, *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void LB_cleanup(float *f1_gpu, float* f2_gpu, float *feq_gpu, float *source_term_gpu, 
                         float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu)
{
    cudaFree(f1_gpu);
    cudaFree(f2_gpu);
    cudaFree(feq_gpu);
    cudaFree(rho_gpu);
    cudaFree(ux_gpu);
    cudaFree(uy_gpu);
    cudaFree(uz_gpu);
    cudaFree(source_term_gpu);
    printf("Lattice Boltzmann object cleaned\n");
}

__host__ void LB_simulate(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, 
                          float *f1_gpu, float* f2_gpu, float *feq_gpu, float *source_term_gpu, 
                          float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, 
                          int NX, int NY, int NZ, void (*cal_force_spread)(Vertex**, int*, int, int, float*, float*, float*, cudaStream_t *),  void (*advection_force)(Vertex **, int *, int, int,
                            float *, float *, float *, cudaStream_t *),
                          Vertex **nodeLists, int *vertex_size_per_mesh, int num_threads, int num_mesh, cudaStream_t *streams)
{
    dim3 nthx(nThreads.x, nThreads.y, nThreads.z);
    dim3 ngrid(NX/nthx.x, NY/nthx.y, NZ/nthx.z);
    
    LB_reset_max(streams);
    LB_clear_Forces( Fx_gpu, Fy_gpu, Fz_gpu, source_term_gpu, streams, NX, NY, NZ);

    checkCudaErrors(cudaDeviceSynchronize());
    printf("After clear forces\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());
    
        cal_force_spread(nodeLists, vertex_size_per_mesh, num_threads, num_mesh, Fx_gpu, Fy_gpu, Fz_gpu, streams);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After spreading forces\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    //     LB_add_gravity<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu,Fz_gpu, rho_gpu);
    // checkCudaErrors(cudaDeviceSynchronize());
    // printf("After adding gravity\n");
    // update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    // checkCudaErrors(cudaDeviceSynchronize());
    // check_max_params<<<1,1>>>();
    // checkCudaErrors(cudaDeviceSynchronize());
    
        LB_compute_local_params<<<ngrid, nthx>>>(f1_gpu, Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After Computing local params\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_compute_equi_distribution<<<ngrid, nthx>>>(rho_gpu, ux_gpu, uy_gpu, uz_gpu, feq_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After coputing EQUI distributions\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_compute_source_term<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, source_term_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After Computing source terms\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_collide<<<ngrid, nthx>>>(f1_gpu, f2_gpu, feq_gpu, source_term_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After Collision\n");
    check_max_params<<<1,1>>>();
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());

        LB_stream<<<ngrid, nthx>>>(f1_gpu, f2_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After Streaming\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_enforce_boundary_wall<<<ngrid, nthx>>>(f1_gpu, f2_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After boundary wall conditions\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    advection_force(nodeLists, vertex_size_per_mesh, num_threads, num_mesh,
                    ux_gpu, uy_gpu, uz_gpu, streams);
    printf("\n\n\n\n");
}