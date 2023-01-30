#include "LatticeBoltzmann.cuh"

__device__ const float RHO_ATM = 0.001f;
__device__ const float RHO_FLUID = 1.0f;
__device__ const float SMAGRINSKY_CONST= 0.94f;//0.003f;
__device__ const float FILL_OFFSET = 0.01f;
__device__ const float LONELY_THRESH = 0.1f;
__device__ float GRAV_CONST1 = 0.001f;
__device__ float cs_inv_sq = 3.0f;
__device__ float non_dim_tau;// = 1.5f;
__device__ float non_dim_nu;

__device__ float w[19] = {1.0f/3.0f, 1.0f/18.0f,  1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

__device__ int cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0};
__device__ int cy[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1};
__device__ int cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1};

__device__ int finv[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

__device__ int lb_sim_domain[3];
__device__ float vel_factor = 1.0f;

__device__ float max_ux = -9999.0f;
__device__ float max_uy = -9999.0f;
__device__ float max_uz = -9999.0f;
__device__ float max_rho = -9999.0f;
__device__ float max_Fx = -9999.0f;
__device__ float max_Fy = -9999.0f;
__device__ float max_Fz = -9999.0f;
__device__ float total_mass = 0.0f;


float Ct=1.0f;
float Cl=1.0f;
float GRAV_CONST = 0.001f;


float *count_loc;
float *Fx_gpu, *Fy_gpu, *Fz_gpu;
float *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;
float *f1_gpu, *f2_gpu, *feq_gpu, *source_term_gpu;
float *mass_gpu, *strain_rate_gpu, *delMass;
float *empty_filled_cell;
unsigned int *cell_type_gpu;
float *temp_cell_type_gpu;

__device__ float clamp(float val, float ulimit, float llimit)
{
    if (val>ulimit)
        return ulimit;
    else if(val<llimit)
        return llimit;
    else return val;
}

__device__ static float atomicMax(float* addr, float value)
{
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ static float atomicMin (float* addr, float value) 
{
    float old;
    old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ float calculate_gas_DF(float rho, float ux, float uy, float uz, int i)
{
    float u_dot_c, u_dot_u = ux * ux + uy * uy + uz * uz;
    u_dot_c= cx[i] * ux + cy[i] * uy + cz[i] * uz;
    float feq = w[i]*rho*(1+cs_inv_sq*(u_dot_c +0.5*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5*u_dot_u));//fmaxf(w[i]*rho*(1+cs_inv_sq*(u_dot_c +0.5*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5*u_dot_u)), 0.0f);//
    return feq;
}

__device__ int calculate_fill_fraction(float rho, float mass, int type)
{
    if (type == FLUID || type == OBSTACLE)
		return 1;
    else if (type == INTERFACE)
    {
        if(rho>0)
        {
            float ep = mass/rho;
            if(ep>1)
                ep = 1.0f;
            else if(ep<0.0f)
                ep = 0.0f;
            return ep;
        }
        else
            return 0.0f;
    }
    else
        return 0.0f;
}

__device__ glm::f32vec3 calculate_normal(int idx, int idy, int idz, float *rho_gpu, float *mass_gpu, unsigned int  *temp_cell_type_gpu)
{
    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    glm::f32vec3 norm = glm::f32vec3(0.0f);
    int xm = gpu_scalar_index((idx+NX-1)%NX, (idy+NY)%NY, (idz+NZ)%NZ, lb_sim_domain);
    int xp = gpu_scalar_index((idx+NX+1)%NX, (idy+NY)%NY, (idz+NZ)%NZ, lb_sim_domain);
    int ym = gpu_scalar_index((idx+NX)%NX, (idy+NY-1)%NY, (idz+NZ)%NZ, lb_sim_domain);
    int yp = gpu_scalar_index((idx+NX)%NX, (idy+NY+1)%NY, (idz+NZ)%NZ, lb_sim_domain);
    int zm = gpu_scalar_index((idx+NX)%NX, (idy+NY)%NY, (idz+NZ-1)%NZ, lb_sim_domain);
    int zp = gpu_scalar_index((idx+NX)%NX, (idy+NY)%NY, (idz+NZ+1)%NZ, lb_sim_domain);
    
    norm.x= 0.5f*(calculate_fill_fraction(mass_gpu[xm], rho_gpu[xm], temp_cell_type_gpu[xm]) - calculate_fill_fraction(mass_gpu[xp], rho_gpu[xp], temp_cell_type_gpu[xp]));
    norm.y = 0.5f*(calculate_fill_fraction(mass_gpu[ym], rho_gpu[ym], temp_cell_type_gpu[ym]) - calculate_fill_fraction(mass_gpu[yp], rho_gpu[yp], temp_cell_type_gpu[yp]));
    norm.z = 0.5f*(calculate_fill_fraction(mass_gpu[zm], rho_gpu[zm], temp_cell_type_gpu[zm]) - calculate_fill_fraction(mass_gpu[zp], rho_gpu[zp], temp_cell_type_gpu[zp]));

    return norm;
}

__global__ void cell_initialize(float *mass_gpu, unsigned int *cell_type_gpu, float *rho_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb;

    if(idx == 0 || idy == 0 || idz == 0 || idx == NX-1 || idy == NY-1 || idz == NZ-1)
    {
        cell_type_gpu[x_] = OBSTACLE;
        mass_gpu[x_] = -99999.0f;
    }
    else if(rho_gpu[x_]==RHO_ATM)
    {
        cell_type_gpu[x_] = EMPTY;
        mass_gpu[x_] = 0.0f;
        rho_gpu[x_] = RHO_ATM;
    }
    else if(rho_gpu[x_]==RHO_FLUID)
    {
        cell_type_gpu[x_] = (FLUID);
        rho_gpu[x_] = RHO_FLUID;
        mass_gpu[x_] = rho_gpu[x_];
    }
}

__global__ void construct_interface(unsigned int *cell_type_gpu, float *mass_gpu, float *rho_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb;
    if(cell_type_gpu[x_] == FLUID)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index((idx+NX+cx[i])%NX, (idy+NY+cy[i])%NY, (idz+NZ+cz[i])%NZ, lb_sim_domain);
            if(!(cell_type_gpu[nb] == OBSTACLE) && (cell_type_gpu[nb] == EMPTY))
            {
                // printf(" Kevin ");
                cell_type_gpu[nb] = INTERFACE ;
                rho_gpu[nb] = RHO_FLUID;
                mass_gpu[nb] = 0.5f;
            }
        }
    }
}

__global__ void IF_cell_update_mass(float *rho_gpu, float *mass_gpu, float *delMass, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    
    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;
    mass_gpu[x_] += delMass[x_];
}

__global__ void IF_update_cell_type(float *rho_gpu, float *mass_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb;
    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;
    
    if(temp_cell_type_gpu[x_] == INTERFACE)
    {
        bool NO_FLUID_NB = true;
        bool NO_EMPTY_NB = true;
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index(idx+cx[i], idy+cy[i], idz+cz[i], lb_sim_domain);
            NO_FLUID_NB = (NO_FLUID_NB && temp_cell_type_gpu[nb]!=FLUID);
            NO_EMPTY_NB = (NO_EMPTY_NB && temp_cell_type_gpu[nb]!=EMPTY);
        }
        if(mass_gpu[x_]>=(1.0f + FILL_OFFSET)*rho_gpu[x_]|| (NO_EMPTY_NB))// && mass_gpu[x_]>=(1-FILL_OFFSET)*rho_gpu[x_]))
            temp_cell_type_gpu[x_] = IF_TO_FLUID;
        else if(mass_gpu[x_]<=(0.0f - FILL_OFFSET)*rho_gpu[x_] || (NO_FLUID_NB))// && mass_gpu[x_]<=FILL_OFFSET*rho_gpu[x_]))
            temp_cell_type_gpu[x_] = IF_TO_EMPTY;
    }
}

__global__ void IF_filled_nb_flag_update(unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb, nbb;
    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;
    if(temp_cell_type_gpu[x_] == IF_TO_FLUID)// || temp_cell_type_gpu[x_] == FLUID)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index(idx+cx[i], idy+cy[i], idz+cz[i], lb_sim_domain);
            if(temp_cell_type_gpu[nb] == EMPTY)
                temp_cell_type_gpu[nb] = EMPTY_TO_IF;
            else if(temp_cell_type_gpu[nb] == IF_TO_EMPTY)
                temp_cell_type_gpu[nb] = INTERFACE;
        }
    }
}

__global__ void IF_filled_nb_DF_update(unsigned int *temp_cell_type_gpu, float *f1_gpu, float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, float *mass_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb, nbb;
    float ux=0.0f, uy=0.0f, uz=0.0f, rho=0.0f;
    int count_nb=0;
    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;
    if(temp_cell_type_gpu[x_] == EMPTY_TO_IF)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index(idx+cx[i], idy+cy[i], idz+cz[i], lb_sim_domain);
            if(temp_cell_type_gpu[nb] == INTERFACE || temp_cell_type_gpu[nb] == FLUID || temp_cell_type_gpu[nb] == IF_TO_FLUID)
            {
                ux += ux_gpu[nb];
                uy += uy_gpu[nb];
                uz += uz_gpu[nb];
                rho += rho_gpu[nb];
                count_nb+=1;
            }
        }
        if(count_nb>0)
        {
            float count_inv = 1.0f/(float)count_nb;
            rho = rho*count_inv;
            ux = ux*count_inv;
            uy = uy*count_inv;
            uz = uz*count_inv;
            //mass = mass/count_nb;
        }
        else
        {
            rho = 0.0f;
            ux = 0.0f;
            uy = 0.0f;
            uz = 0.0f;
        }

        for(int i=0;i<19;i++)
            f1_gpu[gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain)] = calculate_gas_DF(rho, ux, uy, uz, i);
        // if(rho<0.0f)
        //     printf(" (em %f) ", rho);
        ux_gpu[x_] = ux;
        uy_gpu[x_] = uy;
        uz_gpu[x_] = uz;
        rho_gpu[x_] = rho;
        mass_gpu[x_] = 0.0f;
    }
}

__global__ void IF_empty_nb_flag_update(unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int nb;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;
    if(temp_cell_type_gpu[x_] == IF_TO_EMPTY)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index(idx+cx[i], idy+cy[i], idz+cz[i], lb_sim_domain);
            if((temp_cell_type_gpu[nb] == FLUID) ||(temp_cell_type_gpu[nb] == IF_TO_FLUID))
                temp_cell_type_gpu[nb] = INTERFACE;
        }
    }
}

__global__ void IF_distribute_excess_mass(float *mass_gpu, float *rho_gpu, unsigned int *temp_cell_type_gpu, float *delMass)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int nb;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    float mass_ex=0.0f;
    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;
    if(temp_cell_type_gpu[x_] == FLUID)
    {
        mass_ex = mass_gpu[x_]- RHO_FLUID;
        mass_gpu[x_] = RHO_FLUID;
    }
    else if(temp_cell_type_gpu[x_] == EMPTY)
    {
        mass_ex = mass_gpu[x_];
        mass_gpu[x_] = 0.0f;
    }
    else if(temp_cell_type_gpu[x_] == INTERFACE)
    {
        if(mass_gpu[x_]>=rho_gpu[x_]*(1+ FILL_OFFSET))
        {
            mass_ex = mass_gpu[x_]-rho_gpu[x_];
            mass_gpu[x_] = rho_gpu[x_];
        }
        else if(mass_gpu[x_]<=rho_gpu[x_]*(0.0f-FILL_OFFSET))
        {
            mass_ex = mass_gpu[x_];
            mass_gpu[x_] = 0.0f;
        }
    }
    else if(temp_cell_type_gpu[x_] == IF_TO_FLUID)
    {
        mass_ex = mass_gpu[x_]- RHO_FLUID;
        mass_gpu[x_] = RHO_FLUID;
    }
    else if(temp_cell_type_gpu[x_] == IF_TO_EMPTY)
    {
        mass_ex = mass_gpu[x_];
        mass_gpu[x_] = 0.0f;
    }
    else if(temp_cell_type_gpu[x_] == EMPTY_TO_IF)
    {
        if(mass_gpu[x_]>=rho_gpu[x_]*(1 - FILL_OFFSET))
        {
            mass_ex = mass_gpu[x_]-rho_gpu[x_];
            mass_gpu[x_] = rho_gpu[x_];
        }
        else if(mass_gpu[x_]<=rho_gpu[x_]*(0.0f + FILL_OFFSET))
        {
            mass_ex = mass_gpu[x_];
            mass_gpu[x_] = 0.0f;
        }
    }

    int count = 0;
    for(int i=1;i<19;i++)
    {
        nb = gpu_scalar_index((idx+NX+cx[i])%NX, (idy+NY+cy[i])%NY, (idz+NZ+cz[i])%NZ, lb_sim_domain);
        count += ((temp_cell_type_gpu[nb] == INTERFACE) || (temp_cell_type_gpu[nb] == (EMPTY_TO_IF)) || (temp_cell_type_gpu[nb] == (FLUID)) || (temp_cell_type_gpu[nb] == (IF_TO_FLUID)));
    }
   delMass[x_] = count>0?(1.0f/(float)count)*mass_ex:0.0f;
   mass_gpu[x_] += count>0?0.0f:mass_ex;
}

__global__ void IF_collect_fluid_mass(float *mass_gpu, unsigned int *temp_cell_type_gpu, float *delMass, float *rho_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb;
    
    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;


    if(temp_cell_type_gpu[x_] == EMPTY_TO_IF)
        temp_cell_type_gpu[x_] = INTERFACE;
    if(temp_cell_type_gpu[x_] == IF_TO_FLUID)
    {
        // if(mass_gpu[x_]>=rho_gpu[x_]*(1.0f+FILL_OFFSET))
            temp_cell_type_gpu[x_] = FLUID;
        // else
        //     temp_cell_type_gpu[x_] = INTERFACE;
    }
            // temp_cell_type_gpu[x_] = FLUID;
    if(temp_cell_type_gpu[x_] == IF_TO_EMPTY)
    {
        // if(mass_gpu[x_]<=rho_gpu[x_]*(0.0f-FILL_OFFSET))
            temp_cell_type_gpu[x_] = EMPTY;
        // else
        //     temp_cell_type_gpu[x_] = INTERFACE;
    }

    if((temp_cell_type_gpu[x_] == INTERFACE) || (temp_cell_type_gpu[x_] == (FLUID)))
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index((idx+NX+cx[i])%NX, (idy+NY+cy[i])%NY, (idz+NZ+cz[i])%NZ, lb_sim_domain);
            if(temp_cell_type_gpu[nb] != OBSTACLE)
                mass_gpu[x_] += delMass[nb];
        }
    }
}

__global__ void IF_stream_mass_transfer(float *f1_gpu, float *f2_gpu, float *rho_gpu, unsigned int *temp_cell_type_gpu, float *mass_gpu, float *delMass)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int nb, coordNB, x_, coord;
    float delM = 0.0f;
    x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);

    if(temp_cell_type_gpu[x_] == OBSTACLE)
        return;
    if(temp_cell_type_gpu[x_] == FLUID)
    {
        for(int i=1;i<19;i++)
        {
            coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
            int coord_inv = gpu_fieldn_index(idx, idy, idz, finv[i], lb_sim_domain);
            delMass[x_] += f1_gpu[coord_inv] - f2_gpu[coord];//(fmaxf(f1_gpu[coord_inv], 0.0f) - fmaxf(f2_gpu[coord], 0.0f));//
        }
    }
    if(temp_cell_type_gpu[x_] == INTERFACE)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index(idx+cx[i], idy+cy[i], idz+cz[i],lb_sim_domain);
            coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
            int coord_inv = gpu_fieldn_index(idx, idy, idz, finv[i], lb_sim_domain);
            float avg_frac = 0.5f*(calculate_fill_fraction(rho_gpu[nb], mass_gpu[nb], temp_cell_type_gpu[nb]) + calculate_fill_fraction(rho_gpu[x_], mass_gpu[x_], temp_cell_type_gpu[x_]));
            if(temp_cell_type_gpu[nb] == (FLUID))
                delMass[x_] += f1_gpu[coord_inv] - f2_gpu[coord];//(fmaxf(f1_gpu[coord_inv], 0.0f) - fmaxf(f2_gpu[coord], 0.0f));//
            else if(temp_cell_type_gpu[nb] == INTERFACE)
                delMass[x_] += (f1_gpu[coord_inv] - f2_gpu[coord])*avg_frac;//(fmaxf(f1_gpu[coord_inv], 0.0f) - fmaxf(f2_gpu[coord], 0.0f))*avg_frac;//
            else if(temp_cell_type_gpu[nb] == EMPTY)
                delMass[x_] += 0.0f;
        }
    }
}

__global__ void LB_compute_local_params(float *f1_gpu, float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float lat_rho=0.000f, lat_rho_ux=0.000f, lat_rho_uy=0.000f, lat_rho_uz=0.000f;

    float f_val = 0.000f;
    float cs = 1.0f/sqrt(cs_inv_sq);
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[x_] == OBSTACLE|| temp_cell_type_gpu[x_] == EMPTY))
    {
        for(int i=0;i<19;i++)
        {
            x_ = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
            f_val = f1_gpu[x_];
            lat_rho += f_val;
            lat_rho_ux += f_val*cx[i];
            lat_rho_uy += f_val*cy[i];
            lat_rho_uz += f_val*cz[i];
        }
        //lat_rho += 1.0f;
        x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);

        rho_gpu[x_] = lat_rho;
        float rho_inv = (lat_rho <= 0.0f?0.0f:(1.0f/lat_rho));
        float newU = (lat_rho_ux + 0.5f * Fx_gpu[x_]) * rho_inv;
        float newV = (lat_rho_uy + 0.5f * Fy_gpu[x_]) * rho_inv;
        float newW = (lat_rho_uz + 0.5f * Fz_gpu[x_]) * rho_inv;

        float SPEED_LIMIT = sqrt(1.0f/3.0f);
        float len = glm::length(glm::f32vec3(newU, newV, newW));
        float factor = 1.0f;
        if(len>SPEED_LIMIT)
            factor = SPEED_LIMIT/len;

        ux_gpu[x_] = newU*factor;
        uy_gpu[x_] = newV*factor;
        uz_gpu[x_] = newW*factor;
    }
}

__global__ void LB_compute_equi_distribution(float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu,
                                             float* feq_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float lat_rho, lat_ux, lat_uy, lat_uz;
    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] == OBSTACLE|| temp_cell_type_gpu[coord] == EMPTY))
    {
        lat_rho =  rho_gpu[coord];
        lat_ux = ux_gpu[coord];
        lat_uy = uy_gpu[coord];
        lat_uz = uz_gpu[coord];

        float u_dot_c, u_dot_u = lat_ux * lat_ux + lat_uy * lat_uy + lat_uz * lat_uz;
        for(int i=0;i<19;i++)
        {
            u_dot_c= cx[i] * lat_ux + cy[i] * lat_uy + cz[i] * lat_uz;
            feq_gpu[gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain)] = w[i]*lat_rho*(1.0f+cs_inv_sq*(u_dot_c +0.5f*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5f*u_dot_u));//fmaxf(w[i]*lat_rho*(1.0f+cs_inv_sq*(u_dot_c +0.5f*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5f*u_dot_u)), 0.0f);//
        }
    }
}

__global__ void LB_compute_source_term(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *source_term_gpu,
                                        float *ux_gpu, float *uy_gpu, float *uz_gpu, float *strain_rate_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float ci_dot_u = 0.0f, ci_dot_F = 0.0f;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] == OBSTACLE|| temp_cell_type_gpu[coord] == EMPTY))
    {
        float lat_ux = ux_gpu[coord];
        float lat_uy = uy_gpu[coord];
        float lat_uz = uz_gpu[coord];

        float quant = powf(non_dim_nu, 2.0f)+18.0f*SMAGRINSKY_CONST*SMAGRINSKY_CONST*sqrt(strain_rate_gpu[coord]);
        float S = (1.0f/(6.0f*powf(SMAGRINSKY_CONST, 2.0f)))*(sqrt(quant) - non_dim_nu);
        float tau = cs_inv_sq * (non_dim_nu + SMAGRINSKY_CONST*SMAGRINSKY_CONST*S)+0.5f;
        // float tau = non_dim_tau;//+ 0.5f;
        float tau_inv = (1.0f/tau);
        float coef = (1.0f-0.5f*tau_inv);
        float Fi = 0;
        float u_dot_F = lat_ux*Fx_gpu[coord] + lat_uy*Fy_gpu[coord] + lat_uz*Fz_gpu[coord];
        for(int i=0;i<19;i++)
        {
            ci_dot_F = cx[i]*Fx_gpu[coord] + cy[i]*Fy_gpu[coord] + cz[i]*Fz_gpu[coord];
            ci_dot_u = cx[i]*lat_ux + cy[i]*lat_uy + cz[i]*lat_uz;
            Fi = cs_inv_sq * ((ci_dot_F - u_dot_F) + cs_inv_sq*(ci_dot_F*ci_dot_u));
            source_term_gpu[gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain)] = w[i]* coef * Fi;
        }
    }
}

__global__ void LB_equi_Initialization( float *f1_gpu, float *feq_gpu,
                                        float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, 
                                        float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float lat_rho, lat_ux, lat_uy, lat_uz;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[x_] == OBSTACLE|| temp_cell_type_gpu[x_] == EMPTY))
    {
        lat_rho =  rho_gpu[x_];
        float rho_inv = 1.0f;//(lat_rho <= 0.0f)?0.0f:(1.0f/lat_rho);
        lat_ux = ux_gpu[x_] + 0.5f*rho_inv*Fx_gpu[x_];
        lat_uy = uy_gpu[x_] + 0.5f*rho_inv*Fy_gpu[x_];
        lat_uz = uz_gpu[x_] + 0.5f*rho_inv*Fz_gpu[x_];
        ux_gpu[x_] = lat_ux;
        uy_gpu[x_] = lat_uy;
        uz_gpu[x_] = lat_uz;

        float u_dot_c, u_dot_u = lat_ux * lat_ux + lat_uy * lat_uy + lat_uz * lat_uz;
        float val = 0.0f;
        for(int i=0;i<19;i++)
        {
            x_ = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
            u_dot_c= cx[i] * lat_ux + cy[i] * lat_uy + cz[i] * lat_uz;
            val = w[i]*lat_rho*(1.0f + cs_inv_sq*(u_dot_c +0.5*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5*u_dot_u));
            feq_gpu[x_] = val;
            f1_gpu[x_] = val;
        }
        
    }
}

__global__ void LB_collide(float *f1_gpu, float* f2_gpu, float *feq_gpu, float *source_term_gpu, float *strain_rate_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] == OBSTACLE|| temp_cell_type_gpu[coord] == EMPTY))
    {
        float omega, source;
        float quant = powf(non_dim_nu, 2.0f)+18.0f*SMAGRINSKY_CONST*SMAGRINSKY_CONST*sqrt(strain_rate_gpu[coord]);
        float S = (1.0f/(6.0f*powf(SMAGRINSKY_CONST, 2.0f)))*(sqrt(quant) - non_dim_nu);
        float tau = cs_inv_sq * (non_dim_nu + SMAGRINSKY_CONST*SMAGRINSKY_CONST*S)+0.5f;
        // float tau = non_dim_tau;//+0.5f;
        float tau_inv = (1.0f/tau);
        for(int i =0;i<19;i++)
        {
            coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
            omega = -1.0f*tau_inv*(f1_gpu[coord] - feq_gpu[coord]);//-1.0f*tau_inv*(fmaxf(f1_gpu[coord], 0.0f) - fmaxf(feq_gpu[coord], 0.0f));//
            source = source_term_gpu[coord];
            f2_gpu[coord] = f1_gpu[coord] + (omega + source);//fmaxf(fmaxf(f1_gpu[coord], 0.0f) + (omega + source), 0.0f);//
        }
    }
}

__global__ void LB_stream(float *f1_gpu, float *f2_gpu, float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, unsigned int *temp_cell_type_gpu, float *mass_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int nb, coordNB, x_, coord;
    x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);

    if(!(temp_cell_type_gpu[x_] == OBSTACLE|| temp_cell_type_gpu[x_] == EMPTY))
    {
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 0, lb_sim_domain)] = f2_gpu[gpu_fieldn_index(idx, idy, idz, 0, lb_sim_domain)];

        if(temp_cell_type_gpu[x_] == FLUID)
        {
            for(int i=1;i<19;i++)
            {
                nb = gpu_scalar_index(idx+cx[finv[i]], idy+cy[finv[i]], idz+cz[finv[i]],lb_sim_domain);
                coordNB = gpu_fieldn_index(idx+cx[finv[i]], idy+cy[finv[i]], idz+cz[finv[i]], i, lb_sim_domain);
                coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
                int coord_inv = gpu_fieldn_index(idx, idy, idz, finv[i], lb_sim_domain);
                if(temp_cell_type_gpu[nb] == FLUID || temp_cell_type_gpu[nb] == INTERFACE)
                    f1_gpu[coord] = f2_gpu[coordNB];//fmaxf(f2_gpu[coordNB], 0.0f);//
                else if(temp_cell_type_gpu[nb] == OBSTACLE)
                    f1_gpu[coord] = f2_gpu[coord_inv];//fmaxf(f2_gpu[coord_inv], 0.0f);//
            }
        }
        else if(temp_cell_type_gpu[x_] == INTERFACE)
        {
            float lat_ux = ux_gpu[x_];
            float lat_uy = uy_gpu[x_];
            float lat_uz = uz_gpu[x_];
            for(int i=1;i<19;i++)
            {
                nb = gpu_scalar_index(idx+cx[finv[i]], idy+cy[finv[i]], idz+cz[finv[i]],lb_sim_domain);
                coordNB = gpu_fieldn_index(idx+cx[finv[i]], idy+cy[finv[i]], idz+cz[finv[i]], i, lb_sim_domain);
                coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
                int coord_inv = gpu_fieldn_index(idx, idy, idz, finv[i], lb_sim_domain);
                if(temp_cell_type_gpu[nb] == FLUID || temp_cell_type_gpu[nb] == INTERFACE)
                    f1_gpu[coord] = f2_gpu[coordNB];//fmaxf(f2_gpu[coordNB], 0.0f);//
                else if(temp_cell_type_gpu[nb] == EMPTY)
                {
                    float avg_pop = 0.5f*(calculate_gas_DF(RHO_FLUID, lat_ux, lat_uy, lat_uz, i) + calculate_gas_DF(RHO_FLUID, lat_ux, lat_uy, lat_uz, finv[i]));
                    f1_gpu[coord] = 2.0f*avg_pop - f2_gpu[coord_inv];//fmaxf(2.0f*avg_pop - fmaxf(f2_gpu[coord_inv], 0.0f), 0.0f);//
                }
                else if(temp_cell_type_gpu[nb] == OBSTACLE)
                    f1_gpu[coord] = f2_gpu[coord_inv];//fmaxf(f2_gpu[coord_inv], 0.0f);//
            }

            // glm::f32vec3 norm = calculate_normal(idx, idy, idz, rho_gpu, mass_gpu, temp_cell_type_gpu);
            // for(int i=1;i<19;i++)
            // {
            //     float n_ci = norm.x*cx[finv[i]]+norm.y*cy[finv[i]]+norm.z*cz[finv[i]];
            //     int coord_inv = gpu_fieldn_index(idx, idy, idz, finv[i], lb_sim_domain);
            //     if(n_ci<0.0f)
            //     {
            //         float avg_pop = 0.5f*(calculate_gas_DF(RHO_FLUID, lat_ux, lat_uy, lat_uz, i) + calculate_gas_DF(RHO_FLUID, lat_ux, lat_uy, lat_uz, finv[i]));
            //         f1_gpu[coord] = 2.0f*avg_pop - f2_gpu[coord_inv];//fmaxf(2.0f*avg_pop - fmaxf(f2_gpu[coord_inv], 0.0f), 0.0f);//
            //     }
            // }
        }
    }
}

__global__ void check_max_params()
{
    printf("Max values of params = %f %f %f %f\t%f %f %f\t%f\n", max_ux, max_uy, max_uz, max_rho, max_Fx, max_Fy, max_Fz, total_mass);
}

__global__ void update_max_params(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, unsigned int *temp_cell_type_gpu,
                                  float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] == (OBSTACLE)))
    {
        atomicMax(&max_ux, abs(ux_gpu[coord]));
        atomicMax(&max_uy, abs(uy_gpu[coord]));
        atomicMax(&max_uz, abs(uz_gpu[coord]));
        atomicMax(&max_rho, (rho_gpu[coord]));
        atomicMax(&max_Fx, abs(Fx_gpu[coord]));
        atomicMax(&max_Fy, abs(Fy_gpu[coord]));
        atomicMax(&max_Fz, abs(Fz_gpu[coord]));
    }
}

__global__ void update_total_mass(float *mass_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] == (OBSTACLE)) && !(temp_cell_type_gpu[coord] == (EMPTY)))
        atomicAdd(&total_mass, mass_gpu[coord]);
}

__global__ void LB_compute_stress(float *source_term_gpu, float *f1_gpu, float *feq_gpu, float *strain_rate_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(temp_cell_type_gpu[coord] == FLUID || temp_cell_type_gpu[coord] == INTERFACE)
    {
        float quant = powf(non_dim_nu, 2.0f)+18.0f*SMAGRINSKY_CONST*SMAGRINSKY_CONST*sqrt(strain_rate_gpu[coord]);
        float S = (1.0f/(6.0f*powf(SMAGRINSKY_CONST, 2.0f)))*(sqrt(quant) - non_dim_nu);
        float tau = cs_inv_sq * (non_dim_nu + SMAGRINSKY_CONST*SMAGRINSKY_CONST*S)+0.5f;
        float tau_inv = (1.0f/tau);
        float coef1 = 1.0f - 0.5f*tau_inv;
        float coef2 = 0.5f;
        float sumAB[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int coordN;
        float fi, feq, Fi;
        for(int i=0;i<19;i++)
        {
            coordN = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
            fi = f1_gpu[coordN];
            feq = feq_gpu[coordN];
            Fi = source_term_gpu[coordN];
            sumAB[0] += cx[i]*cx[i]*(coef2*(fi-feq) + coef2*coef1*Fi);
            sumAB[1] += cy[i]*cy[i]*(coef2*(fi-feq) + coef2*coef1*Fi);
            sumAB[2] += cz[i]*cz[i]*(coef2*(fi-feq) + coef2*coef1*Fi);
            sumAB[3] += cx[i]*cy[i]*(coef2*(fi-feq) + coef2*coef1*Fi);
            sumAB[4] += cx[i]*cz[i]*(coef2*(fi-feq) + coef2*coef1*Fi);
            sumAB[5] += cy[i]*cz[i]*(coef2*(fi-feq) + coef2*coef1*Fi);
        }
        strain_rate_gpu[coord] = powf(sumAB[0],2.0f) + powf(sumAB[1], 2.0f) + powf(sumAB[2], 2.0f) + 2.0f*(powf(sumAB[3], 2.0f) + powf(sumAB[4], 2.0f) + powf(sumAB[5], 2.0f));
    }
}

__global__ void LB_add_gravity(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *rho_gpu, unsigned int *temp_cell_type_gpu, float GRAV_CONST)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float domain_size = 2.0f;
    float Cl = domain_size/max(max(lb_sim_domain[0], lb_sim_domain[1]), lb_sim_domain[2]);
    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] == (OBSTACLE)))
        atomicAdd(&Fy_gpu[coord], -1.0f * GRAV_CONST1 * rho_gpu[coord]);
}

__host__ void LB_clear_Forces(cudaStream_t *streams, int NX, int NY, int NZ)
{
    int total_lattice_points = NX*NY*NZ;
    checkCudaErrors(cudaMemset ((void*)(Fx_gpu), 0, total_lattice_points*sizeof(float)));
    checkCudaErrors(cudaMemset ((void*)(Fy_gpu), 0, total_lattice_points*sizeof(float)));
    checkCudaErrors(cudaMemset ((void*)(Fz_gpu), 0, total_lattice_points*sizeof(float)));
    checkCudaErrors(cudaMemset ((void*)(source_term_gpu), 0, 19*total_lattice_points*sizeof(float)));
    checkCudaErrors(cudaMemset ((void*)(count_loc), 0, total_lattice_points*sizeof(int)));

    checkCudaErrors(cudaMemset ((void*)(delMass), 0, total_lattice_points*sizeof(float)));
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void LB_reset_max(cudaStream_t *streams)
{
    float val_1 = -9999.0f;
    float val_2 = 0.0f;
    float val_3 = 1.0f;
    checkCudaErrors(cudaMemcpyToSymbol(max_rho, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_ux, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_uy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_uz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_Fx, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_Fy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_Fz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(total_mass, &val_2, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(vel_factor, &val_3, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void copy_cell_type(float *temp_cell_type_gpu, unsigned int *cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    temp_cell_type_gpu[coord] = cell_type_gpu[coord];
}

__host__ void LB_compute_sim_param(int NX, int NY, int NZ, float viscosity, float Re)
{
    float domain_size = 2.0;
    float cs_inv_sq = 3.0f;
    float lat_l_no_dim = max(max(NX, NY), NZ);
    
    float delX = domain_size/lat_l_no_dim;
    float deltaT = sqrt(GRAV_CONST*delX/10.0f);
    float nu_star = viscosity*(deltaT/(delX*delX));
    float tau_star = cs_inv_sq*(nu_star) +0.5f;

    checkCudaErrors(cudaMemcpyToSymbol(non_dim_tau, &tau_star, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(non_dim_nu, &nu_star, sizeof(float), 0, cudaMemcpyHostToDevice));

    Cl = delX;
    Ct = deltaT;
    printf("..................Simulation parameters are as follows.........................\n");
    printf("Lattice Reynolds number : %f\n", Re);
    printf("non dimentional del T :%f\n", deltaT);
    printf("non dimentional tau :%f\n", tau_star);
    printf("non dimentional Viscosity :%f\n", nu_star);
    printf("non dimentional gravity %f\n", GRAV_CONST);
    printf("Velocity conversion factor %f\n", delX/deltaT);
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ float LB_allocate_memory(int NX, int NY, int NZ)
{
    float total_size_allocated = 0;
    int total_lattice_points = NX*NY*NZ;
    unsigned int mem_size_ndir  = sizeof(float)*total_lattice_points*(19);
    unsigned int mem_size_scalar = sizeof(float)*total_lattice_points;

    checkCudaErrors(cudaMalloc((void**)&f1_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;
    checkCudaErrors(cudaMalloc((void**)&f2_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;
    checkCudaErrors(cudaMalloc((void**)&feq_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;
    checkCudaErrors(cudaMalloc((void**)&source_term_gpu, mem_size_ndir));
    total_size_allocated += mem_size_ndir;

    checkCudaErrors(cudaMalloc((void**)&rho_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&ux_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&uy_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&uz_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&mass_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&Fx_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&Fy_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&Fz_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&strain_rate_gpu, mem_size_scalar));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&cell_type_gpu, total_lattice_points*sizeof(unsigned int)));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&temp_cell_type_gpu, total_lattice_points*sizeof(float)));
    total_size_allocated += mem_size_scalar;
    checkCudaErrors(cudaMalloc((void**)&delMass, total_lattice_points*sizeof(float)));
    total_size_allocated += mem_size_scalar;

    checkCudaErrors(cudaMalloc((void**)&count_loc, mem_size_scalar));
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
    
    checkCudaErrors(cudaMemcpyToSymbol(lb_sim_domain, &nu, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(lb_sim_domain, &nv, sizeof(int), sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(lb_sim_domain, &nw, sizeof(int), 2*sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void LB_init_memory(float **rho, float **ux, float **uy, float **uz,
                             cudaStream_t *streams, int NX, int NY, int NZ)
{
    int total_lattice_points = NX*NY*NZ;
    unsigned int mem_size_ndir  = sizeof(float)*total_lattice_points*(19);
    unsigned int mem_size_scalar = sizeof(float)*total_lattice_points;

    checkCudaErrors(cudaMemcpy((void*)(rho_gpu), (*rho), mem_size_scalar,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)(ux_gpu), (*ux), mem_size_scalar,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)(uy_gpu), (*uy), mem_size_scalar,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void*)(uz_gpu), (*uz), mem_size_scalar,  cudaMemcpyHostToDevice));

    cudaMemset((void*)(mass_gpu), 0, mem_size_scalar);

    checkCudaErrors(cudaMemset((void*)(Fx_gpu), 0, mem_size_scalar));
    checkCudaErrors(cudaMemset((void*)(Fy_gpu), 0, mem_size_scalar));
    checkCudaErrors(cudaMemset((void*)(Fz_gpu), 0, mem_size_scalar));
    checkCudaErrors(cudaMemset((void*)(strain_rate_gpu), 0, mem_size_scalar));

    checkCudaErrors(cudaMemset((void*)(f1_gpu), 0, mem_size_ndir));
    checkCudaErrors(cudaMemset((void*)(f2_gpu), 0, mem_size_ndir));
    checkCudaErrors(cudaMemset((void*)(feq_gpu), 0, mem_size_ndir));
    checkCudaErrors(cudaMemset((void*)(source_term_gpu), 0, mem_size_ndir));

    checkCudaErrors(cudaMemset((void*)(count_loc), 0, mem_size_scalar));
    checkCudaErrors(cudaMemset((void*)(delMass), 0, mem_size_scalar));
}

__host__ float LB_init(int NX, int NY, int NZ, float Reynolds, float mu,
                      float **rho, float **ux, float **uy, float **uz,
                      cudaStream_t *streams)
{
    float total_size_allocated = LB_allocate_memory(NX, NY, NZ);
    LB_init_memory(rho, ux, uy, uz, streams, NX, NY, NZ);
    LB_init_symbols(NX, NY, NZ, Reynolds, mu, streams);
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 nthx(nThreads.x, nThreads.y, nThreads.z);
    dim3 ngrid(NX/nthx.x, NY/nthx.y, NZ/nthx.z);

    LB_compute_sim_param(NX, NY, NZ, mu, Reynolds);
    
    cell_initialize<<<ngrid, nthx>>>(mass_gpu, cell_type_gpu, rho_gpu);
    construct_interface<<<ngrid, nthx>>>(cell_type_gpu, mass_gpu, rho_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    copy_cell_type<<<ngrid, nthx>>>(temp_cell_type_gpu, cell_type_gpu);

    LB_reset_max(streams);
    LB_clear_Forces(streams, NX, NY, NZ);
    LB_add_gravity<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, cell_type_gpu, GRAV_CONST);
    LB_equi_Initialization<<<ngrid, nthx>>>(f2_gpu, feq_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, Fx_gpu, Fy_gpu, Fz_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());

    return total_size_allocated;
}

__host__ void LB_cleanup()
{
    cudaFree(f1_gpu);
    cudaFree(f2_gpu);
    cudaFree(feq_gpu);
    cudaFree(rho_gpu);
    cudaFree(ux_gpu);
    cudaFree(uy_gpu);
    cudaFree(uz_gpu);
    cudaFree(mass_gpu);
    cudaFree(Fx_gpu);
    cudaFree(Fy_gpu);
    cudaFree(Fz_gpu);
    cudaFree(cell_type_gpu);
    cudaFree(temp_cell_type_gpu);
    cudaFree(delMass);
    cudaFree(strain_rate_gpu);
    cudaFree(source_term_gpu);

    cudaFree(count_loc);
    printf("Lattice Boltzmann object cleaned\n");
}

__host__ void LB_simulate_RB(int NX, int NY, int NZ, float Ct,
                            void (*cal_force_spread_RB)(int, int, float, cudaStream_t *), 
                            void (*advect_velocity)(int, int, cudaStream_t *),
                            int num_threads, int num_mesh, cudaStream_t *streams)
{
    dim3 nthx(nThreads.x, nThreads.y, nThreads.z);
    dim3 ngrid(NX/nthx.x, NY/nthx.y, NZ/nthx.z);
    int mem_size_scalar = NX*NY*NZ*sizeof(float);

    LB_clear_Forces(streams, NX, NY, NZ);
    checkCudaErrors(cudaDeviceSynchronize());
    LB_stream<<<ngrid, nthx>>>(f1_gpu, f2_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu, mass_gpu);
    
    LB_compute_local_params<<<ngrid, nthx>>>(f1_gpu, Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu);
    advect_velocity(num_threads, num_mesh, streams);
    cal_force_spread_RB(num_threads, num_mesh, Ct, streams);
    LB_add_gravity<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, cell_type_gpu, GRAV_CONST);
    
    LB_compute_local_params<<<ngrid, nthx>>>(f1_gpu, Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu);
    
    IF_stream_mass_transfer<<<ngrid, nthx>>>(f1_gpu, f2_gpu, rho_gpu, cell_type_gpu, mass_gpu, delMass);
    IF_cell_update_mass<<<ngrid, nthx>>>(rho_gpu, mass_gpu, delMass, cell_type_gpu);
    IF_update_cell_type<<<ngrid, nthx>>>(rho_gpu, mass_gpu, cell_type_gpu);
    IF_filled_nb_flag_update<<<ngrid, nthx>>>(cell_type_gpu);
    IF_filled_nb_DF_update<<<ngrid, nthx>>>(cell_type_gpu, f1_gpu, rho_gpu, ux_gpu, uy_gpu,uz_gpu, mass_gpu);
    IF_empty_nb_flag_update<<<ngrid, nthx>>>(cell_type_gpu);
    checkCudaErrors(cudaMemset((void*)(delMass), 0, mem_size_scalar));
    IF_distribute_excess_mass<<<ngrid, nthx>>>(mass_gpu, rho_gpu, cell_type_gpu, delMass);
    IF_collect_fluid_mass<<<ngrid, nthx>>>(mass_gpu, cell_type_gpu, delMass, rho_gpu);
    checkCudaErrors(cudaMemset ((void*)(delMass), 0, mem_size_scalar));

    LB_compute_equi_distribution<<<ngrid, nthx>>>(rho_gpu, ux_gpu, uy_gpu, uz_gpu, feq_gpu, cell_type_gpu);
    LB_compute_stress<<<ngrid, nthx>>>(source_term_gpu, f1_gpu, feq_gpu, strain_rate_gpu, cell_type_gpu);
    LB_compute_source_term<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, source_term_gpu, ux_gpu, uy_gpu, uz_gpu, strain_rate_gpu, cell_type_gpu);
    LB_collide<<<ngrid, nthx>>>(f1_gpu, f2_gpu, feq_gpu, source_term_gpu, strain_rate_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    
    printf("After update neighbours mass cells\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, cell_type_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_total_mass<<<ngrid, nthx>>>(mass_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    // checkCudaErrors(cudaMemcpy((void*)(temp_cell_type_gpu), (cell_type_gpu), mem_size_scalar,  cudaMemcpyDeviceToHost));

    copy_cell_type<<<ngrid, nthx>>>(temp_cell_type_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("\n");
}