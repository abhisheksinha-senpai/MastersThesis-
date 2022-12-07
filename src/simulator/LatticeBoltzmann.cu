#include "LatticeBoltzmann.cuh"

__device__ const float RHO_ATM = 0.0001f;
__device__ const float GRAV_CONST = 0.01f;
__device__ const float SMAGRINSKY_CONST = 0.01;
__device__ const float FILL_OFFSET = 0.001f;
__device__ const float LONELY_THRESH = 0.1f;

__device__ float cs_inv_sq = 3.0f;
__device__ float non_dim_tau = 1.5f;
__device__ float non_dim_nu;


__device__ float w[19] = {1.0f/3.0f, 1.0f/18.0f,  1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

__device__ int cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0};
__device__ int cy[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1};
__device__ int cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1};

__device__ int finv[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

__device__ int lb_sim_domain[3];

__device__ float max_ux = -9999.0f;
__device__ float max_uy = -9999.0f;
__device__ float max_uz = -9999.0f;
__device__ float max_rho = -9999.0f;
__device__ float max_Fx = -9999.0f;
__device__ float max_Fy = -9999.0f;
__device__ float max_Fz = -9999.0f;
__device__ float max_mass = -9999.0f;

__device__ unsigned int FLUID = (unsigned int)(1 << 0);
__device__ unsigned int  INTERFACE  = (unsigned int)(1 << 1);
__device__ unsigned int  EMPTY =  (unsigned int)(1 << 2);
__device__ unsigned int  OBSTACLE =  (unsigned int)(1 << 3);
__device__ unsigned int  NO_FLUID_NEIGH =  (unsigned int)(1 << 4);
__device__ unsigned int  NO_EMPTY_NEIGH =  (unsigned int)(1 << 5);
__device__ unsigned int  NO_IFACE_NEIGH =  (unsigned int)(1 << 6);
__device__ unsigned int  IF_TO_FLUID =  (unsigned int)(1 << 7);
__device__ unsigned int  IF_TO_EMPTY =  (unsigned int)(1 << 8);


float Ct;
float Cl;

float *count_loc;

float *Fx_gpu, *Fy_gpu, *Fz_gpu;
float *rho_gpu, *ux_gpu, *uy_gpu, *uz_gpu;
float *f1_gpu, *f2_gpu, *feq_gpu, *source_term_gpu;
float *mass_gpu, *strain_rate_gpu, *delMass;
float *empty_filled_cell;
//cell Type: 0-fluid, 1-interface, 2-gas, 3-obstacle
unsigned int *cell_type_gpu, *temp_cell_type_gpu;

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

__device__ float calculate_gas_DF(float rho, float ux, float uy, float uz, int i)
{
    float u_dot_c, u_dot_u = ux * ux + uy * uy + uz * uz;
    u_dot_c= cx[i] * ux + cy[i] * uy + cz[i] * uz;
    float feq = w[i]*rho*(1+cs_inv_sq*(u_dot_c +0.5*cs_inv_sq*powf(u_dot_c, 2.0f) - 0.5*u_dot_u));
    return feq;
}

__device__ int calculate_mass_fraction(float rho, float mass, int type)
{
    if (type & (FLUID | OBSTACLE))
		return 1;
    else if (type & INTERFACE)
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
    
    norm.x= 0.5f*(calculate_mass_fraction(mass_gpu[xm], rho_gpu[xm], temp_cell_type_gpu[xm]) - calculate_mass_fraction(mass_gpu[xp], rho_gpu[xp], temp_cell_type_gpu[xp]));
    norm.y = 0.5f*(calculate_mass_fraction(mass_gpu[ym], rho_gpu[ym], temp_cell_type_gpu[ym]) - calculate_mass_fraction(mass_gpu[yp], rho_gpu[yp], temp_cell_type_gpu[yp]));
    norm.z =0.5f*(calculate_mass_fraction(mass_gpu[zm], rho_gpu[zm], temp_cell_type_gpu[zm]) - calculate_mass_fraction(mass_gpu[zp], rho_gpu[zp], temp_cell_type_gpu[zp]));

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
        mass_gpu[x_] = rho_gpu[x_];
    }
    else if(rho_gpu[x_]<=RHO_ATM)
    {
        cell_type_gpu[x_] = EMPTY;
        mass_gpu[x_] = 0.0f;
    }
    else if(rho_gpu[x_]==1.0f)
    {
        cell_type_gpu[x_] = FLUID;
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
    if(cell_type_gpu[x_] & FLUID)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index((idx+NX+cx[i])%NX, (idy+NY+cy[i])%NY, (idz+NZ+cz[i])%NZ, lb_sim_domain);
            if(!(cell_type_gpu[nb] & OBSTACLE) && (cell_type_gpu[nb] & EMPTY))
            {
                cell_type_gpu[nb] = INTERFACE;
                rho_gpu[nb] = 0.1f;
                mass_gpu[nb] = 0.1f;
                
            }
        }
    }
}

__global__ void update_cell_fill_state(float *rho_gpu, float *mass_gpu, float *delMass, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    
    if(!(temp_cell_type_gpu[x_] & OBSTACLE))
        mass_gpu[x_] += delMass[x_];
}

__global__ void update_fluid_cell_type(float *rho_gpu, float *mass_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(temp_cell_type_gpu[x_] & (INTERFACE))
    {
        if((mass_gpu[x_]>=(1.0f + FILL_OFFSET)*rho_gpu[x_]) || (mass_gpu[x_]>=(1-LONELY_THRESH)*rho_gpu[x_] && (temp_cell_type_gpu[x_] & NO_EMPTY_NEIGH)))
        {
            //printf("%d %d %d ", idx, idy, idz);
            temp_cell_type_gpu[x_] = IF_TO_FLUID;
        }
        else if((mass_gpu[x_]<=(0.0f - FILL_OFFSET)*rho_gpu[x_]) || (mass_gpu[x_]<=(LONELY_THRESH)*rho_gpu[x_] && (temp_cell_type_gpu[x_] & NO_FLUID_NEIGH)) || ((temp_cell_type_gpu[x_] & NO_IFACE_NEIGH) && (temp_cell_type_gpu[x_] & NO_FLUID_NEIGH)))
        {
            //printf("%d %d %d ", idx, idy, idz);
            temp_cell_type_gpu[x_] = IF_TO_EMPTY;
        }
    }
    temp_cell_type_gpu[x_] &= ~(NO_FLUID_NEIGH | NO_EMPTY_NEIGH | NO_IFACE_NEIGH);
}

__device__ void initialize_empty_nb_DF(int idx, int idy, int idz, float *f1_gpu, float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, unsigned int *temp_cell_type_gpu, float *mass_gpu)
{
    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];
    int nb, count_nb=0;
    float ux=0.0f, uy=0.0f, uz=0.0f, rho=0.0f, mass=0.0f;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    for(int i=1;i<19;i++)
    {
        nb = gpu_scalar_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, lb_sim_domain);
        if(temp_cell_type_gpu[nb] &(INTERFACE|FLUID))
        {
            ux += ux_gpu[nb];
            uy += uy_gpu[nb];
            uz += uz_gpu[nb];
            rho += rho_gpu[nb];
            //mass += mass_gpu[nb];
            count_nb+=1;
        }
    }
    if(count_nb>0)
    {
        rho = rho/count_nb;
        ux = ux/count_nb;
        uy = uy/count_nb;
        uz = uz/count_nb;
        mass = mass/count_nb;
    }
    for(int i=0;i<19;i++)
        f1_gpu[gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain)] = calculate_gas_DF(rho, ux, uy, uz, i);
    ux_gpu[x_] = ux;
    uy_gpu[x_] = uy;
    uz_gpu[x_] = uz;
    rho_gpu[x_] = rho;
    //mass_gpu[x_] = mass;
}

__global__ void update_filled_cell_nb_DF(unsigned int *temp_cell_type_gpu, float *f1_gpu, float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, float *mass_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb, nbb;
    if(temp_cell_type_gpu[x_] & IF_TO_FLUID)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, lb_sim_domain);
            if(temp_cell_type_gpu[nb] & EMPTY)
            {
                //printf("f(%d %d %d--%d %d %d)  ",idx, idy, idz, (idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ );
                initialize_empty_nb_DF((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, f1_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, temp_cell_type_gpu, mass_gpu);
                temp_cell_type_gpu[nb] = INTERFACE;
                //printf("f(%d %d %d--%d %d %d)",idx, idy, idz, (idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ );
            }
        }
        for (int i = 1; i < 19; i++)		// omit zero vector
        {
            nb = gpu_scalar_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, lb_sim_domain);
            if(temp_cell_type_gpu[nb] & IF_TO_EMPTY)
            {
                //printf("e(%d %d %d--%d %d %d)  ",idx, idy, idz, (idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ );
                temp_cell_type_gpu[nb] = INTERFACE;
            }
        }
    }
}

__global__ void update_empty_cells_nb(unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int nb;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(temp_cell_type_gpu[x_] & IF_TO_EMPTY)
    {
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, lb_sim_domain);
            if(temp_cell_type_gpu[nb] & FLUID)
                temp_cell_type_gpu[nb]  = INTERFACE;
        }
    }
}

__global__ void distribute_excess_mass(float *mass_gpu, float *rho_gpu, unsigned int *temp_cell_type_gpu, float *delMass)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int nb;
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    glm::f32vec3 norm = calculate_normal(idx, idy, idz, rho_gpu, mass_gpu, temp_cell_type_gpu);
    float mass_ex=0.0f;
    if(temp_cell_type_gpu[x_] & IF_TO_FLUID)
    {
        mass_ex = mass_gpu[x_]-rho_gpu[x_];
        mass_gpu[x_] = rho_gpu[x_];
    }
    else if(temp_cell_type_gpu[x_] & IF_TO_EMPTY)
    {
        mass_ex = mass_gpu[x_];
        mass_gpu[x_] = 0.0f;
        norm = -1.0f*norm;
    }
    if(temp_cell_type_gpu[x_] & (IF_TO_EMPTY|IF_TO_FLUID))
    {
        float ni_total=0;
        float delM = 0.0f;
        float ni[19] = { 0 };
        unsigned int isIF[19] = { 0 };
        unsigned int numIF = 0;
        for (int i = 1; i < 19; i++)
        {
            nb = gpu_scalar_index((idx+NX+cx[i])%NX, (idy+NY+cy[i])%NY, (idz+NZ+cz[i])%NZ, lb_sim_domain);
            if(temp_cell_type_gpu[nb] & (INTERFACE))
            {
                ni[i] = norm.x*cx[i] + norm.y*cy[i] + norm.z*cz[i];
                if(ni[i]<0.0f)
                    ni[i] = 0.0f;
                ni_total += ni[i];
                isIF[i] = 1;
                numIF++;
            }
        }
        if (ni_total > 0)
        {
            float ni_fac = 1/ni_total;
            for (int i = 1; i < 19; i++)		// omit zero vector
            {
                nb = gpu_scalar_index((idx+NX+cx[i])%NX, (idy+NY+cy[i])%NY, (idz+NZ+cz[i])%NZ, lb_sim_domain);
                atomicAdd(&delMass[nb], mass_ex * ni[i]*ni_fac);
                //printf("a %f  ",  mass_ex * ni[i]*ni_fac);
            }
        }
        else if (numIF > 0)
        {
            // distribute uniformly
            float mex_rel = mass_ex / numIF;
            for (int i = 1; i < 19; i++)		// omit zero vector
            {
                nb = gpu_scalar_index((idx+NX+cx[i])%NX, (idy+NY+cy[i])%NY, (idz+NZ+cz[i])%NZ, lb_sim_domain);
                float val = isIF[i] ? mex_rel : 0;
                atomicAdd(&delMass[nb], val);
                //printf("b %f  ", val);
            }
        }
    }
}

__global__ void collect_fluid_mass(float *mass_gpu, unsigned int *temp_cell_type_gpu, float *delMass)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb;
    if(temp_cell_type_gpu[x_] & INTERFACE)
    {
        mass_gpu[x_] += delMass[x_];
    }
    if(temp_cell_type_gpu[x_] & IF_TO_FLUID)
        temp_cell_type_gpu[x_] = FLUID;
    else if(temp_cell_type_gpu[x_] & IF_TO_EMPTY)
        temp_cell_type_gpu[x_] = EMPTY;
}

__global__ void update_fluid_neigh(unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int NX = lb_sim_domain[0];
    int NY = lb_sim_domain[1];
    int NZ = lb_sim_domain[2];

    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    int nb;

    if(!(temp_cell_type_gpu[x_] & OBSTACLE))
    {
        temp_cell_type_gpu[x_] |= (NO_FLUID_NEIGH | NO_EMPTY_NEIGH | NO_IFACE_NEIGH);
        for(int i=1;i<19;i++)
        {
            nb = gpu_scalar_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, lb_sim_domain);
            //printf("B %fb", temp_cell_type_gpu[x_]);
            if(temp_cell_type_gpu[nb] & FLUID)
            {
                temp_cell_type_gpu[x_] = (temp_cell_type_gpu[x_] & (~NO_FLUID_NEIGH));

            }
            else if(temp_cell_type_gpu[nb] & EMPTY)
            {
                temp_cell_type_gpu[x_] = (temp_cell_type_gpu[x_] & (~NO_EMPTY_NEIGH));
            }
            else if(temp_cell_type_gpu[nb] & INTERFACE)
            {
                temp_cell_type_gpu[x_] = (temp_cell_type_gpu[x_] & (~NO_IFACE_NEIGH));
            }
        }
        if (temp_cell_type_gpu[x_] & NO_EMPTY_NEIGH)
            temp_cell_type_gpu[x_] = temp_cell_type_gpu[x_] & (~NO_FLUID_NEIGH);
    }
}

__global__ void LB_compute_local_params(float *f1_gpu, float *Fx_gpu, float *Fy_gpu, float* Fz_gpu, 
                                        float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu,
                                        unsigned int *temp_cell_type_gpu, float *mass_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float lat_rho=0.000f, lat_rho_ux=0.000f, lat_rho_uy=0.000f, lat_rho_uz=0.000f;

    float f_val = 0.000f;
    float cs = 1.0f/sqrt(cs_inv_sq);
    int x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[x_] & (EMPTY|OBSTACLE)))
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
        x_ = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
        
        rho_gpu[x_] = lat_rho;
        float rho_inv = (lat_rho <= 0.0f?0.0f:(1.0f/lat_rho));

        float newU = (lat_rho_ux + 0.5f * Fx_gpu[x_]) * rho_inv;
        float newV = (lat_rho_uy + 0.5f * Fy_gpu[x_]) * rho_inv;
        float newW = (lat_rho_uz + 0.5f * Fz_gpu[x_]) * rho_inv;

        float v_max = 0.816496580927726f;
        float nU= glm::length(glm::f32vec3(newU, newV, newW));
        if(nU>v_max)
        {
            newU = newU*(v_max/nU);
            newV = newV*(v_max/nU);
            newW = newW*(v_max/nU);
        }

        ux_gpu[x_] = newU;
        uy_gpu[x_] = newV;
        uz_gpu[x_] = newW;
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
    if(!(temp_cell_type_gpu[coord] & (OBSTACLE|EMPTY)))
    {
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
}

__global__ void LB_compute_source_term(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *source_term_gpu,
                                        float *ux_gpu, float *uy_gpu, float *uz_gpu, float *strain_rate_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float ci_dot_u = 0.0f, ci_dot_F = 0.0f;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] & (OBSTACLE|EMPTY)))
    {
        float lat_ux = ux_gpu[coord];
        float lat_uy = uy_gpu[coord];
        float lat_uz = uz_gpu[coord];

        // float quant = powf(non_dim_nu, 2.0f)+18.0f*SMAGRINSKY_CONST*SMAGRINSKY_CONST*sqrt(strain_rate_gpu[coord]);
        // float S = (1.0f/(6.0f*powf(SMAGRINSKY_CONST, 2.0f)))*(sqrt(quant) - non_dim_nu);
        // float tau = cs_inv_sq * (non_dim_nu + SMAGRINSKY_CONST*SMAGRINSKY_CONST*S)+0.5f;
        float tau = non_dim_tau + 0.5f;
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
    if(!(temp_cell_type_gpu[x_] & (OBSTACLE|EMPTY)))
    {
        lat_rho =  rho_gpu[x_];
        float rho_inv = (lat_rho <= 0.0f)?0.0f:(1.0f/lat_rho);
        lat_ux = ux_gpu[x_] - 0.5f*rho_inv*Fx_gpu[x_];
        lat_uy = uy_gpu[x_] - 0.5f*rho_inv*Fy_gpu[x_];
        lat_uz = uz_gpu[x_] - 0.5f*rho_inv*Fz_gpu[x_];
        ux_gpu[x_] = lat_ux;
        uy_gpu[x_] = lat_uy;
        uz_gpu[x_] = lat_uz;

        float u_dot_c, u_dot_u = lat_ux * lat_ux + lat_uy * lat_uy + lat_uz * lat_uz;
        float val = 0;
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
    if(!(temp_cell_type_gpu[coord] & (OBSTACLE|EMPTY)))
    {
        float omega, source;
        // float quant = powf(non_dim_nu, 2.0f)+18.0f*SMAGRINSKY_CONST*SMAGRINSKY_CONST*sqrt(strain_rate_gpu[coord]);
        // float S = (1.0f/(6.0f*powf(SMAGRINSKY_CONST, 2.0f)))*(sqrt(quant) - non_dim_nu);
        // float tau = cs_inv_sq * (non_dim_nu + SMAGRINSKY_CONST*SMAGRINSKY_CONST*S)+0.5f;
        float tau = non_dim_tau +0.5f;
        float tau_inv = (1.0f/tau);
        for(int i =0;i<19;i++)
        {
            coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
            omega = -1.0f*tau_inv*(f1_gpu[coord] - feq_gpu[coord]);
            source = source_term_gpu[coord];
            f2_gpu[coord] = f1_gpu[coord]+(omega + source);
        }
    }
}

__global__ void LB_stream(float *f1_gpu, float *f2_gpu, float *mass_gpu, float *rho_gpu, float *ux_gpu, float *uy_gpu, float *uz_gpu, unsigned int *temp_cell_type_gpu, float *delMass)
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
    //delMass[x_] =0.0f;
    if(!(temp_cell_type_gpu[x_] & (OBSTACLE|EMPTY)))
    {
        f1_gpu[gpu_fieldn_index(idx, idy, idz, 0, lb_sim_domain)] =  f2_gpu[gpu_fieldn_index(idx, idy, idz, 0, lb_sim_domain)];
        if(temp_cell_type_gpu[x_] & FLUID)
        {
            //mass_gpu[x_] = rho_gpu[x_];
            for(int i=1;i<19;i++)
            {
                nb = gpu_scalar_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ,lb_sim_domain);
                coordNB = gpu_fieldn_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, i, lb_sim_domain);
                coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
                int coord_inv = gpu_fieldn_index(idx, idy, idz, finv[i], lb_sim_domain);
                if(temp_cell_type_gpu[nb] & (FLUID|INTERFACE))
                {
                    delMass[x_] +=  (f2_gpu[coordNB] - f2_gpu[coord_inv]);
                    f1_gpu[coord] =  f2_gpu[coordNB];
                }
                else if(temp_cell_type_gpu[nb] & (OBSTACLE))
                    f1_gpu[coord] =  f2_gpu[coord_inv];
            }
        }
        else if(temp_cell_type_gpu[x_] & INTERFACE)
        {
            float lat_ux = ux_gpu[x_];
            float lat_uy = uy_gpu[x_];
            float lat_uz = uz_gpu[x_];
            float ei_dot_n;
            for(int i=1;i<19;i++)
            {
                nb = gpu_scalar_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ,lb_sim_domain);
                coordNB = gpu_fieldn_index((idx+NX+cx[finv[i]])%NX, (idy+NY+cy[finv[i]])%NY, (idz+NZ+cz[finv[i]])%NZ, i, lb_sim_domain);
                coord = gpu_fieldn_index(idx, idy, idz, i, lb_sim_domain);
                int coord_inv = gpu_fieldn_index(idx, idy, idz, finv[i], lb_sim_domain);
                if(temp_cell_type_gpu[nb] & FLUID)
                {
                    delMass[x_] += (f2_gpu[coordNB] - calculate_mass_fraction(rho_gpu[x_], mass_gpu[x_], temp_cell_type_gpu[x_])*f2_gpu[coord_inv]);
                    f1_gpu[coord] =  f2_gpu[coordNB];
                }
                else if(temp_cell_type_gpu[nb] & INTERFACE)
                {
                    delMass[x_] += calculate_mass_fraction(rho_gpu[nb], mass_gpu[nb], temp_cell_type_gpu[nb])*f2_gpu[coordNB]-calculate_mass_fraction(rho_gpu[x_], mass_gpu[x_], temp_cell_type_gpu[x_])*f2_gpu[coord_inv];
                    f1_gpu[coord] =  f2_gpu[coordNB];
                }
                else if(temp_cell_type_gpu[nb] & EMPTY)
                    f1_gpu[coord] = (calculate_gas_DF(RHO_ATM, lat_ux, lat_uy, lat_uz, i) + calculate_gas_DF(RHO_ATM, lat_ux, lat_uy, lat_uz, finv[i])) - f2_gpu[coord_inv];
                else if(temp_cell_type_gpu[nb] & OBSTACLE)
                    f1_gpu[coord] =  f2_gpu[coord_inv];
                
                glm::f32vec3 norm = calculate_normal(idx, idy, idz, rho_gpu, mass_gpu, temp_cell_type_gpu);
                for(int i=1;i<19;i++)
                {
                    ei_dot_n = norm.x*cx[finv[i]]+norm.y*cy[finv[i]]+norm.z*cz[finv[i]];
                    if(ei_dot_n>0)
                        f1_gpu[coord] = (calculate_gas_DF(RHO_ATM, lat_ux, lat_uy, lat_uz, i) + calculate_gas_DF(RHO_ATM, lat_ux, lat_uy, lat_uz, finv[i])) - f2_gpu[coord_inv];
                }
            }
        }
    }
}

__global__ void check_max_params()
{
    printf("Max values of params = %f %f %f %f\t%f %f %f\t%f\n", max_ux, max_uy, max_uz, max_rho, max_Fx, max_Fy, max_Fz, max_mass);
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
    atomicMax(&max_rho, (rho_gpu[coord]));
    atomicMax(&max_Fx, abs(Fx_gpu[coord]));
    atomicMax(&max_Fy, abs(Fy_gpu[coord]));
    atomicMax(&max_Fz, abs(Fz_gpu[coord]));
}

__global__ void update_max_mass(float *mass_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    atomicMax(&max_mass, mass_gpu[coord]);
}

__global__ void LB_compute_stress(float *source_term_gpu, float *f1_gpu, float *feq_gpu, float *strain_rate_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(temp_cell_type_gpu[coord] <2)
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

__global__ void LB_add_gravity(float *Fx_gpu, float *Fy_gpu, float *Fz_gpu, float *rho_gpu, unsigned int *temp_cell_type_gpu)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    float domain_size = 2.0f;
    float Cl = domain_size/max(max(lb_sim_domain[0], lb_sim_domain[1]), lb_sim_domain[2]);
    int coord = gpu_scalar_index(idx, idy, idz, lb_sim_domain);
    if(!(temp_cell_type_gpu[coord] & (OBSTACLE)))
        atomicAdd(&Fy_gpu[coord], -1.0f * GRAV_CONST * rho_gpu[coord]);
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

    // checkCudaErrors(cudaMemsetAsync ((void*)(Fx_gpu), 0, total_lattice_points*sizeof(float), streams[0]));
    // checkCudaErrors(cudaMemsetAsync ((void*)(Fy_gpu), 0, total_lattice_points*sizeof(float), streams[1]));
    // checkCudaErrors(cudaMemsetAsync ((void*)(Fz_gpu), 0, total_lattice_points*sizeof(float), streams[2]));
    // checkCudaErrors(cudaMemsetAsync ((void*)(source_term_gpu), 0, 19*total_lattice_points*sizeof(float), streams[3]));
    // checkCudaErrors(cudaMemsetAsync ((void*)(count_loc), 0, total_lattice_points*sizeof(int), streams[4]));
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void LB_reset_max(cudaStream_t *streams)
{
    float val_1 = -9999.0f;
    checkCudaErrors(cudaMemcpyToSymbol(max_rho, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_ux, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_uy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_uz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_Fx, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_Fy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_Fz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(max_mass, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaMemcpyToSymbolAsync(max_rho, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[0]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(max_ux, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[1]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(max_uy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[2]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(max_uz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[3]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(max_Fx, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[4]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(max_Fy, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[5]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(max_Fz, &val_1, sizeof(float), 0, cudaMemcpyHostToDevice, streams[6]));
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void LB_compute_sim_param(int NX, int NY, int NZ, float viscosity, float Re)
{
    float domain_size = 2.0;
    float cs_inv_sq = 3.0f;
    float lat_l_no_dim = max(max(NX, NY), NZ);
    
    float delX = domain_size/lat_l_no_dim;
    float deltaT = sqrt(GRAV_CONST*delX/9.8);
    float nu_star = viscosity*(deltaT/(delX*delX));
    float tau_star = cs_inv_sq*(nu_star) +0.5f;

    checkCudaErrors(cudaMemcpyToSymbol(non_dim_tau, &tau_star, sizeof(float), 0, cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpyToSymbol(non_dim_nu, &nu_star, sizeof(float), 0, cudaMemcpyHostToDevice));

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
    checkCudaErrors(cudaMalloc((void**)&temp_cell_type_gpu, total_lattice_points*sizeof(unsigned int)));
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

    // checkCudaErrors(cudaMemcpyToSymbolAsync(lb_sim_domain, &nu, sizeof(int), 0, cudaMemcpyHostToDevice, streams[0]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(lb_sim_domain, &nv, sizeof(int), sizeof(int), cudaMemcpyHostToDevice, streams[1]));
    // checkCudaErrors(cudaMemcpyToSymbolAsync(lb_sim_domain, &nw, sizeof(int), 2*sizeof(int), cudaMemcpyHostToDevice, streams[2]));
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
    

    // checkCudaErrors(cudaMemcpyAsync((void*)(rho_gpu), (*rho), mem_size_scalar,  cudaMemcpyHostToDevice, streams[5]));
    // checkCudaErrors(cudaMemcpyAsync((void*)(ux_gpu), (*ux), mem_size_scalar,  cudaMemcpyHostToDevice, streams[6]));
    // checkCudaErrors(cudaMemcpyAsync((void*)(uy_gpu), (*uy), mem_size_scalar,  cudaMemcpyHostToDevice, streams[7]));
    // checkCudaErrors(cudaMemcpyAsync((void*)(uz_gpu), (*uz), mem_size_scalar,  cudaMemcpyHostToDevice, streams[8]));

    // checkCudaErrors(cudaMemsetAsync ((void*)(mass_gpu), 0, mem_size_scalar, streams[14]));

    // checkCudaErrors(cudaMemsetAsync ((void*)(f1_gpu), 0, mem_size_ndir, streams[9]));
    // checkCudaErrors(cudaMemsetAsync ((void*)(f2_gpu), 0, mem_size_ndir, streams[10]));
    // checkCudaErrors(cudaMemsetAsync ((void*)(feq_gpu), 0, mem_size_ndir, streams[11]));
    // checkCudaErrors(cudaMemsetAsync ((void*)(source_term_gpu), 0, mem_size_ndir, streams[12]));
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
    //checkCudaErrors(cudaMemcpy((void*)(temp_cell_type_gpu), (cell_type_gpu), NX*NY*NZ*sizeof(int),  cudaMemcpyDeviceToDevice ));
    checkCudaErrors(cudaDeviceSynchronize());
    
    LB_reset_max(streams);
    LB_clear_Forces(streams, NX, NY, NZ);
    LB_add_gravity<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, cell_type_gpu);
    LB_equi_Initialization<<<ngrid, nthx>>>(f1_gpu, feq_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, Fx_gpu, Fy_gpu, Fz_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
#ifdef DEBUG
    printf("At start of simulations.......\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());
#endif

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

#ifdef DEBUG
__host__ void LB_simulate_RB(int NX, int NY, int NZ, float Ct,
                             void (*cal_force_spread_RB)(int, int, float, cudaStream_t *), 
                             void (*advect_velocity)(int, int, cudaStream_t *),
                             int num_threads, int num_mesh, cudaStream_t *streams)
{
    dim3 nthx(nThreads.x, nThreads.y, nThreads.z);
    dim3 ngrid(NX/nthx.x, NY/nthx.y, NZ/nthx.z);
    
    LB_reset_max(streams);
    LB_clear_Forces(streams, NX, NY, NZ);
    checkCudaErrors(cudaDeviceSynchronize());

    printf("Time step beings here.......\n");
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_add_gravity<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After adding gravity\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_compute_local_params<<<ngrid, nthx>>>(f1_gpu, Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu, mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After local params\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_compute_equi_distribution<<<ngrid, nthx>>>(rho_gpu, ux_gpu, uy_gpu, uz_gpu, feq_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After equi dist\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_compute_source_term<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, source_term_gpu, ux_gpu, uy_gpu, uz_gpu, strain_rate_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After source terms\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        LB_collide<<<ngrid, nthx>>>(f1_gpu, f2_gpu, feq_gpu, source_term_gpu, strain_rate_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After collision\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    //cudaMemset((void*)(delMass), 0,  NX*NY*NZ*sizeof(float));
    //checkCudaErrors(cudaDeviceSynchronize());
        LB_stream<<<ngrid, nthx>>>(f1_gpu, f2_gpu, mass_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu, delMass);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After stream\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        update_cell_fill_state<<<ngrid, nthx>>>(rho_gpu, mass_gpu, delMass, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After update update mass cells\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        update_fluid_cell_type<<<ngrid, nthx>>>(rho_gpu, mass_gpu, cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After update fuild cell type\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    //checkCudaErrors(cudaMemcpy((void*)(temp_cell_type_gpu), (cell_type_gpu), NX*NY*NZ*sizeof(int),  cudaMemcpyDeviceToDevice ));
        update_filled_cell_nb_DF<<<ngrid, nthx>>>(cell_type_gpu, f1_gpu, rho_gpu, ux_gpu, uy_gpu,uz_gpu, mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After update filled cell neighbour\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    //checkCudaErrors(cudaMemcpy((void*)(temp_cell_type_gpu), (cell_type_gpu), NX*NY*NZ*sizeof(int),  cudaMemcpyDeviceToDevice ));
        update_empty_cells_nb<<<ngrid, nthx>>>(cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After update empty cell neighbour\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    // checkCudaErrors(cudaMemcpy((void*)(temp_cell_type_gpu), (cell_type_gpu), NX*NY*NZ*sizeof(int),  cudaMemcpyDeviceToDevice ));
    checkCudaErrors(cudaMemset((void*)(delMass), 0,  NX*NY*NZ*sizeof(float)));
        distribute_excess_mass<<<ngrid, nthx>>>(mass_gpu, rho_gpu, cell_type_gpu, delMass);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After distribute excess mass\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        collect_fluid_mass<<<ngrid, nthx>>>(mass_gpu, cell_type_gpu, delMass);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After distribute excess mass\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

        update_fluid_neigh<<<ngrid, nthx>>>(cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("After update neighbours mass cells\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());
    printf("\n\n\n");
}

#else
__host__ void LB_simulate_RB(int NX, int NY, int NZ, float Ct,
    void (*cal_force_spread_RB)(int, int, float, cudaStream_t *), 
    void (*advect_velocity)(int, int, cudaStream_t *),
    int num_threads, int num_mesh, cudaStream_t *streams)
{
    dim3 nthx(nThreads.x, nThreads.y, nThreads.z);
    dim3 ngrid(NX/nthx.x, NY/nthx.y, NZ/nthx.z);

    LB_reset_max(streams);
    LB_clear_Forces(streams, NX, NY, NZ);
    checkCudaErrors(cudaDeviceSynchronize());

    LB_add_gravity<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, cell_type_gpu);
    LB_compute_local_params<<<ngrid, nthx>>>(f1_gpu, Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu, mass_gpu);
    advect_velocity(num_threads, num_mesh, streams);
    cal_force_spread_RB(num_threads, num_mesh, Ct, streams);
    LB_compute_local_params<<<ngrid, nthx>>>(f1_gpu, Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu, mass_gpu);
    LB_compute_equi_distribution<<<ngrid, nthx>>>(rho_gpu, ux_gpu, uy_gpu, uz_gpu, feq_gpu, cell_type_gpu);
    LB_compute_source_term<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, source_term_gpu, ux_gpu, uy_gpu, uz_gpu, strain_rate_gpu, cell_type_gpu);
    LB_collide<<<ngrid, nthx>>>(f1_gpu, f2_gpu, feq_gpu, source_term_gpu, strain_rate_gpu, cell_type_gpu);
    LB_stream<<<ngrid, nthx>>>(f1_gpu, f2_gpu, mass_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu, cell_type_gpu, delMass);
    update_cell_fill_state<<<ngrid, nthx>>>(rho_gpu, mass_gpu, delMass, cell_type_gpu);
    update_fluid_cell_type<<<ngrid, nthx>>>(rho_gpu, mass_gpu, cell_type_gpu);
    update_filled_cell_nb_DF<<<ngrid, nthx>>>(cell_type_gpu, f1_gpu, rho_gpu, ux_gpu, uy_gpu,uz_gpu, mass_gpu);
    update_empty_cells_nb<<<ngrid, nthx>>>(cell_type_gpu);
    checkCudaErrors(cudaMemset((void*)(delMass), 0,  NX*NY*NZ*sizeof(float)));
    distribute_excess_mass<<<ngrid, nthx>>>(mass_gpu, rho_gpu, cell_type_gpu, delMass);
    collect_fluid_mass<<<ngrid, nthx>>>(mass_gpu, cell_type_gpu, delMass);
    update_fluid_neigh<<<ngrid, nthx>>>(cell_type_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    
    printf("After update neighbours mass cells\n");
    LB_reset_max(streams);
    update_max_params<<<ngrid, nthx>>>(Fx_gpu, Fy_gpu, Fz_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    update_max_mass<<<ngrid, nthx>>>(mass_gpu);
    checkCudaErrors(cudaDeviceSynchronize());
    check_max_params<<<1,1>>>();
    checkCudaErrors(cudaDeviceSynchronize());
    printf("\n");
}
#endif