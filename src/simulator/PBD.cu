#include "PBD.cuh"

Softbody::Softbody(Vertex *body, int numVert, Edge *edgeList, int numEdge, float edgeCompliance, float volCompliance, float Cl)
{
    this->numVerts = numVert;
    this->verts = body;
    this->numEdge = numEdge;
    this->edge = edgeList;
    initPhysics(edgeCompliance, volCompliance, Cl);
}

void Softbody::initPhysics(float edgeCompliance, float volCompliance, float Cl)
{
    // for (int i = 0; i < this->numTets; i++)
    // {
    //     float vol = abs(this->getTetVolume(i));
    //     this->tets[i].V0 = vol;
    //     this->tets[i].k = volCompliance;
    //     float pInvMass = vol > 0.0 ? 1.0 / (vol / 4.0) : 0.0;
    //     this->verts[this->tets[i].vertID[0]].invMass += pInvMass;
    //     this->verts[this->tets[i].vertID[1]].invMass += pInvMass;
    //     this->verts[this->tets[i].vertID[2]].invMass += pInvMass;
    //     this->verts[this->tets[i].vertID[3]].invMass += pInvMass;
    // }

    for (int i = 0; i < this->numEdge; i++) 
    {
        glm::f32vec3 id0 = this->verts[this->edge[i].vertID[0]].Position;
        glm::f32vec3 id1 = this->verts[this->edge[i].vertID[1]].Position;
        this->edge[i].L0 = glm::length(id0-id1);
        this->edge[i].k = edgeCompliance;
        float vol = this->edge[i].L0/Cl;
        // printf(" %f ", vol);
        float pInvMass = vol > 0.0 ? 1.0 / (vol / 2.0f) : 0.0;
        this->verts[this->edge[i].vertID[0]].invMass += pInvMass;
        this->verts[this->edge[i].vertID[1]].invMass += pInvMass;
    }
}

float Softbody::getTetVolume(int tetId) 
{
    glm::f32vec3 id0 = this->verts[this->tets[tetId].vertID[0]].Position;
    glm::f32vec3 id1 = this->verts[this->tets[tetId].vertID[1]].Position;
    glm::f32vec3 id2 = this->verts[this->tets[tetId].vertID[2]].Position;
    glm::f32vec3 id3 = this->verts[this->tets[tetId].vertID[3]].Position;

    glm::f32vec3 t1 = id1-id0;
    glm::f32vec3 t2 = id2-id0;
    glm::f32vec3 t3 = id3-id0;
    float volume = glm::dot(t3, glm::cross(t1, t2))/6.0;
    return volume;
}

void Softbody::SolveEdges(float dt)
{
    // printf(" $$$ Total vertices %d  Total Edges %d  Total Tets %d \n",numVerts, numEdge, numTets);
    float alpha = 1.0f/(dt*dt);
    int count = 0;
    for(int i=0;i<this->numEdge;i++)
    {
        float w0 = this->verts[this->edge[i].vertID[0]].invMass;
        float w1 = this->verts[this->edge[i].vertID[1]].invMass;
        float w = w0 + w1;
        if(w == 0.0f)
            continue;
        glm::f32vec3 grads = this->verts[this->edge[i].vertID[0]].Position - this->verts[this->edge[i].vertID[1]].Position;
        float cur_length = glm::length(grads);
        if(cur_length==0.0f)
            continue;
        grads *= (1.0f/cur_length);
        float C = cur_length - edge[i].L0;
        float s = -C/(w + edge[i].k*alpha);

        this->verts[this->edge[i].vertID[0]].Position += grads*s*w0;
        this->verts[this->edge[i].vertID[1]].Position -= grads*s*w1;
    }
}

void Softbody::solveVolumes(float dt) 
{
    // printf(" ^^^ Total vertices %d  Total Edges %d  Total Tets %d \n", numVerts, numEdge, numTets);
    float alpha = 1.0f/(dt*dt);
    int count = 0;
    for (int i = 0; i < this->numTets; i++) 
    {
        float w = 0.0f;
        glm::f32vec3 grads[] = {glm::f32vec3(0.0f), glm::f32vec3(0.0f), glm::f32vec3(0.0f), glm::f32vec3(0.0f)};
        glm::f32vec3 id0, id1, id2, temp0, temp1;
        for (int j = 0; j < 4; j++) 
        {
            id0 = this->verts[this->tets[i].vertID[this->tets[i].face[j].x]].Position;
            id1 = this->verts[this->tets[i].vertID[this->tets[i].face[j].y]].Position;
            id2 = this->verts[this->tets[i].vertID[this->tets[i].face[j].z]].Position;

            temp0 = id1 - id0;
            temp1 = id2 - id0;
            grads[j] = (1.0f/6.0f)*glm::cross(temp0, temp1);
            w += (this->verts[this->tets[i].vertID[j]].invMass) * powf(glm::length(grads[j]), 2.0f);
        }

        if (w == 0.0f)
            continue;

        float vol = abs(this->getTetVolume(i));
        float restVol = this->tets[i].V0;
        float C = vol - restVol;
        float s = -C / (w + alpha*this->tets[i].k);

        for (int j = 0; j < 4; j++)
            this->verts[this->tets[i].vertID[j]].Position += grads[j]*s*this->verts[this->tets[i].vertID[j]].invMass;
    }
}


void Softbody::preSolve(float dt, glm::f32vec3 dimensions, float Ct, float Cl)
{
    bool flag  = false;
    // printf(" %f ", dt*(glm::length(glm::f32vec3(0.0f,GRAV_CONST, 0.0f)))*Cl/(Ct*Ct));
    for (int i = 0; i < this->numVerts; i++) 
    {
        if (this->verts[i].invMass == 0.0)
            continue;
        this->verts[i].Base_Velocity -= dt*(this->verts[i].Force*Cl/(1000*Ct*Ct) + glm::f32vec3(0.0f,9.8, 0.0f));//+
        this->verts[i].Prev_Position = this->verts[i].Position;
        this->verts[i].Position += dt*this->verts[i].Base_Velocity;
        float x = this->verts[i].Position.x;
        float y = this->verts[i].Position.y;
        float z = this->verts[i].Position.z;
        if (x < 1.0f)
            this->verts[i].Position.x = 1.0f;
        else if(x>dimensions.x-2)
            this->verts[i].Position.x = dimensions.x-2;
        if (y < 1.0f)
            this->verts[i].Position.y = 1.0f;
        else if(y>dimensions.y-2)
            this->verts[i].Position.y = dimensions.y-2;
        if (z < 1.0f)
            this->verts[i].Position.z = 1.0f;
        else if(z>dimensions.z-2)
            this->verts[i].Position.z = dimensions.z-2;
        this->verts[i].Force = glm::f32vec3(0.0f);
    }
}

void Softbody::postSolve(float dt)
{
         // printf(" *** Total vertices %d  Total Edges %d  Total Tets %d \n", numVerts, numEdge, numTets);
    for (int i = 0; i < this->numVerts; i++) 
    {
        if (this->verts[i].invMass == 0.0)
            continue;
        this->verts[i].Base_Velocity = (this->verts[i].Position - this->verts[i].Prev_Position)*(1.0f/dt);
    }
}