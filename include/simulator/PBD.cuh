#ifndef PBD_H_
#define PBD_H_

#include "glm/glm.hpp"
#include "Definitions.hpp"
#include "utilities.cuh"
#include "Helper.cuh"

class Softbody
{
private:
    void initPhysics(float edgeCompliance, float volCompliance, float Cl);
    int numVerts;
    int numTets;
    int numEdge;
    Vertex *verts;
    Tetrahedral *tets;
    Edge *edge;

public:
    Softbody(Vertex *body, int numVert, Edge *edgeList, int numEdge, float edgeCompliance, float volCompliance, float Cl);
    void preSolve(float dt, glm::f32vec3 mod_scale, float Ct, float Cl);
    void solveVolumes(float dt);
    void SolveEdges(float dt);
    float getTetVolume(int tetId);
    void postSolve(float dt);
};

#endif