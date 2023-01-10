#ifndef PBD_H_
#define PBD_H_

#include "glm/glm.hpp"
#include "Definitions.hpp"
#include "utilities.cuh"

class Softbody
{
private:
    void initPhysics(float edgeCompliance, float volCompliance);
    int numVerts;
    int numTets;
    int numEdge;
    Vertex *verts;
    Tetrahedral *tets;
    Edge *edge;

public:
    Softbody(Vertex *body, int numVert, Tetrahedral* tets, int numTets, Edge *edgeList, int numEdge, float edgeCompliance, float volCompliance);
    void preSolve(float dt, glm::f32vec3 force, glm::f32vec3 mod_scale);
    void solveVolumes(float dt);
    void SolveEdges(float dt);
    float getTetVolume(int tetId);
    void postSolve(float dt);
};

#endif