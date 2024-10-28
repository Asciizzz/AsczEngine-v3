#ifndef CUBE3D_CUH
#define CUBE3D_CUH

#include <Playground.cuh>

#include <FpsHandler.cuh>

class Cube3D {
public:
    Vec3f center;
    Vec3f world[8];

};

#endif