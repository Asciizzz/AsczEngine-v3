#ifndef BUFFER3D_CUH
#define BUFFER3D_CUH

#include <Mesh3D.cuh>

class Buffer3D {
public:
    float *depth;
    float *normal;
    float *color;
    float *world;

    Buffer3D(int width=0, int height=0);
};

#endif