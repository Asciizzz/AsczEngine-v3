#ifndef BUFFER3D_CUH
#define BUFFER3D_CUH

#include <Camera3D.cuh>
#include <Mesh3D.cuh>

class Buffer3D {
public:
    int buffWidth, buffHeight, buffSize;

    // Device pointers
    float *depth;
    Vec4f *color;
    Vec3f *world;
    Vec3f *normal;
    Vec2f *texture;

    Buffer3D();
    void resize(int width, int height, int pixelSize=1);
    void free();
};

#endif