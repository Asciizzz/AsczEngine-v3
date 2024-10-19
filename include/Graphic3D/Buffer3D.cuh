#ifndef BUFFER3D_CUH
#define BUFFER3D_CUH

#include <Camera3D.cuh>
#include <Mesh3D.cuh>

class Buffer3D {
public:
    int width, height, size;
    int blockSize = 256;
    int blockCount;

    // Device pointers
    float *depth;
    Vec4f *color;
    Vec3f *world;
    Vec3f *normal;
    Vec2f *texture;
    UInt *meshID;

    Buffer3D();
    ~Buffer3D();
    void resize(int width, int height, int pixelSize=1);
    void free();

    void clearBuffer();
};

__global__ void clearBufferKernel(
    float *depth,
    Vec4f *color,
    Vec3f *world,
    Vec3f *normal,
    Vec2f *texture,
    UInt *meshID,
    int size
);

#endif