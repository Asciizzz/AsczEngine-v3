#ifndef BUFFER3D_CUH
#define BUFFER3D_CUH

#include <Matrix.cuh>
#include <cuda_runtime.h>

class Buffer3D {
public:
    int width, height, size;
    int blockSize = 256;
    int blockNum;

    // Device pointers
    bool *active;
    float *depth;
    Vec4f *color;

    Vec3f *world;
    Vec3f *normal;
    Vec2f *texture;

    ULLInt *faceID;
    Vec3f *bary;

    Buffer3D();
    void resize(int width, int height, int pixelSize=1);
    void free();

    void clearBuffer();

    // Fun buffer functions
    void nightSky();
};

__global__ void clearBufferKernel(
    bool *active, float *depth, Vec4f *color,
    Vec3f *world, Vec3f *normal, Vec2f *texture,
    ULLInt *faceID, Vec3f *bary, int size
);

// FUN BUFFER FUNCTIONS
__global__ void nightSkyKernel(Vec4f *color, int width, int height);

#endif