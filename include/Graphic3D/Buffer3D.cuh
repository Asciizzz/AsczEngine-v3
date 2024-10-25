#ifndef BUFFER3D_CUH
#define BUFFER3D_CUH

#include <Matrix.cuh>
#include <cuda_runtime.h>

/* Note:

Buffer is allowed to use AoS memory layout because it is
relatively small, restricted to the size of the screen.

Slight memory overhead is acceptable for the sake of
simplicity and readability.

*/

class Buffer3D {
public:
    int width, height, size;
    int blockSize = 256;
    int blockNum;

    // Device pointers
    bool *active;
    float *depth;
    ULLInt *faceID;
    Vecptr3f bary;

    Vecptr3f world;
    Vecptr2f texture;
    Vecptr3f normal;
    Vecptr4f color;


    Buffer3D();
    void resize(int width, int height, int pixelSize=1);
    void free();

    void clearBuffer();

    // Fun buffer functions
    void nightSky();
};

__global__ void clearBufferKernel(
    bool *active, float *depth, ULLInt *faceID,
    float *brx, float *bry, float *brz, // Bary
    float *wx, float *wy, float *wz, // World
    float *tu, float *tv, // Texture
    float *nx, float *ny, float *nz, // Normal
    float *cr, float *cg, float *cb, float *ca, // Color
    int size
);

// FUN BUFFER FUNCTIONS
__global__ void nightSkyKernel(
    float *cr, float *cg, float *cb, float *ca, // Color
    int width, int height
);

#endif