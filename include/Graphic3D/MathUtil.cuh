#ifndef MATHUTIL_CUH
#define MATHUTIL_CUH

#include <cmath>
#include <vector>

class MathUtil {
public:
    // Translation
    __host__ __device__
    static void translate3D(
        float &x, float &y, float &z, // The vertex
        float dx, float dy, float dz // The translation
    );
    // Rotation
    __host__ __device__
    static void rotate3D(
        float &x, float &y, float &z, // The vertex
        float ox, float oy, float oz, // The origin
        float wx=0, float wy=0, float wz=0 // The axis rotation
    );
    // Scaling
    __host__ __device__
    static void scale3D(
        float &x, float &y, float &z, // The vertex
        float ox, float oy, float oz, // The origin
        float sx=1, float sy=1, float sz=1 // The scale factor
    );
    __host__ __device__
    static void scale3D(
        float &x, float &y, float &z, // The vertex
        float ox, float oy, float oz, // The origin
        float scl=1 // The scale factor
    );
};

// Kernel for mesh transformations
__global__ void translateVertices(
    float *px, float *py, float *pz,
    float dx, float dy, float dz,
    uint32_t *vMeshIds,
    uint32_t meshId,
    uint32_t numVertices
);
__global__ void rotateVertices(
    float *px, float *py, float *pz,
    float ox, float oy, float oz,
    float wx, float wy, float wz,
    uint32_t *vMeshIds,
    uint32_t meshId,
    uint32_t numVertices
);
__global__ void scaleVertices(
    float *px, float *py, float *pz,
    float ox, float oy, float oz,
    float sx, float sy, float sz,
    uint32_t *vMeshIds,
    uint32_t meshId,
    uint32_t numVertices
);

#endif