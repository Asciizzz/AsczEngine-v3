#ifndef MESH3D_CUH
#define MESH3D_CUH

#include <Matrix.cuh>
#include <cuda_runtime.h>

/* HYBRID AOS-SOA MEMORY LAYOUT

Vertex data: x y z nx ny nz u v

Instead of creating an array for all 8 attributes
We will group related attributes together

We will have 4 arrays for vertex data:
- worldition (x y z)
- Normal (nx ny nz)
- Textureture (u v)
- obj Id (id)
*/

#define Meshs3D std::vector<Mesh3D>

struct Mesh {
    std::vector<float> wx, wy, wz;
    std::vector<float> nx, ny, nz;
    std::vector<float> tu, tv;
    std::vector<float> cr, cg, cb, ca;
    std::vector<ULLInt> fw, ft, fn;

    // Note: a <= i < b, [a, b)
    Vec2ulli w_range, n_range, t_range, c_range;
    Vec2ulli fw_range, ft_range, fn_range;

    Mesh(
        std::vector<float> wx, std::vector<float> wy, std::vector<float> wz,
        std::vector<float> nx, std::vector<float> ny, std::vector<float> nz,
        std::vector<float> tu, std::vector<float> tv,
        std::vector<float> cr, std::vector<float> colorG, std::vector<float> colorB, std::vector<float> colorA,
        std::vector<ULLInt> fw, std::vector<ULLInt> ft, std::vector<ULLInt> fn
    );
    Mesh();

    // Return vertex data
    Vec3f w3f(ULLInt i);
    Vec3f n3f(ULLInt i);
    Vec2f t2f(ULLInt i);
    Vec4f c4f(ULLInt i);

    // Instant Transformations
    void translateStatic(Vec3f t);
    void rotateStatic(Vec3f origin, Vec3f rot, bool rotNormal=true);
    void scaleStatic(Vec3f origin, Vec3f scl, bool sclNormal=true);

    // Runtime Transformations (in the Graphic3D.Mesh3D)
    void translateRuntime(Vec3f t);
    void rotateRuntime(Vec3f origin, Vec3f rot);
    void scaleRuntime(Vec3f origin, Vec3f scl);
};

// Cool device mesh (for parallel processing)
class Mesh3D {
public:
    Vec3f_ptr world;
    Vec3f_ptr normal;
    Vec2f_ptr texture;
    Vec4f_ptr color;
    Vec4f_ptr screen;
    Vec4ulli_ptr faces;

    Mesh3D(ULLInt numWs=0, ULLInt numTs=0, ULLInt numNs=0, ULLInt numFs=0);

    void mallocVertices(ULLInt numWs=0, ULLInt numTs=0, ULLInt numNs=0);
    void freeVertices();
    void resizeVertices(ULLInt numWs, ULLInt numTs, ULLInt numNs);

    void mallocFaces(ULLInt numFs=0);
    void freeFaces();
    void resizeFaces(ULLInt numFs);

    void free();

    // Mesh operators
    void operator+=(Mesh &mesh);
    void operator+=(std::vector<Mesh> &meshs);
};

// Kernel for preparing faces
__global__ void incrementFaceIdxKernel(ULLInt *f, ULLInt offset, ULLInt numFs);

// Kernel for transformations
// Note: rotation and scaling also affects normals

__global__ void translateKernel(
    float *wx, float *wy, float *wz, float tx, float ty, float tz, ULLInt numWs
);

__global__ void rotateWsKernel(
    float *wx, float *wy, float *wz,
    float ox, float oy, float oz,
    float rx, float ry, float rz, ULLInt numWs
);
__global__ void rotateNsKernel(
    float *nx, float *ny, float *nz,
    float rx, float ry, float rz, ULLInt numNs
);

__global__ void scaleWsKernel(
    float *wx, float *wy, float *wz,
    float ox, float oy, float oz,
    float sx, float sy, float sz, ULLInt numWs
);
__global__ void scaleNsKernel(
    float *nx, float *ny, float *nz,
    float sx, float sy, float sz, ULLInt numNs
);

#endif