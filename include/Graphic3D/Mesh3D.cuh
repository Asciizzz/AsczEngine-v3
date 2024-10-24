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

    Mesh(
        std::vector<float> wx, std::vector<float> wy, std::vector<float> wz,
        std::vector<float> nx, std::vector<float> ny, std::vector<float> nz,
        std::vector<float> tu, std::vector<float> tv,
        std::vector<float> cr, std::vector<float> colorG, std::vector<float> colorB, std::vector<float> colorA,
        std::vector<ULLInt> fw, std::vector<ULLInt> ft, std::vector<ULLInt> fn
    );
    Mesh();

    Vec3f w3f(ULLInt i);
    Vec3f n3f(ULLInt i);
    Vec2f t2f(ULLInt i);
    Vec4f c4f(ULLInt i);
    Vec3ulli fw3ulli(ULLInt i);
    Vec3ulli ft3ulli(ULLInt i);
    Vec3ulli fn3ulli(ULLInt i);
};

// Cool device mesh (for parallel processing)
class Mesh3D {
public:
    // Beta: SoA vertex data
    Vecptr3f world;
    Vecptr3f normal;
    Vecptr2f texture;
    Vecptr4f color;
    Vecptr4f screen;

    // Faces index (triangles)
    // Every 3 indices is a face
    Vecptr4ulli faces;

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
};

// Kernel for preparing faces
__global__ void incrementFaceIdxKernel(ULLInt *f, ULLInt offset, ULLInt numFs);

#endif