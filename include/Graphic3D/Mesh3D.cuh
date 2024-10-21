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
- Mesh ID (id)
*/

#define Meshs std::vector<Mesh>
#define Meshs3D std::vector<Mesh3D>

struct Mesh {
    Vecs3f world;
    Vecs3f normal;
    Vecs2f texture;
    Vecs4f color;
    UInts meshId;

    Vecs3x3uli faces;

    Mesh(UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, Vecs3x3uli &faces);
    Mesh(Mesh &mesh, UInt id);
    Mesh() {}

    Mesh operator+=(Mesh &mesh);
    Mesh operator+(Mesh mesh);
};

class Mesh3D {
public:
    // Number of world coorinates, normal, texture and faces
    ULLInt numWs, numNs, numTs, numFs;

    // Block properties
    ULLInt blockSize = 256;
    ULLInt blockNumWs, blockNumNs, blockNumTs, blockNumFs;

    // Vertices
    Vec3f *world;
    Vec3f *normal;
    Vec2f *texture;
    Vec4f *screen; // Projected screen space

    UInt *wMeshId;
    UInt *nMeshId;
    UInt *tMeshId;

    Vec4f *color; // Color is like that weird kid that doesn't fit in
    // Buf he's our friend so we keep him around

    // Faces (triangles)
    Vec3x3uli *faces;

    Mesh3D(ULLInt numWs=0, ULLInt numNs=0, ULLInt numTs=0, ULLInt numFs=0);
    Mesh3D(UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, Vecs3x3uli &faces);
    Mesh3D(Mesh &mesh);

    // Memory management
    void mallocVertices();
    void resizeVertices(ULLInt numWs, ULLInt numNs, ULLInt numTs);
    void freeVertices();

    void mallocFaces();
    void resizeFaces(ULLInt numFs);
    void freeFaces();

    void free();

    // Upload host data to device
    void uploadData(UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, Vecs3x3uli &faces);

    // Mesh operators
    void operator+=(Mesh3D &mesh);

    // Transformations (with mesh ID)
    void translate(UInt mID, Vec3f t);
    void rotate(UInt mID, Vec3f origin, Vec3f rot);
    void scale(UInt mID, Vec3f origin, Vec3f scl);
    // Transformations (all mesh IDs)
    void translate(Vec3f t);
    void rotate(Vec3f origin, Vec3f rot);
    void scale(Vec3f origin, Vec3f scl);
};

// Kernel for preparing vertices
__global__ void incrementFaceIdxKernel(Vec3x3uli *faces, ULLInt offsetW, ULLInt offsetN, ULLInt offsetT, ULLInt numFs, ULLInt newNumFs);
__global__ void setMeshIdKernel(UInt *meshId, ULLInt numWs, UInt id);

// Kernel for transforming vertices
// Note: the reason bool allID is used is because we can't overload kernels
__global__ void translateWorldKernel(Vec3f *world, UInt *wMeshId, bool allID, ULLInt numWs, UInt mID, Vec3f t);
__global__ void rotateWorldKernel(Vec3f *world, UInt *wMeshId, bool allID, ULLInt numWs, UInt mID, Vec3f origin, Vec3f rot);
__global__ void scaleWorldKernel(Vec3f *world, UInt *wMeshId, bool allID, ULLInt numWs, UInt mID, Vec3f origin, Vec3f scl);

// Only rotation and scaling are needed for normals
__global__ void rotateNormalKernel(Vec3f *normal, UInt *nMeshId, bool allID, ULLInt numNs, UInt mID, Vec3f origin, Vec3f rot);
__global__ void scaleNormalKernel(Vec3f *normal, UInt *nMeshId, bool allID, ULLInt numNs, UInt mID, Vec3f origin, Vec3f scl);

#endif