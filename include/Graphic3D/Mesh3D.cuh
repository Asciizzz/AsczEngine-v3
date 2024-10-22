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

class Mesh3D {
public:

    // Block properties
    ULInt blockSize = 256;
    ULInt blockNumWs, blockNumNs, blockNumTs, blockNumFs;

    // Number of world coorinates, normal, texture
    ULInt numWs, numNs, numTs;

    // Vertices
    Vec3f *world;
    Vec3f *normal;
    Vec2f *texture;
    Vec4f *screen; // Projected screen space

    Vec4f *color; // Color is like that weird kid that doesn't fit in
    // Buf he's our friend so we keep him around

    // Faces index (triangles)
    ULInt numFs;
    Vec3uli *faceWs;
    Vec3uli *faceNs;
    Vec3uli *faceTs;

    // Object Ids for vertex attributes
    UInt *wObjId;
    UInt *nObjId;
    UInt *tObjId;

    Mesh3D(ULInt numWs=0, ULInt numNs=0, ULInt numTs=0, ULInt numFs=0);
    Mesh3D(
        UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color,
        Vecs3uli &faceWs, Vecs3uli &faceNs, Vecs3uli &faceTs
    );
    Mesh3D(
        UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, 
        Vecs3uli &faceAll
    );

    // Memory management
    void mallocVertices();
    void resizeVertices(ULInt numWs, ULInt numNs, ULInt numTs);
    void freeVertices();

    void mallocFaces();
    void resizeFaces(ULInt numFs);
    void freeFaces();

    void free();

    // Upload host data to device
    void uploadData(
        UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color,
        Vecs3uli &faceWs, Vecs3uli &faceNs, Vecs3uli &faceTs
    );

    // Mesh operators
    void operator+=(Mesh3D &mesh);

    // Transformations (with obj Id)
    void translate(UInt objID, Vec3f t);
    void rotate(UInt objID, Vec3f origin, Vec3f rot);
    void scale(UInt objID, Vec3f origin, Vec3f scl);
    // Transformations (all obj Ids)
    void translate(Vec3f t);
    void rotate(Vec3f origin, Vec3f rot);
    void scale(Vec3f origin, Vec3f scl);
};

// Kernel for preparing vertices
__global__ void incrementFaceIdxKernel(Vec3uli *faces, ULInt offset, ULInt numFs, ULInt newNumFs);
__global__ void setObjIdKernel(UInt *objId, ULInt numWs, UInt id);

// Kernel for transforming vertices
// Note: the reason bool allID is used is because we can't overload kernels
__global__ void translateWorldKernel(Vec3f *world, UInt *wObjId, bool allID, ULInt numWs, UInt objID, Vec3f t);
__global__ void rotateWorldKernel(Vec3f *world, UInt *wObjId, bool allID, ULInt numWs, UInt objID, Vec3f origin, Vec3f rot);
__global__ void scaleWorldKernel(Vec3f *world, UInt *wObjId, bool allID, ULInt numWs, UInt objID, Vec3f origin, Vec3f scl);

// Only rotation and scaling are needed for normals
__global__ void rotateNormalKernel(Vec3f *normal, UInt *nObjId, bool allID, ULInt numNs, UInt objID, Vec3f origin, Vec3f rot);
__global__ void scaleNormalKernel(Vec3f *normal, UInt *nObjId, bool allID, ULInt numNs, UInt objID, Vec3f origin, Vec3f scl);

#endif