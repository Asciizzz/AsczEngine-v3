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
    ULLInt blockSize = 256;
    ULLInt blockNumWs, blockNumNs, blockNumTs, blockNumFs;

    // Number of world coorinates, normal, texture
    ULLInt numWs, numNs, numTs;

    // Vertices
    Vec3f *world;
    Vec3f *normal;
    Vec2f *texture;
    Vec4f *screen; // Projected screen space

    Vec4f *color; // Color is like that weird kid that doesn't fit in
    // Buf he's our friend so we keep him around

    // Faces index (triangles)
    ULLInt numFs;
    Vec3ulli *faceWs;
    Vec3ulli *faceNs;
    Vec3ulli *faceTs;

    Mesh3D(ULLInt numWs=0, ULLInt numNs=0, ULLInt numTs=0, ULLInt numFs=0);
    Mesh3D(
        UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color,
        Vecs3ulli &faceWs, Vecs3ulli &faceNs, Vecs3ulli &faceTs
    );
    Mesh3D(
        UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, 
        Vecs3ulli &faceAll
    );

    // Memory management
    void mallocVertices();
    void resizeVertices(ULLInt numWs, ULLInt numNs, ULLInt numTs);
    void freeVertices();

    void mallocFaces();
    void resizeFaces(ULLInt numFs);
    void freeFaces();

    void free();

    // Upload host data to device
    void uploadData(
        UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color,
        Vecs3ulli &faceWs, Vecs3ulli &faceNs, Vecs3ulli &faceTs
    );

    // Mesh operators
    void operator+=(Mesh3D &mesh);
};

// Kernel for preparing vertices
__global__ void incrementFaceIdxKernel(Vec3ulli *faces, ULLInt offset, ULLInt numFs, ULLInt newNumFs);

#endif