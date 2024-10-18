#ifndef MESH3D_CUH
#define MESH3D_CUH

#include <Matrix.cuh>
#include <cuda_runtime.h>

/* HYBRID AOS-SOA MEMORY LAYOUT

Vertex data: x y z nx ny nz u v

Instead of creating an array for all 8 attributes
We will group related attributes together

We will have 4 arrays for vertex data:
- Position (x y z)
- Normal (nx ny nz)
- Texture (u v)
- Mesh ID (id)
*/

struct Mesh {
    Vecs3f pos;
    Vecs3f normal;
    Vecs2f tex;
    Vecs3f color;
    UInts mID;

    Vecs3uli faces;

    Mesh(UInt id, Vecs3f &pos, Vecs3f &normal, Vecs2f &tex, Vecs3f &color, Vecs3uli &faces);
    Mesh(Mesh &mesh);

    Mesh operator+=(Mesh &mesh);
};

class Mesh3D {
public:
    // Number of vertices and faces
    ULLInt numVs, numFs;

    // Vertices
    Vec3f *pos;
    Vec3f *normal;
    Vec2f *tex;
    Vec3f *color;
    UInt *mID;

    // Faces (triangles)
    Vec3uli *faces;

    Mesh3D(ULLInt numVs=0, ULLInt numFs=0);
    Mesh3D(UInt id, Vecs3f &pos, Vecs3f &normal, Vecs2f &tex, Vecs3f &color, Vecs3uli &faces);
    Mesh3D(Mesh &mesh);

    // Memory management
    void mallocVertices();
    void resizeVertices(ULLInt numVs);
    void freeVertices();

    void mallocFaces();
    void resizeFaces(ULLInt numFs);
    void freeFaces();

    // Upload host data to device
    void uploadData(UInt id, Vecs3f &pos, Vecs3f &normal, Vecs2f &tex, Vecs3f &color, Vecs3uli &faces);

    // Mesh operators
    void operator+=(Mesh3D &mesh);

    // DEBUG
    void printVertices(bool pos=true, bool normal=true, bool tex=true, bool color=true, bool mID=true);
    void printFaces();
};

// Kernel for preparing vertices
__global__ void incrementFaceIdxKernel(Vec3uli *faces, ULLInt numFs, ULLInt offset);
__global__ void setMeshIDKernel(UInt *mID, ULLInt numVs, UInt id);

#endif