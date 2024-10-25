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

// Lame host mesh
struct Mesh {
    ULLInt numWs, numNs, numTs;
    Vecs3f world;
    Vecs3f normal;
    Vecs2f texture;
    Vecs4f color;

    ULLInt numFs;
    ULLInts faceWs;
    ULLInts faceNs;
    ULLInts faceTs;

    Mesh(
        Vecs3f world, Vecs3f normal, Vecs2f texture, Vecs4f color,
        ULLInts faceWs, ULLInts faceTs, ULLInts faceNs
    );
    Mesh(
        Vecs3f world, Vecs3f normal, Vecs2f texture, Vecs4f color, ULLInts faceSame
    );
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

    Mesh3D(ULLInt numWs=0, ULLInt numNs=0, ULLInt numTs=0, ULLInt numFs=0);

    void mallocVertices(ULLInt numWs=0, ULLInt numNs=0, ULLInt numTs=0);
    void freeVertices();
    void resizeVertices(ULLInt numWs, ULLInt numNs, ULLInt numTs);

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