#ifndef MESH3D_CUH
#define MESH3D_CUH

#include <Matrix.cuh>
#include <cuda_runtime.h>

#define UInt unsigned int
#define ULInt unsigned long int
#define ULLInt unsigned long long int

/* We will use a hybrid AoS/SoA approach

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
    std::vector<Vec3f> pos;
    std::vector<Vec3f> normal;
    std::vector<Vec2f> tex;
    std::vector<UInt> mID;
    std::vector<Vec3uli> faces;
};

class Mesh3D {
public:
    // Number of vertices and faces
    ULLInt numVs, numFs;

    // Vertices
    Vec3f *pos;
    Vec3f *normal;
    Vec2f *tex;
    UInt *mID;

    // Faces (triangles)
    Vec3uli *faces;

    Mesh3D(ULLInt numVs=0, ULLInt numFs=0);
    Mesh3D(
        UInt mID,
        std::vector<Vec3f> &pos,
        std::vector<Vec3f> &normal,
        std::vector<Vec2f> &tex,
        std::vector<Vec3uli> &faces
    );

    // Memory management
    void mallocVertices();
    void resizeVertices(ULLInt numVs);
    void freeVertices();

    void mallocFaces();
    void resizeFaces(ULLInt numFs);
    void freeFaces();

    // Upload host data to device
    void uploadData(
        UInt mID,
        std::vector<Vec3f> &pos,
        std::vector<Vec3f> &normal,
        std::vector<Vec2f> &tex,
        std::vector<Vec3uli> &faces
    );

    // Mesh operators
    void operator+=(Mesh3D &mesh);

    // DEBUG
    void printVertices(bool pos=true, bool normal=true, bool tex=true, bool mID=true);
};

// Kernel for preparing vertices
__global__ void incrementFaceIdxKernel(
    Vec3uli *faces, ULLInt numFs, ULLInt offset
);
__global__ void setMeshIDKernel(
    UInt *mID, ULLInt numVs, UInt id
);

#endif