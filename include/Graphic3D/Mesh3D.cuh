#ifndef MESH3D_CUH
#define MESH3D_CUH

#include <Vector.cuh>
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

#define VectF std::vector<float>
#define VectLLI std::vector<LLInt>
#define VectULLI std::vector<ULLInt>

struct Mesh {
    /* VERY IMPORTANT NOTE:

    Section 1 is only used for initialization
    Once the global Mesh3D object append it
    it will become practically useless

    When we want to perform transformations
    we will use the range values in section 2
    to apply directly to the device memory
    of the global Mesh3D object
    */

    // Section 1: initialization
    
    // Vertex data
    VectF wx, wy, wz;
    VectF tu, tv;
    VectF nx, ny, nz;

    // Face data
    VectULLI fw;
    VectLLI ft, fn, fm;

    // Material data
    VectF kar, kag, kab;
    VectF kdr, kdg, kdb;
    VectF ksr, ksg, ksb;

    // Section 2: runtime, note: i = [a, b)
    Vec2ulli w_range, n_range, t_range, c_range;

    Mesh();
    Mesh(
        // Vertex data
        VectF wx, VectF wy, VectF wz,
        VectF tu, VectF tv,
        VectF nx, VectF ny, VectF nz,
        // Face data
        VectULLI fw, VectLLI ft, VectLLI fn, VectLLI fm,
        // Material data
        VectF kar, VectF kag, VectF kab,
        VectF kdr, VectF kdg, VectF kdb,
        VectF ksr, VectF ksg, VectF ksb
    );

    void push(Mesh &mesh);

    // Return vertex data
    Vec3f w3f(ULLInt i);
    Vec2f t2f(ULLInt i);
    Vec3f n3f(ULLInt i);

    // Section 1 transformations
    void translateIni(Vec3f t);
    void rotateIni(Vec3f origin, float r, short axis); // 0: x, 1: y, 2: z
    void scaleIni(Vec3f origin, Vec3f scl, bool sclNormal=true);

    // Section 2 transformations
    void translateRuntime(Vec3f t);
    void rotateRuntime(Vec3f origin, float r, short axis);
    void scaleRuntime(Vec3f origin, Vec3f scl);
};

// Vertex Ptr
struct Vertex_ptr {
    Vec4f_ptr s;
    Vec3f_ptr w;
    Vec2f_ptr t;
    Vec3f_ptr n;

    void malloc(ULLInt size);
    void free();
    void operator+=(Vertex_ptr &vertex);
};

// Face Ptr
struct Face_ptr {
    ULLInt *v;
    LLInt *t;
    LLInt *n;
    LLInt *m; // -1 by default for no material
    ULLInt size;

    void malloc(ULLInt size);
    void free();
    void operator+=(Face_ptr &face);
};

// Material Ptr
struct Material_ptr {
    Vec3f_ptr ka;
    Vec3f_ptr kd;
    Vec3f_ptr ks;

    ULLInt size;
    void malloc(ULLInt size);
    void free();
    void operator+=(Material_ptr &material);
};

// Device mesh (SoA for coalesced memory access)
class Mesh3D {
public:
    Vertex_ptr v; // s w t n c
    Face_ptr f; // v t n m
    Material_ptr m; // ka kd ks map_Kd ns

    // Free
    void free();
    // Resize + Append
    void push(Mesh &mesh);
    void push(std::vector<Mesh> &meshs);
};

// Kernel for preparing faces
__global__ void incFaceIdxKernel1(ULLInt *f, ULLInt offset, ULLInt numFs);
__global__ void incFaceIdxKernel2(LLInt *f, ULLInt offset, ULLInt numFs);

// Kernel for transformations
// Note: rotation and scaling also affects normals

__global__ void translateMeshKernel(
    float *wx, float *wy, float *wz, float tx, float ty, float tz, ULLInt numWs
);

__global__ void rotateMeshKernel(
    float *wx, float *wy, float *wz, ULLInt numWs,
    float *nx, float *ny, float *nz, ULLInt numNs,
    float ox, float oy, float oz,
    float r, short axis
);

__global__ void scaleMeshKernel(
    float *wx, float *wy, float *wz, ULLInt numWs,
    float *nx, float *ny, float *nz, ULLInt numNs,
    float ox, float oy, float oz,
    float sx, float sy, float sz
);

#endif