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
#define VectULLI std::vector<ULLInt>
#define VectLL std::vector<long long>

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
    VectF wx, wy, wz;
    VectF tu, tv;
    VectF nx, ny, nz;
    VectF cr, cg, cb, ca;
    VectULLI fw, ft, fn;
    VectLL fm;

    // Section 2: runtime, note: i = [a, b)
    Vec2ulli w_range, n_range, t_range, c_range;

    Mesh(
        // Vertex data
        VectF wx, VectF wy, VectF wz,
        VectF tu, VectF tv,
        VectF nx, VectF ny, VectF nz,
        VectF cr, VectF cg, VectF cb, VectF ca,
        // Face data
        VectULLI fw, VectULLI ft, VectULLI fn, VectLL fm = {}
    );
    Mesh();

    void push(Mesh &mesh);

    // Return vertex data
    Vec3f w3f(ULLInt i);
    Vec2f t2f(ULLInt i);
    Vec3f n3f(ULLInt i);
    Vec4f c4f(ULLInt i);

    // Section 1 transformations
    void translateIni(Vec3f t);
    void rotateIni(Vec3f origin, float r, short axis); // 0: x, 1: y, 2: z
    void scaleIni(Vec3f origin, Vec3f scl, bool sclNormal=true);

    // Section 2 transformations
    void translateRuntime(Vec3f t);
    void rotateRuntime(Vec3f origin, float r, short axis);
    void scaleRuntime(Vec3f origin, Vec3f scl);
};

// Face Ptr
struct Face_ptr {
    ULLInt *v, *t, *n;
    long long *m; // -1 by default for no material
    ULLInt size;

    void malloc(ULLInt size);
    void free();
    void operator+=(Face_ptr &face);
};

// Device mesh (SoA for coalesced memory access)
class Mesh3D {
public:
    // Vertex data
    Vec4f_ptr s; // x y z w
    Vec3f_ptr w; // x y z
    Vec2f_ptr t; // u v
    Vec3f_ptr n; // nx ny nz
    Vec4f_ptr c; // r g b a
    // Face data
    Face_ptr f; // v t n mat
    // Material data
    Vec3f_ptr ka; // r g b - ignore for now
    Vec3f_ptr kd; // r g b - focus on this
    Vec3f_ptr ks; // r g b - ignore for now
    ULLInt *map_Kd; // Texture id - focus on this
    float *ns; // shininess - ignore for now
    // Texture data (very sophisticated)
    ULLInt tNum; // Number of textures
    float *txtr; // Texture data (flattened to 1D)
    ULLInt *tstart; // Texture start index
    Vec2f_ptr *tsize; // Texture width and height

    // Free
    void free();
    // Resize + Append
    void push(Mesh &mesh);
    void push(std::vector<Mesh> &meshs);
};

// Kernel for preparing faces
__global__ void incrementFaceIdxKernel(ULLInt *f, ULLInt offset, ULLInt numFs);

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