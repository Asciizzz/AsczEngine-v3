#ifndef MESH3D_CUH
#define MESH3D_CUH

#include <Vector.cuh>
#include <cuda_runtime.h>
#include <map>

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

#define MeshMap std::map<std::string, Mesh>
#define ObjRangeMap std::map<std::string, ObjRange>

#define VectF std::vector<float>

#define VectI std::vector<int>
#define VectLLI std::vector<LLInt>
#define VectULLI std::vector<ULLInt>

#define VectStr std::vector<std::string>

struct ObjRange {
    ULLInt w1, w2;
    ULLInt t1, t2;
    ULLInt n1, n2;
    ULLInt f1, f2;

    ObjRange(
        ULLInt w1=0, ULLInt w2=0,
        ULLInt t1=0, ULLInt t2=0,
        ULLInt n1=0, ULLInt n2=0,
        ULLInt f1=0, ULLInt f2=0
    );
    
    void operator=(ObjRange &range);

    void offsetW(ULLInt offset);
    void offsetT(ULLInt offset);
    void offsetN(ULLInt offset);
    void offsetF(ULLInt offset);
};
struct Mesh {
    // Initialization data (Will be obsolete after push to Mesh3D)

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
    VectLLI mkd;

    // Texture data
    VectF txr, txg, txb;
    VectI txw, txh; VectLLI txof;

    // Object data
    ObjRangeMap objmapST; // Static object (when created)
    ObjRangeMap objmapRT; // Runtime object (in device memory)
    VectStr objmapKs; // Keys to ensure order

    // Other data
    bool allocated = false;
    std::string name = "default";

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
        VectF ksr, VectF ksg, VectF ksb,
        VectLLI mkd,
        // Texture data
        VectF txr, VectF txg, VectF txb,
        VectI txw, VectI txh, VectLLI txof,
        // Object data
        ObjRangeMap objmap, VectStr objmapKs
    );

    // Return vertex data
    Vec3f w3f(ULLInt i);
    Vec2f t2f(ULLInt i);
    Vec3f n3f(ULLInt i);

    // Initialize transformations
    void translateIni(Vec3f t);
    void rotateIni(Vec3f origin, float r, short axis); // 0: x, 1: y, 2: z
    void scaleIni(Vec3f origin, Vec3f scl, bool sclNormal=true);

    // Runtime transformations (uses device memory)
    void translateRuntime(std::string mapkey, Vec3f t);
    void rotateRuntime(std::string mapkey, Vec3f origin, float r, short axis);
    void scaleRuntime(std::string mapkey, Vec3f origin, float scl);

    std::string getObjRtMapLog();
};

/* Note:

Vertex_ptr and Texture_ptr don't have size because
every values have their own size

*/

// Vertex Ptr
struct Vertex_ptr {
    Vec4f_ptr s;
    Vec3f_ptr w;
    Vec2f_ptr t;
    Vec3f_ptr n;

    void malloc(ULLInt ws, ULLInt ts, ULLInt ns);
    void free();
    void operator+=(Vertex_ptr &vertex);
};

// Face Ptr
struct Face_ptr {
    // Data x3
    ULLInt *v; // Vertex index
    LLInt *t; // Texture index
    LLInt *n; // Normal index

    // Data x1
    LLInt *m; // Material index

    ULLInt size = 0; // size = 3 * count
    ULLInt count = 0; // count = size / 3

    void malloc(ULLInt fcount);
    void free();
    void operator+=(Face_ptr &face);
};

// Material Ptr
struct Material_ptr {
    Vec3f_ptr ka;
    Vec3f_ptr kd;
    Vec3f_ptr ks;
    Vec1lli_ptr mkd;
    ULLInt size = 0;

    void malloc(ULLInt size);
    void free();
    void operator+=(Material_ptr &material);
};

// Texture Ptr
struct Texture_ptr {
    /* Explanation:
    
    Every texture will be flattened into a 1D array

    The offset array will store the starting index of each texture

    Example: a 100x100 and 200x200 texture

    t will be a 1D array of size 100*100 + 200*200
    w and h is straightforward
    offset will be {0, 100*100} for the above example
        offset_n = offset_n-1 + current_t_size
    */

    Vec3f_ptr tx; // Color rgb
    Vec2i_ptr wh; // Width and height
    Vec1lli_ptr of; // Offset

    ULLInt size = 0; // Size of textures
    ULLInt count = 0; // Number of textures

    void malloc(ULLInt tsize, ULLInt tcount);
    void free();
    void operator+=(Texture_ptr &texture);
};

// Device mesh (SoA for coalesced memory access)
class Mesh3D {
private:
    std::map<std::string, int> repeatname; // Repetition counter

public:
    Vertex_ptr v; // s w t n c
    Face_ptr f; // v t n m
    Material_ptr m; // ka kd ks map_Kd ns
    Texture_ptr t; // tx wh of

    MeshMap meshmap;

    // Free everything
    void free();
    /* Important note:

    Push is generally a pretty bad operation as it perform frequent memory allocation

    In the future, we will implement a pre-allocated memory pool for Mesh3D

    Pushing also comes with the downside of unreusable components

    For example if you were to push 10 meshes with the same texture
    the texture will be duplicated 10 times

    This will be fixed in the future
    */
    void push(Mesh &mesh, bool correction=true);
    void push(std::vector<Mesh> &meshs, bool correction=true);
    
    // ========== ENTIRELY For debugging ==========
    int curlogpart = 0;
    int maxlogpart = 0;
    VectStr meshmapLog;
    void logMeshMap(int linePerPart=20);
    std::string getMeshMapLog();
};

// Kernel for preparing faces
__global__ void incULLIntKernel(ULLInt *f, ULLInt offset, ULLInt numFs);
__global__ void incLLIntKernel(LLInt *f, ULLInt offset, ULLInt numFs);

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
    float ox, float oy, float oz, float scl
);

#endif