#ifndef GRAPHIC3D_CUH
#define GRAPHIC3D_CUH

/* This is a class that contains:

Mesh3D
Camera3D
Buffer3D
Projection

We will not put any data into the Shader classes

*/

#include <Mesh3D.cuh> 
#include <Camera3D.cuh>
#include <Buffer3D.cuh>

// BETA: LightSrc
struct LightSrc {
    Vec3f dir = {-1, -1, -1};
    float ambient = 0.1;
    float specular = 1.2;
    Vec3f color = {1, 1, 1};

    std::string data() {
        std::string str = "LightSrc:\n";
        str += "| Dir: " + std::to_string(dir.x) + " " + std::to_string(dir.y) + " " + std::to_string(dir.z) + "\n";
        str += "| Ambient: " + std::to_string(ambient) + "\n";
        str += "| Specular: " + std::to_string(specular) + "\n";
        str += "| Color: " + std::to_string(color.x) + " " + std::to_string(color.y) + " " + std::to_string(color.z) + "\n";
        return str;
    }
};

class Graphic3D {
public:
    // Singleton
    static Graphic3D& instance() {
        static Graphic3D instance;
        return instance;
    }
    Graphic3D(const Graphic3D&) = delete;
    Graphic3D &operator=(const Graphic3D&) = delete;

    // Resolution
    Vec2f res = {800, 600};
    Vec2f res_half = {400, 300};
    int pixelSize = 8;
    void setResolution(float w, float h, float ps=4);

    // For tile-based rasterization
    int tileWidth = 20;
    int tileHeight = 20;
    int tileNumX, tileNumY, tileNum;
    void setTileSize(int tw, int th);

    // Free everything
    void free();

    // Mesh3D and its properties
    Mesh3D mesh;
    void appendMesh(Mesh3D &m, bool del=true);

    // Face properties
    ULLInt numVisibFs;
    ULLInt *d_numVisibFs;
    Vec4ulli *visibFWs;

    void mallocGFaces();
    void freeGFaces();
    void resizeGFaces();

    // Face stream for chunking
    cudaStream_t *faceStreams;
    size_t chunkSize = 5e5;
    int chunkNum;
    void mallocFaceStreams();
    void freeFaceStreams();
    void resizeFaceStreams();

    // Camera3D and Buffer3D
    Camera3D camera;
    Buffer3D buffer;

    // BETA: LightSrc and shadow mapping
    LightSrc light;

    // Transformations for mesh with specific id
    void translateMesh(UInt objId, Vec3f t);
    void rotateMesh(UInt objId, Vec3f origin, Vec3f rot);
    void scaleMesh(UInt objId, Vec3f origin, Vec3f scl);

private:
    Graphic3D() {}
};

// Helpful device functions
__device__ bool atomicMinFloat(float* addr, float value);

#endif