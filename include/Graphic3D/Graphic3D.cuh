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

    // Data
    Mesh3D mesh;
    Camera3D camera;
    Buffer3D buffer;

    void operator+=(Mesh3D &m);

    // For vertex shader and rasterization
    Vec4f *projection; // x, y, depth, isInsideFrustum
    void allocateProjection();
    void freeProjection();
    void resizeProjection();

    // For bresenham and rasterization (BETA)
    Vec2uli *edges; // Note: the indices are from the world space
    void allocateEdges();
    void freeEdges();
    void resizeEdges();

    // BETA: LightSrc and shadow mapping
    LightSrc light;

    int sWidth, sHeight, sSize;
    bool *shadowActive;
    float *shadowDepth;
    Vec3f *lightProj;
    void allocateShadow(int sw, int sh);
    void freeShadow();

private:
    Graphic3D() {}
};

// Helpful device functions
__device__ bool atomicMinFloat(float* addr, float value);

// Helpful kernels
__global__ void facesToEdgesKernel(
    Vec2uli *edges, Vec3x3uli *faces, ULLInt numFs
);

#endif