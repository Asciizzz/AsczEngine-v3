#ifndef RENDER3D_CUH
#define RENDER3D_CUH

#include <Buffer3D.cuh>

class Render3D {
public:
    // Singleton
    static Render3D& instance() {
        static Render3D instance;
        return instance;
    }
    Render3D(const Render3D&) = delete;
    Render3D &operator=(const Render3D&) = delete;

    Vec2f res = {800, 600};
    Vec2f res_half = {400, 300};
    int pixelSize = 4;
    void setResolution(float w, float h);

    Mesh3D mesh;
    Camera3D camera;
    Buffer3D buffer;

    Vec4f *projection; // x, y, depth, isInsideFrustum
    void allocateProjection();
    void freeProjection();
    void resizeProjection();

    // Render pipeline
    void vertexProjection();
    void rasterizeFaces();

private:
    Render3D() {}
};

// Pipeline Kernels
__global__ void vertexProjectionKernel(Vec4f *projection, Vec3f *world, Camera3D camera, int p_s, ULLInt numVs);
__global__ void rasterizeFacesKernel(
    // Mesh data
    Vec4f *projection, Vec4f *color, Vec3uli *faces, ULLInt numFs,
    // Buffer data
    float *buffDepth, Vec4f *buffColor, int buffWidth, int buffHeight
);

#endif