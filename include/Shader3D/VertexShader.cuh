#ifndef VERTEXSHADER_CUH
#define VERTEXSHADER_CUH

#include <Buffer3D.cuh>

class VertexShader {
public:
    // Singleton
    static VertexShader& instance() {
        static VertexShader instance;
        return instance;
    }
    VertexShader(const VertexShader&) = delete;
    VertexShader &operator=(const VertexShader&) = delete;

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
    
    // Render functions
    __host__ __device__ static Vec4f toScreenSpace(
        Camera3D &camera, Vec3f world, int buffWidth, int buffHeight
    );

    // Render pipeline
    void cameraProjection();
    void createDepthMap();
    void rasterization();

private:
    VertexShader() {}
};

__device__ bool atomicMinFloat(float* addr, float value);
// Pipeline Kernels
__global__ void cameraProjectionKernel(
    Vec4f *projection, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numVs
);
__global__ void createDepthMapKernel(
    // Mesh data
    Vec4f *projection, Vec3f *world, Vec3uli *faces, ULLInt numFs,
    // Buffer data
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary,
    int buffWidth, int buffHeight
);
__global__ void rasterizationKernel(
    // Mesh data
    Vec4f *color, Vec3f *world, Vec3f *normal, Vec2f *texture, UInt *meshID, Vec3uli *faces,
    // Buffer data
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    UInt *buffMeshId, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight
);

#endif