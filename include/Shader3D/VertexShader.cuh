#ifndef VERTEXSHADER_CUH
#define VERTEXSHADER_CUH

#include <Graphic3D.cuh>

class VertexShader {
public:
    // Render functions
    __host__ __device__ static Vec4f toScreenSpace(
        Camera3D &camera, Vec3f world, int buffWidth, int buffHeight
    );

    // Render pipeline
    static void cameraProjection();
    static void createDepthMap();
    static void rasterization();
};

// Pipeline Kernels
__global__ void cameraProjectionKernel(
    Vec4f *projection, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numVs
);
__global__ void createDepthMapKernel(
    // Mesh data
    Vec4f *projection, Vec3f *world, Vec3x3uli *faces, ULLInt numFs,
    // Buffer data
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary,
    int buffWidth, int buffHeight
);
__global__ void rasterizationKernel(
    // Mesh data
    Vec4f *color, Vec3f *world, Vec3f *normal, Vec2f *texture, UInt *meshID, Vec3x3uli *faces,
    // Buffer data
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    UInt *buffMeshId, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight
);

#endif