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
    Vec4f *screen, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numWs
);

// Tile-based depth map creation (using nested parallelism, or dynamic parallelism)
__global__ void createDepthMapKernel(
    Vec4f *screen, Vec3f *world, Vec3x3uli *faces, ULLInt numFs,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight,
    int tileNumX, int tileNumY, int tileWidth, int tileHeight
);

__global__ void rasterizationKernel(
    // World data
    Vec3f *world, Vec3f *buffWorld, UInt *wMeshId, UInt *buffWMeshId,
    // Normal data
    Vec3f *normal, Vec3f *buffNormal, UInt *nMeshId, UInt *buffNMeshId,
    // Texture data
    Vec2f *texture, Vec2f *buffTexture, UInt *tMeshId, UInt *buffTMeshId,
    // Color data (shared with world for now)
    Vec4f *color, Vec4f *buffColor,
    // Face data
    Vec3x3uli *faces, ULLInt *buffFaceId, Vec3f *bary, Vec3f *buffBary,
    // Other data
    bool *buffActive, int buffWidth, int buffHeight
);

#endif