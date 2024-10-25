#ifndef VERTEXSHADER_CUH
#define VERTEXSHADER_CUH

#include <Graphic3D.cuh>

class VertexShader {
public:
    // Render pipeline
    static void cameraProjection();
    static void createRuntimeFaces();
    static void createDepthMap();
    static void createDepthMapBeta();
    static void rasterization();
};

// Camera projection (MVP) kernel
__global__ void cameraProjectionKernel(
    float *screenX, float *screenY, float *screenZ, float *screenW,
    float *worldX, float *worldY, float *worldZ,
    Mat4f mvp, ULLInt numWs
);

// Filter visible faces
__global__ void createRuntimeFacesKernel(
    float *screenX, float *screenY, float *screenZ, float *screenW,
    float *worldX, float *worldY, float *worldZ,
    float *normalX, float *normalY, float *normalZ,
    float *textureX, float *textureY,
    float *colorX, float *colorY, float *colorZ, float *colorW,

    ULLInt *faceWs, ULLInt *faceTs, ULLInt *faceNs, ULLInt numFs,
    Face3D *runtimeFaces, ULLInt *faceCounter
);

// Tile-based depth map creation (using nested parallelism, or dynamic parallelism)
__global__ void createDepthMapKernel(
    Face3D *runtimeFaces, ULLInt faceCounter,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight,
    int tileNumX, int tileNumY, int tileWidth, int tileHeight
);

// Fill the buffer with datas
__global__ void rasterizationKernel(
    Face3D *runtimeFaces, ULLInt *buffFaceId,
    Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture, Vec4f *buffColor,
    bool *buffActive, Vec3f *buffBary, int buffWidth, int buffHeight
);

#endif