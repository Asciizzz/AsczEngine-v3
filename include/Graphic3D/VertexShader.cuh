#ifndef VERTEXSHADER_CUH
#define VERTEXSHADER_CUH

#include <Graphic3D.cuh>

/* IMPORTANT NOTES REGARDING TEXTURE MAP

Instead of interpolating the texture coordinates in the fragment shader,
We will interpolate u/w and v/w in the vertex shader, and then divide them
by interpolated 1/w in the fragment shader. This will ensure that the texture
have a correct perspective correction.

*/

class VertexShader {
public:
    // Render pipeline
    static void cameraProjection();
    static void createRuntimeFaces();
    static void frustumClipping();
    static void createDepthMap();
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
    const float *screenX, const float *screenY, const float *screenZ, const float *screenW,
    const float *worldX, const float *worldY, const float *worldZ,
    const float *normalX, const float *normalY, const float *normalZ,
    const float *textureX, const float *textureY,
    const float *colorX, const float *colorY, const float *colorZ, float *colorW,
    const ULLInt *faceWs, const ULLInt *faceTs, const ULLInt *faceNs, ULLInt numFs,

    float *runtimeSx, float *runtimeSy, float *runtimeSz, float *runtimeSw,
    float *runtimeWx, float *runtimeWy, float *runtimeWz,
    float *runtimeTu, float *runtimeTv,
    float *runtimeNx, float *runtimeNy, float *runtimeNz,
    float *runtimeCr, float *runtimeCg, float *runtimeCb, float *runtimeCa,
    ULLInt *faceCounter
);

// Frustum clipping
__global__ void clipFrustumKernel(
    const float *preSx, const float *preSy, const float *preSz, const float *preSw,
    const float *preWx, const float *preWy, const float *preWz,
    const float *preTu, const float *preTv,
    const float *preNx, const float *preNy, const float *preNz,
    const float *preCr, const float *preCg, const float *preCb, const float *preCa,
    ULLInt *preCounter,

    float *postSx, float *postSy, float *postSz, float *postSw,
    float *postWx, float *postWy, float *postWz,
    float *postTu, float *postTv,
    float *postNx, float *postNy, float *postNz,
    float *postCr, float *postCg, float *postCb, float *postCa,
    ULLInt *postCounter,

    Plane3D plane
);

// Tile-based depth map creation
__global__ void createDepthMapKernel(
    const float *runtimeSx, const float *runtimeSy, const float *runtimeSz, const float *runtimeSw, ULLInt faceCounter, ULLInt faceOffset,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId,
    float *buffBaryX, float *buffBaryY, float *buffBaryZ,
    int buffWidth, int buffHeight, int tileNumX, int tileNumY, int tileSizeX, int tileSizeY
);

// Fill the buffer with datas
__global__ void rasterizationKernel(
    const float *runtimeSw,
    const float *runtimeWx, const float *runtimeWy, const float *runtimeWz,
    const float *runtimeTu, const float *runtimeTv,
    const float *runtimeNx, const float *runtimeNy, const float *runtimeNz,
    const float *runtimeCr, const float *runtimeCg, const float *runtimeCb, const float *runtimeCa,

    const bool *buffActive, const ULLInt *buffFaceId,
    float *buffBrx, float *buffBry, float *buffBrz, // Bary
    float *buffWx, float *buffWy, float *buffWz, // World
    float *buffTu, float *buffTv, // Texture
    float *buffNx, float *buffNy, float *buffNz, // Normal
    float *buffCr, float *buffCg, float *buffCb, float *buffCa, // Color
    int buffWidth, int buffHeight
);

#endif