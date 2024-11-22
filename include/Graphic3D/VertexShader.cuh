#ifndef VERTEXSHADER_CUH
#define VERTEXSHADER_CUH

#include <Graphic3D.cuh>

/* IMPORTANT NOTES REGARDING TEXTURE MAP

Instead of interpolating the texture coordinates in the fragment shader,
We will interpolate u/w and v/w in the vertex shader, and then divide them
by interpolated 1/w in the fragment shader. This will ensure that the texture
have a correct perspective correction.

ADDITIONAL NOTE:

This applies for EVERYTHING that needs to be perspective corrected:
- Normal
- Color
- World position

*/

struct VertexTemp {
    int num = 0;
    Vec4f s[6];
    Vec3f w[6];
    Vec2f t[6];
    Vec3f n[6];
    Vec4f c[6];
};

class VertexShader {
public:
    __device__ static bool insideFrustum(const Vec4f &v);

    // Render pipeline
    static void cameraProjection();
    static void frustumCulling();
    static void createDepthMap();
    static void rasterization();
};

// Camera projection
__global__ void cameraProjectionKernel(
    const float *wx, const float *wy, const float *wz,
    float *sx, float *sy, float *sz, float *sw,
    Mat4f mvp, ULLInt numVs
);

// Create runtime faces
__global__ void frustumCullingKernel(
    // Orginal mesh data
    const float *sx, const float *sy, const float *sz, const float *sw,
    const float *wx, const float *wy, const float *wz,
    const float *tu, const float *tv,
    const float *nx, const float *ny, const float *nz,
    const ULLInt *fWs, const LLInt *fTs, const LLInt *fNs,
    const LLInt *fMs, const bool *fAs, ULLInt numFs,

    // Runtime faces
    float *rtSx, float *rtSy, float *rtSz, float *rtSw,
    float *rtWx, float *rtWy, float *rtWz,
    float *rtTu, float *rtTv,
    float *rtNx, float *rtNy, float *rtNz,
    bool *rtActive, LLInt *rtMat, float *rtArea
);

__global__ void runtimeIndexingKernel(
    const bool *rtActive, const float *rtArea, ULLInt numFs,
    ULLInt *rtIndex1, ULLInt *d_rtCount1,
    ULLInt *rtIndex2, ULLInt *d_rtCount2
);

// Tile-based depth map creation
__global__ void createDepthMapKernel(
    const ULLInt *rtIndex,
    const bool *rtActive, const float *rtSx, const float *rtSy, const float *rtSz, const float *rtSw,
    ULLInt fCount, ULLInt fOffset,
    bool *bActive, float *bDepth, ULLInt *bFidx,
    float *bBrX, float *bBrY, float *bBrZ,
    int bWidth, int bHeight, int tNumX, int tNumY, int tSizeX, int tSizeY
);

// Fill the buffer with datas
__global__ void rasterizationKernel(
    const float *rtSw, const LLInt *rtMat,
    const float *rtWx, const float *rtWy, const float *rtWz,
    const float *rtTu, const float *rtTv,
    const float *rtNx, const float *rtNy, const float *rtNz,

    const bool *bActive, const ULLInt *bFidx,
    LLInt *bMidx,  // Material
    float *bBrx, float *bBry, float *bBrz, // Bary
    float *bWx, float *bWy, float *bWz, // World
    float *bTu, float *bTv, // Texture
    float *bNx, float *bNy, float *bNz, // Normal
    int bWidth, int bHeight
);

#endif