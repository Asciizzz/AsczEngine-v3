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

/* CREATOR NOTE:

PLEASE CLIP FIRST, PROJECT LATER

THERE IS LITERALLY NO REASON TO INTERPOLATE
SCREEN COORDINATES INSIDE THE CLIPPING KERNEL
*/

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
    const float *cr, const float *cg, const float *cb, const float *ca,
    const ULLInt *fWs, const ULLInt *fTs, const ULLInt *fNs, ULLInt numFs,

    // Runtime faces
    float *rtSx, float *rtSy, float *rtSz, float *rtSw,
    float *rtWx, float *rtWy, float *rtWz,
    float *rtTu, float *rtTv,
    float *rtNx, float *rtNy, float *rtNz,
    float *rtCr, float *rtCg, float *rtCb, float *rtCa,
    bool *rtActive
);

__global__ void filterRuntimeKernel(
    float *rtSx1, float *rtSy1, float *rtSz1, float *rtSw1,
    float *rtWx1, float *rtWy1, float *rtWz1,
    float *rtTu1, float *rtTv1,
    float *rtNx1, float *rtNy1, float *rtNz1,
    float *rtCr1, float *rtCg1, float *rtCb1, float *rtCa1,
    bool *rtActive1, ULLInt numFs1,

    float *rtSx2, float *rtSy2, float *rtSz2, float *rtSw2,
    float *rtWx2, float *rtWy2, float *rtWz2,
    float *rtTu2, float *rtTv2,
    float *rtNx2, float *rtNy2, float *rtNz2,
    float *rtCr2, float *rtCg2, float *rtCb2, float *rtCa2,
    bool *rtActive2, ULLInt *d_rtCount
);

// Tile-based depth map creation
__global__ void createDepthMapKernel(
    const bool *rtActive, const float *rtSx, const float *rtSy, const float *rtSz, const float *rtSw,
    ULLInt faceCounter, ULLInt faceOffset,
    bool *bActive, float *bDepth, ULLInt *bFaceId,
    float *bBrX, float *bBrY, float *bBrZ,
    int bWidth, int bHeight, int tNumX, int tNumY, int tSizeX, int tSizeY
);

// Fill the buffer with datas
__global__ void rasterizationKernel(
    const float *rtSw,
    const float *rtWx, const float *rtWy, const float *rtWz,
    const float *rtTu, const float *rtTv,
    const float *rtNx, const float *rtNy, const float *rtNz,
    const float *rtCr, const float *rtCg, const float *rtCb, const float *rtCa,

    const bool *bActive, const ULLInt *bFaceId,
    float *bBrx, float *bBry, float *bBrz, // Bary
    float *bWx, float *bWy, float *bWz, // World
    float *bTu, float *bTv, // Texture
    float *bNx, float *bNy, float *bNz, // Normal
    float *bCr, float *bCg, float *bCb, float *bCa, // Color
    int bWidth, int bHeight
);

#endif