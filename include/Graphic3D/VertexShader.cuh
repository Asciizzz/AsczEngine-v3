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
    static void createRuntimeFaces();
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
__global__ void createRuntimeFacesKernel(
    // Orginal mesh data
    const float *sx, const float *sy, const float *sz, const float *sw,
    const float *wx, const float *wy, const float *wz,
    const float *nx, const float *ny, const float *nz,
    const float *tu, const float *tv,
    const float *cr, const float *cg, const float *cb, const float *ca,
    const ULLInt *fWs, const ULLInt *fTs, const ULLInt *fNs, ULLInt numFs,

    // Runtime faces
    float *rtSx, float *rtSy, float *rtSz, float *rtSw,
    float *rtWx, float *rtWy, float *rtWz,
    float *rtTu, float *rtTv,
    float *rtNx, float *rtNy, float *rtNz,
    float *rtCr, float *rtCg, float *rtCb, float *rtCa,
    bool *rtActive, ULLInt *rtCount
);

// Tile-based depth map creation
__global__ void createDepthMapKernel(
    const bool *runtimeActive,
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