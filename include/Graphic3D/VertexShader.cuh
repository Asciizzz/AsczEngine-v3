#ifndef VERTEXSHADER_CUH
#define VERTEXSHADER_CUH

#include <Graphic3D.cuh>

/* IMPORTANT NOTES REGARDING TEXTURE MAP

Instead of interpolating the texture coordinates in the fragment shader,
We will interpolate u/w and v/w in the vertex shader, and then divide them
by interpolated 1/w in the fragment shader. This will ensure that the texture
have a correct perspective correction.

*/

struct Plane { // Ax + By + Cz + D = 0
    float a, b, c, d;

    __device__ Plane(Vec3f v1, Vec3f v2, Vec3f v3) {
        Vec3f n = (v2 - v1) & (v3 - v1);
        n.norm();
        a = n.x;
        b = n.y;
        c = n.z;
        d = -(a * v1.x + b * v1.y + c * v1.z);
    }

    __device__ float equation(Vec3f v) {
        return a * v.x + b * v.y + c * v.z + d;
    }
};

struct Vertex {
    Vec4f screen;
    Vec3f ndc;
    Vec3f world;
    Vec2f texture;
    Vec3f normal;
    Vec4f color;
};

class VertexShader {
public:
    // Render pipeline
    static void cameraProjection();
    static void createRuntimeFaces();
    static void frustumCulling();
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

__global__ void frustumCullingKernel(
    const float *runtimeSx, const float *runtimeSy, const float *runtimeSz, const float *runtimeSw,
    const float *runtimeWx, const float *runtimeWy, const float *runtimeWz,
    const float *runtimeTu, const float *runtimeTv,
    const float *runtimeNx, const float *runtimeNy, const float *runtimeNz,
    const float *runtimeCr, const float *runtimeCg, const float *runtimeCb, const float *runtimeCa,
    ULLInt faceCounter,

    float *cullSx, float *cullSy, float *cullSz, float *cullSw,
    float *cullWx, float *cullWy, float *cullWz,
    float *cullTu, float *cullTv,
    float *cullNx, float *cullNy, float *cullNz,
    float *cullCr, float *cullCg, float *cullCb, float *cullCa,
    ULLInt *cullCounter
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