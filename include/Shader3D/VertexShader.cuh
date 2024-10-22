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
    static void filterVisibleFaces();
    static void createDepthMap();
    static void rasterization();
};

// Camera projection (MVP) kernel
__global__ void cameraProjectionKernel(
    Vec4f *screen, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numWs
);

// Filter visible faces
__global__ void filterVisibleFacesKernel(
    Vec4f *screen,
    Vec3ulli *faceWs, Vec3ulli *faceNs, Vec3ulli *faceTs, ULLInt numFs,
    Vec3ulli *visibFWs, Vec3ulli *visibFNs, Vec3ulli *visibFTs, ULLInt *numVisibFs
);

// Tile-based depth map creation (using nested parallelism, or dynamic parallelism)
__global__ void createDepthMapKernel(
    Vec4f *screen, Vec3f *world, Vec3ulli *visibFWs, ULLInt numVisibFs,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight,
    int tileNumX, int tileNumY, int tileWidth, int tileHeight
);

// Fill the buffer with datas
__global__ void rasterizationKernel(
    // World data
    Vec3f *world, Vec3f *buffWorld, UInt *wObjId, UInt *buffWObjId,
    // Normal data
    Vec3f *normal, Vec3f *buffNormal, UInt *nObjId, UInt *buffNObjId,
    // Texture data
    Vec2f *texture, Vec2f *buffTexture, UInt *tObjId, UInt *buffTObjId,
    // Color data (shared with world for now)
    Vec4f *color, Vec4f *buffColor,
    // Face data
    Vec3ulli *visibFWs, Vec3ulli *visibFNs, Vec3ulli *visibFTs, ULLInt *buffFaceId,
    // Barycentric data
    Vec3f *bary, Vec3f *buffBary,
    // Other data
    bool *buffActive, int buffWidth, int buffHeight
);

#endif