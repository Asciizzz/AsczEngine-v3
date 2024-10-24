#ifndef VERTEXSHADER_CUH
#define VERTEXSHADER_CUH

#include <Graphic3D.cuh>

/* EXPLANATION FOR RUNTIME FACES:

Faces in mesh are indices to vertices, normals, textures, and colors.

Faces in runtime are face objects that contain the actual vertices, normals, textures, and colors.

The reason for this is due to the introduction of new faces after clipping and splitting. The indices system of the mesh cannot handle this, so we need actual face objects.

The createRuntimeFaces will perform 2 main tasks:

- Convert the indices to actual vertices, normals, textures, and colors.
- Clip and split the faces for intersection with the camera frustum.

*/

class VertexShader {
public:
    // Render functions
    __host__ __device__ static Vec4f toScreenSpace(
        Camera3D &camera, Vec3f world, int buffWidth, int buffHeight
    );

    // Render pipeline
    static void cameraProjection();
    static void createRuntimeFaces();
    static void createDepthMap();
    static void rasterization();
};

// Camera projection (MVP) kernel
__global__ void cameraProjectionKernel(
    Vec4f *screen, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numWs
);

// Filter visible faces
__global__ void createRuntimeFacesKernel(
    Vec4f *screen, Vec3f *world, Vec3f *normal, Vec2f *texture, Vec4f *color,
    Vec3ulli *faceWs, Vec3ulli *faceNs, Vec3ulli *faceTs, ULLInt numFs,
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
    Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture, Vec4f *buffColor,
    Face3D *runtimeFaces, ULLInt *buffFaceId, Vec3f *buffBary,
    bool *buffActive, int buffWidth, int buffHeight
);

#endif