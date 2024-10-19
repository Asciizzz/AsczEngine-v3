#ifndef LIGHTING3D_CUH
#define LIGHTING3D_CUH

#include <Render3D.cuh>

class Lighting3D {
public:
    // Singleton
    static Lighting3D& instance() {
        static Lighting3D instance;
        return instance;
    }
    Lighting3D(const Lighting3D&) = delete;
    Lighting3D &operator=(const Lighting3D&) = delete;

    // Shadow Map
    float *shadowDepth;
    UInt *shadowMeshID;
    int shadowWidth, shadowHeight, shadowSize;

    void allocateShadowMap(int width, int height);
    void freeShadowMap();
    void resizeShadowMap(int width, int height);

    // Light orthographic projection
    Vec4f *lightProj;
    UInt *lightMeshID;
    void allocateLightProj();
    void freeLightProj();
    void resizeLightProj();

    void phongShading();
    void lightProjection();
    void resetShadowMap();
    void createShadowMap();
    void applyShadowMap();

private:
    Lighting3D() {}
};

// Phong Shading
__global__ void phongShadingKernel(
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
);

// To orthographic projection
__global__ void lightProjectionKernel(
    Vec4f *projection, UInt *meshID, Vec3f *world,
    int smWidth, int smHeight, ULLInt numVs
);

// Reset shadow map
__global__ void resetShadowMapKernel(
    float *shadowDepth, UInt *shadowMeshID, int smWidth, int smHeight
);

// Create shadow map
__global__ void createShadowMapKernel(
    Vec4f *projection, Vec3uli *faces, UInt *meshID, ULLInt numFs,
    float *shadowDepth, UInt *shadowMeshID, int smWidth, int smHeight
);

// Apply shadow map
__global__ void applyShadowMapKernel(
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture, UInt *buffMeshID, int buffWidth, int buffHeight,
    float *shadowDepth, UInt *shadowMeshID, int smWidth, int smHeight
);

#endif