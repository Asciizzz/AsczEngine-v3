#ifndef FRAGMENTSHADER_CUH
#define FRAGMENTSHADER_CUH

#include <Graphic3D.cuh>

class FragmentShader {
public:
    // Shading
    static void phongShading();

    // Shadow Mapping
    static void lightProjection();
    static void resetShadowDepthMap();
    static void createShadowDepthMap();
    static void applyShadowMap();
};

// Phong Shading
__global__ void phongShadingKernel(
    LightSrc light,
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
);

// Shadow Mapping
__global__ void lightProjectionKernel(
    Vec3f *lightProj, Vec3f *world, int sWidth, int sHeight, ULLInt numWs
);
__global__ void resetShadowDepthMapKernel(
    bool *shadowActive, float *shadowDepth, int sWidth, int sHeight
);
__global__ void createShadowDepthMapKernel(
    Vec3f *lightProj, Vec3f *world, Vec3x3uli *faces, ULLInt numFs,
    bool *shadowActive, float *shadowDepth, int sWidth, int sHeight
);
// Check every pixel in the buffer and compare it with the shadow map
__global__ void applyShadowMapKernel(
    bool *buffActive, Vec3f *buffWorld, Vec4f *buffColor, int buffWidth, int buffHeight,
    bool *shadowActive, float *shadowDepth, int sWidth, int sHeight
);

#endif