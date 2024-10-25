#ifndef FRAGMENTSHADER_CUH
#define FRAGMENTSHADER_CUH

#include <Graphic3D.cuh>

class FragmentShader {
public:
    static void applyTexture(); // Beta
    static void phongShading();
};

__global__ void applyTextureKernel( // Beta
    bool *buffActive, float *buffTu, float *buffTv,
    float *buffCr, float *buffCg, float *buffCb, float *buffCa,
    int buffWidth, int buffHeight,
    Vec3f *texture, int textureWidth, int textureHeight
);

__global__ void phongShadingKernel(
    bool *buffActive,
    float *buffWx, float *buffWy, float *buffWz,
    float *buffTu, float *buffTv,
    float *buffNx, float *buffNy, float *buffNz,
    float *buffCr, float *buffCg, float *buffCb, float *buffCa,
    int buffWidth, int buffHeight,

    LightSrc light
);

#endif