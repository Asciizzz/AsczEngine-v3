#ifndef FRAGMENTSHADER_CUH
#define FRAGMENTSHADER_CUH

#include <Graphic3D.cuh>

class FragmentShader {
public:
    // Shading
    static void phongShading();
};

// Phong Shading
__global__ void phongShadingKernel(
    LightSrc light,
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
);

#endif