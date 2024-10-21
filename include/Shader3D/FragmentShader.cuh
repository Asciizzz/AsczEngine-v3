#ifndef FRAGMENTSHADER_CUH
#define FRAGMENTSHADER_CUH

#include <Graphic3D.cuh>

class FragmentShader {
public:
    // Shading
    static void phongShading();

    // Custom Fragment Shader
    static void customFragmentShader();
};

// Phong Shading
__global__ void phongShadingKernel(
    LightSrc light,
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
);

// Custom Fragment Shader
__global__ void customFragmentShaderKernel(
    Vec3f *world, Vec3f *buffWorld, UInt *wObjId, UInt *buffWObjId,
    Vec3f *normal, Vec3f *buffNormal, UInt *nObjId, UInt *buffNObjId,
    Vec2f *texture, Vec2f *buffTexture, UInt *tObjId, UInt *buffTObjId,
    Vec4f *color, Vec4f *buffColor,
    Vec3x3ulli *faces, ULLInt *buffFaceId, Vec3f *bary, Vec3f *buffBary,
    bool *buffActive, float *buffDepth, int buffWidth, int buffHeight
);

#endif