#ifndef FRAGMENTSHADER_CUH
#define FRAGMENTSHADER_CUH

#include <Graphic3D.cuh>

class FragmentShader {
public:
    static void applyTexture(); // Beta
    static void phongShading();

    // Adding shadow mapping (Extremely experimental)
    static void resetShadowMap();
    static void createShadowMap();
    static void applyShadowMap();

    static void customShader();
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

/* Demo:
we will create a shadow map for a light source
that is perpendicular to the xy-plane.
*/
__global__ void resetShadowMapKernel(
    float *shadowDepth, int shdwWidth, int shdwHeight
);
__global__ void createShadowMapKernel(
    const float *worldX, const float *worldY, const float *worldZ,
    const ULLInt *faceWs, ULLInt numFs,
    float *shadowDepth, int shdwWidth, int shdwHeight,
    int shdwTileNumX, int shdwTileNumY, int shdwTileSizeX, int shdwTileSizeY
);
__global__ void applyShadowMapKernel(
    bool *buffActive,
    float *buffWx, float *buffWy, float *buffWz,
    float *buffNx, float *buffNy, float *buffNz,
    float *buffCr, float *buffCg, float *buffCb, float *buffCa,
    int buffWidth, int buffHeight,

    float *shadowDepth, int shdwWidth, int shdwHeight
);

__global__ void customShaderKernel(
    bool *buffActive, ULLInt *buffFaceId, float *buffDepth,
    float *buffBrx, float *buffBry, float *buffBrz, // Bary
    float *buffWx, float *buffWy, float *buffWz, // World
    float *buffTu, float *buffTv, // Texture
    float *buffNx, float *buffNy, float *buffNz, // Normal
    float *buffCr, float *buffCg, float *buffCb, float *buffCa, // Color
    int buffWidth, int buffHeight
);

#endif