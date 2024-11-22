#ifndef FRAGMENTSHADER_CUH
#define FRAGMENTSHADER_CUH

#include <Graphic3D.cuh>

class FragmentShader {
public:
    static void applyMaterial(); // Beta
    static void phongShading();

    // Adding shadow mapping (Extremely experimental)
    static void resetShadowMap();
    static void createShadowMap();
    static void applyShadowMap();

    static void customShader();
};

__global__ void applyMaterialKernel( // Beta
    // Mesh material
    float *kar, float *kag, float *kab,
    float *kdr, float *kdg, float *kdb,
    float *ksr, float *ksg, float *ksb,
    LLInt *mkd,
    // Mesh texture
    float *txr, float *txg, float *txb,
    int *txw, int *txh, LLInt *txof,
    // Buffer
    bool *bActive, LLInt *bMidx,
    float *bCr, float *bCg, float *bCb, float *bCa,
    float *bTu, float *bTv,
    int bWidth, int bHeight
);

__global__ void phongShadingKernel(
    bool *bActive,
    float *bWx, float *bWy, float *bWz,
    float *bTu, float *bTv,
    float *bNx, float *bNy, float *bNz,
    float *bCr, float *bCg, float *bCb, float *bCa,
    int bWidth, int bHeight,

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
    bool *bActive,
    float *bWx, float *bWy, float *bWz,
    float *bNx, float *bNy, float *bNz,
    float *bCr, float *bCg, float *bCb, float *bCa,
    int bWidth, int bHeight,

    float *shadowDepth, int shdwWidth, int shdwHeight
);

__global__ void customShaderKernel(
    bool *bActive, float *bDepth,
    ULLInt *bFidx, 
    float *bBrx, float *bBry, float *bBrz, // Bary
    float *bWx, float *bWy, float *bWz, // World
    float *bTu, float *bTv, // Texture
    float *bNx, float *bNy, float *bNz, // Normal
    float *bCr, float *bCg, float *bCb, float *bCa, // Color
    int bWidth, int bHeight
);

#endif