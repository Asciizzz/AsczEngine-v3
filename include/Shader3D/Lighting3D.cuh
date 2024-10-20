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

private:
    Lighting3D() {}
};

// Phong Shading
__global__ void phongShadingKernel(
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
);

#endif