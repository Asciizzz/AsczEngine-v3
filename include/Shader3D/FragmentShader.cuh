#ifndef FRAGMENTSHADER_CUH
#define FRAGMENTSHADER_CUH

#include <VertexShader.cuh>

class FragmentShader {
public:
    // Singleton
    static FragmentShader& instance() {
        static FragmentShader instance;
        return instance;
    }
    FragmentShader(const FragmentShader&) = delete;
    FragmentShader &operator=(const FragmentShader&) = delete;

    void phongShading();

private:
    FragmentShader() {}
};

// Phong Shading
__global__ void phongShadingKernel(
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
);

#endif