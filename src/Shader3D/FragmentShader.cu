#include <FragmentShader.cuh>

// Phong Shading

void FragmentShader::phongShading() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;

    phongShadingKernel<<<buffer.blockNum, buffer.blockSize>>>(
        graphic.light,
        buffer.active, buffer.color, buffer.world, buffer.normal, buffer.texture,
        buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

__global__ void phongShadingKernel(
    LightSrc light,
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    // Apply colored light
    buffColor[i].x *= light.color.x;
    buffColor[i].y *= light.color.y;
    buffColor[i].z *= light.color.z;
    buffColor[i].limit(0, 255);

    // Find the light direction
    Vec3f lightDir = light.dir * -1;
    Vec3f n = buffNormal[i];

    // Calculate the cosine of the angle between the normal and the light direction
    float dot = n * lightDir;
    
    float cosA = dot / (n.mag() * lightDir.mag());
    if (cosA < 0) cosA = 0;

    float diff = light.ambient * (1 - cosA) + light.specular * cosA;

    // Apply the light
    buffColor[i].x *= diff;
    buffColor[i].y *= diff;
    buffColor[i].z *= diff;
    buffColor[i].limit(0, 255);
}