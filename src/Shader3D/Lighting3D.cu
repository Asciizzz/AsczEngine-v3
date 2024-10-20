#include <Lighting3D.cuh>

// Shadow Map allocation
void Lighting3D::allocateShadowMap(int width, int height) {
    shadowWidth = width;
    shadowHeight = height;
    shadowSize = width * height;

    cudaMalloc(&shadowDepth, shadowSize * sizeof(float));
    cudaMalloc(&shadowMeshID, shadowSize * sizeof(UInt));
}
void Lighting3D::freeShadowMap() {
    if (shadowDepth) cudaFree(shadowDepth);
    if (shadowMeshID) cudaFree(shadowMeshID);
}
void Lighting3D::resizeShadowMap(int width, int height) {
    freeShadowMap();
    allocateShadowMap(width, height);
}

// Light orthographic projection allocation
void Lighting3D::allocateLightProj() {
    Render3D &render = Render3D::instance();
    cudaMalloc(&lightProj, render.mesh.numVs * sizeof(Vec4f));
    cudaMalloc(&lightMeshID, render.mesh.numVs * sizeof(UInt));
}
void Lighting3D::freeLightProj() {
    if (lightProj) cudaFree(lightProj);
    if (lightMeshID) cudaFree(lightMeshID);
}
void Lighting3D::resizeLightProj() {
    freeLightProj();
    allocateLightProj();
}

// Phong Shading

void Lighting3D::phongShading() {
    Render3D &render = Render3D::instance();
    Buffer3D &buffer = render.buffer;

    phongShadingKernel<<<buffer.blockCount, buffer.blockSize>>>(
        buffer.active, buffer.color, buffer.world, buffer.normal, buffer.texture,
        buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

__global__ void phongShadingKernel(
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;
    
    // We will apply simple directional lighting
    Vec3f lightDir = Vec3f(0, 0, -1);
    Vec3f n = buffNormal[i];

    // Calculate the cosine of the angle between the normal and the light direction
    float dot = n * lightDir;
    
    float cosA = dot / (n.mag() * lightDir.mag());
    if (cosA < 0) cosA = 0;

    float diff = 0.4 + 0.8 * cosA;

    // Apply the light
    buffColor[i] = buffColor[i] * diff;
    buffColor[i].limit(0, 255);
}