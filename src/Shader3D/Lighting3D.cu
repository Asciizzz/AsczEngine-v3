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

    float diff = 0.2 + 0.8 * cosA;

    // Apply the light
    buffColor[i] = buffColor[i] * diff;
}

// Light orthographic projection
void Lighting3D::lightProjection() {
    Render3D &render = Render3D::instance();

    lightProjectionKernel<<<render.mesh.blockNumVs, render.mesh.blockSize>>>(
        lightProj, render.mesh.meshID, render.mesh.world,
        shadowWidth, shadowHeight, render.mesh.numVs
    );
    cudaDeviceSynchronize();
}

__global__ void lightProjectionKernel(
    Vec4f *projection, UInt *meshID, Vec3f *world,
    int smWidth, int smHeight, ULLInt numVs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVs) return;

    /* What we will do:

    We will perform an insanely simple orthographic projection
    of the world coordinates of the vertices of the mesh.

    x, y will be limited in range [-400, 400] and z in [-100, 100].

    We will turn them into NDC coordinates and then into screen space.
    */

    Vec3f w = world[i];
    w.x = (w.x + 400) / 800 * smWidth;
    w.y = (w.y + 400) / 800 * smHeight;
    w.z = w.z / 400;
    w.z = (w.z + 1) / 2;

    projection[i] = w.toVec4f();
}

// Reset shadow map
void Lighting3D::resetShadowMap() {
    resetShadowMapKernel<<<shadowSize / 256 + 1, 256>>>(
        shadowDepth, shadowMeshID, shadowWidth, shadowHeight
    );
    cudaDeviceSynchronize();
}

__global__ void resetShadowMapKernel(
    float *shadowDepth, UInt *shadowMeshID, int smWidth, int smHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= smWidth * smHeight) return;

    shadowDepth[i] = 100;
    shadowMeshID[i] = NULL;
}

// Create shadow map
void Lighting3D::createShadowMap() {
    Render3D &render = Render3D::instance();

    createShadowMapKernel<<<render.mesh.blockNumFs, render.mesh.blockSize>>>(
        lightProj, render.mesh.faces, render.mesh.meshID, render.mesh.numFs,
        shadowDepth, shadowMeshID, shadowWidth, shadowHeight
    );
    cudaDeviceSynchronize();
}

__global__ void createShadowMapKernel(
    Vec4f *projection, Vec3uli *faces, UInt *meshID, ULLInt numFs,
    float *shadowDepth, UInt *shadowMeshID, int smWidth, int smHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli f = faces[i];
    Vec4f p0 = projection[f.x];
    Vec4f p1 = projection[f.y];
    Vec4f p2 = projection[f.z];

    // Bounding box
    int minX = min(min(p0.x, p1.x), p2.x);
    int maxX = max(max(p0.x, p1.x), p2.x);
    int minY = min(min(p0.y, p1.y), p2.y);
    int maxY = max(max(p0.y, p1.y), p2.y);

    // Clip the bounding box
    minX = max(minX, 0);
    maxX = min(maxX, smWidth - 1);
    minY = max(minY, 0);
    maxY = min(maxY, smHeight - 1);

    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
        int sIdx = x + y * smWidth;

        Vec3f bary = Vec3f::bary(
            Vec2f(x, y), Vec2f(p0.x, p0.y), Vec2f(p1.x, p1.y), Vec2f(p2.x, p2.y)
        );

        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float zDepth = bary.x * p0.z + bary.y * p1.z + bary.z * p2.z;

        if (atomicMinFloat(&shadowDepth[sIdx], zDepth)) {
            shadowMeshID[sIdx] = meshID[i];
            shadowDepth[sIdx] = zDepth;
        }
    }
}

// Apply shadow map
/*
We will go through each pixel of the buffer and check if the pixel is active.

To avoid shadow acne, we will add a bias based on the slope of the surface.

The slope is the dot product of the normal and the light direction.

*/
void Lighting3D::applyShadowMap() {
    Render3D &render = Render3D::instance();
    Buffer3D &buffer = render.buffer;

    applyShadowMapKernel<<<buffer.blockCount, buffer.blockSize>>>(
        buffer.active, buffer.color, buffer.world, buffer.normal, buffer.texture, buffer.meshID, buffer.width, buffer.height,
        shadowDepth, shadowMeshID, shadowWidth, shadowHeight
    );
    cudaDeviceSynchronize();
}

__global__ void applyShadowMapKernel(
    bool *buffActive, Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture, UInt *buffMeshID, int buffWidth, int buffHeight,
    float *shadowDepth, UInt *shadowMeshID, int smWidth, int smHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    Vec3f lightDir = Vec3f(0, 0, -1);
    Vec3f n = buffNormal[i];
    n.norm();

    float dot = n * lightDir;

    float slope = dot * 0.01;

    // Perform the same projection as in lightProjectionKernel
    // To get the pixel in the shadow map

    Vec3f w = buffWorld[i];
    w.x = (w.x + 400) / 800 * smWidth;
    w.y = (w.y + 400) / 800 * smHeight;
    w.z = w.z / 400;
    w.z = (w.z + 1) / 2;

    int sIdx = w.x + w.y * smWidth;

    // buffColor[i] = Vec4f(255, 255, 255, 255); // Debug

    if (shadowDepth[sIdx] < buffWorld[i].z) {
        buffColor[i].x *= 0.1;
        buffColor[i].y *= 0.1;
        buffColor[i].z *= 0.1;
    }
}