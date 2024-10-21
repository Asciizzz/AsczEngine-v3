#include <FragmentShader.cuh>

// Phong Shading

void FragmentShader::phongShading() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;

    phongShadingKernel<<<buffer.blockCount, buffer.blockSize>>>(
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

// BETA FEATURES

// Light Projection

void FragmentShader::lightProjection() {
    Graphic3D &graphic = Graphic3D::instance();
    Mesh3D &mesh = graphic.mesh;
    Vec3f *lightProj = graphic.lightProj;

    lightProjectionKernel<<<mesh.blockNumWs, mesh.blockSize>>>(
        lightProj, mesh.world, graphic.sWidth, graphic.sHeight, mesh.numWs
    );
    cudaDeviceSynchronize();
}

__global__ void lightProjectionKernel(
    Vec3f *lightProj, Vec3f *world, int sWidth, int sHeight, ULLInt numWs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numWs) return;

    // Convert every point in the x y z range of [-100, 100] to NDC
    // The projection is super simple, we basically project onto the xy plane
    Vec3f w = world[i];
    w.x = (w.x + 100) / 200 * sWidth;
    w.y = (w.y + 100) / 200 * sHeight;
    w.z = (w.z + 100) / 200;

    lightProj[i] = w;
}

// Reset Shadow Depth Map

void FragmentShader::resetShadowDepthMap() {
    Graphic3D &graphic = Graphic3D::instance();

    int blockCount = (graphic.sSize + 255) / 256;

    resetShadowDepthMapKernel<<<blockCount, 256>>>(
        graphic.shadowActive, graphic.shadowDepth, graphic.sWidth, graphic.sHeight
    );
    cudaDeviceSynchronize();
}

__global__ void resetShadowDepthMapKernel(
    bool *shadowActive, float *shadowDepth, int sWidth, int sHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sWidth * sHeight) return;

    shadowActive[i] = false;
    shadowDepth[i] = 10;
}

// Create Shadow Depth Map

void FragmentShader::createShadowDepthMap() {
    Graphic3D &graphic = Graphic3D::instance();
    Mesh3D &mesh = graphic.mesh;
    Vec3f *lightProj = graphic.lightProj;
    bool *shadowActive = graphic.shadowActive;
    float *shadowDepth = graphic.shadowDepth;

    createShadowDepthMapKernel<<<mesh.blockNumFs, mesh.blockSize>>>(
        lightProj, mesh.world, mesh.faces, mesh.numFs,
        shadowActive, shadowDepth, graphic.sWidth, graphic.sHeight
    );
    cudaDeviceSynchronize();
}

__global__ void createShadowDepthMapKernel(
    Vec3f *lightProj, Vec3f *world, Vec3x3uli *faces, ULLInt numFs,
    bool *shadowActive, float *shadowDepth, int sWidth, int sHeight
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli fv = faces[i].v;
    Vec3f lp0 = lightProj[fv.x];
    Vec3f lp1 = lightProj[fv.y];
    Vec3f lp2 = lightProj[fv.z];

    // Bounding box
    int minX = min(lp0.x, min(lp1.x, lp2.x));
    int maxX = max(lp0.x, max(lp1.x, lp2.x));
    int minY = min(lp0.y, min(lp1.y, lp2.y));
    int maxY = max(lp0.y, max(lp1.y, lp2.y));

    // Clip the bounding box
    minX = max(minX, 0);
    maxX = min(maxX, sWidth - 1);
    minY = max(minY, 0);
    maxY = min(maxY, sHeight - 1);

    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
        int sIdx = x + y * sWidth;

        Vec3f bary = Vec3f::bary(
            Vec2f(x, y),
            Vec2f(lp0.x, lp0.y),
            Vec2f(lp1.x, lp1.y),
            Vec2f(lp2.x, lp2.y)
        );

        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float depth = lp0.z * bary.x + lp1.z * bary.y + lp2.z * bary.z;

        if (atomicMinFloat(&shadowDepth[sIdx], depth)) {
            shadowActive[sIdx] = true;
            shadowDepth[sIdx] = depth;
        }
    }
}

// Apply Shadow Map

void FragmentShader::applyShadowMap() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;

    applyShadowMapKernel<<<buffer.blockCount, buffer.blockSize>>>(
        buffer.active, buffer.world, buffer.color, buffer.width, buffer.height,
        graphic.shadowActive, graphic.shadowDepth, graphic.sWidth, graphic.sHeight
    );
    cudaDeviceSynchronize();
}

__global__ void applyShadowMapKernel(
    bool *buffActive, Vec3f *buffWorld, Vec4f *buffColor, int buffWidth, int buffHeight,
    bool *shadowActive, float *shadowDepth, int sWidth, int sHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    Vec3f w = buffWorld[i];
    w.x = (w.x + 100) / 200 * sWidth;
    w.y = (w.y + 100) / 200 * sHeight;
    w.z = (w.z + 100) / 200;

    int sIdx = w.x + w.y * sWidth;

    if (shadowActive[sIdx] && shadowDepth[sIdx] < w.z) {
        buffColor[i].x *= 0.5;
        buffColor[i].y *= 0.5;
        buffColor[i].z *= 0.5;
    }
}