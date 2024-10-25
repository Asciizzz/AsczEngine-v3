#include <FragmentShader.cuh>

// ======================== Static functions ========================

void FragmentShader::applyTexture() { // Beta
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;

    applyTextureKernel<<<buffer.blockNum, buffer.blockSize>>>(
        buffer.active,
        buffer.texture.x, buffer.texture.y,
        buffer.color.x, buffer.color.y, buffer.color.z, buffer.color.w,
        buffer.width, buffer.height,

        graphic.d_texture, graphic.textureWidth, graphic.textureHeight
    );
    cudaDeviceSynchronize();
}

void FragmentShader::phongShading() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;

    phongShadingKernel<<<buffer.blockNum, buffer.blockSize>>>(
        buffer.active,
        buffer.world.x, buffer.world.y, buffer.world.z,
        buffer.texture.x, buffer.texture.y,
        buffer.normal.x, buffer.normal.y, buffer.normal.z,
        buffer.color.x, buffer.color.y, buffer.color.z, buffer.color.w,
        buffer.width, buffer.height,

        graphic.light
    );
    cudaDeviceSynchronize();
}

// ======================== Kernels ========================

__global__ void applyTextureKernel( // Beta
    bool *buffActive, float *buffTu, float *buffTv,
    float *buffCr, float *buffCg, float *buffCb, float *buffCa,
    int buffWidth, int buffHeight,
    Vec3f *texture, int textureWidth, int textureHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    int x = buffTu[i] * textureWidth;
    int y = buffTv[i] * textureHeight;
    int tIdx = x + y * textureWidth;

    if (tIdx >= textureWidth * textureHeight ||
        tIdx < 0) return;

    buffCr[i] = texture[tIdx].x;
    buffCg[i] = texture[tIdx].y;
    buffCb[i] = texture[tIdx].z;
}

__global__ void phongShadingKernel(
    bool *buffActive,
    float *buffWx, float *buffWy, float *buffWz,
    float *buffTu, float *buffTv,
    float *buffNx, float *buffNy, float *buffNz,
    float *buffCr, float *buffCg, float *buffCb, float *buffCa,
    int buffWidth, int buffHeight,

    LightSrc light
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    buffCr[i] *= light.color.x;
    buffCg[i] *= light.color.y;
    buffCb[i] *= light.color.z;

    // Find the light direction
    Vec3f lightDir = light.dir * -1;
    Vec3f n = Vec3f(buffNx[i], buffNy[i], buffNz[i]);

    // Calculate the cosine of the angle between the normal and the light direction
    float dot = n * lightDir;
    
    float cosA = dot / (n.mag() * lightDir.mag());
    if (cosA < 0) cosA = 0;

    float diff = light.ambient * (1 - cosA) + light.specular * cosA;

    // Apply the light
    buffCr[i] *= diff;
    buffCg[i] *= diff;
    buffCb[i] *= diff;

    // Limit the color
    buffCr[i] = fminf(fmaxf(buffCr[i], 0), 255);
    buffCg[i] = fminf(fmaxf(buffCg[i], 0), 255);
    buffCb[i] = fminf(fmaxf(buffCb[i], 0), 255);
}