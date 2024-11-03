#include <FragmentShader.cuh>

// ======================== Static functions ========================

void FragmentShader::applyTexture() { // Beta
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    applyTextureKernel<<<buffer.blockNum, buffer.blockSize>>>(
        buffer.active,
        buffer.texture.x, buffer.texture.y,
        buffer.color.x, buffer.color.y, buffer.color.z, buffer.color.w,
        buffer.width, buffer.height,

        grphic.d_texture, grphic.textureWidth, grphic.textureHeight
    );
    cudaDeviceSynchronize();
}

void FragmentShader::phongShading() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    phongShadingKernel<<<buffer.blockNum, buffer.blockSize>>>(
        buffer.active,
        buffer.world.x, buffer.world.y, buffer.world.z,
        buffer.texture.x, buffer.texture.y,
        buffer.normal.x, buffer.normal.y, buffer.normal.z,
        buffer.color.x, buffer.color.y, buffer.color.z, buffer.color.w,
        buffer.width, buffer.height,

        grphic.light
    );
    cudaDeviceSynchronize();
}

void FragmentShader::resetShadowMap() {
    Graphic3D &grphic = Graphic3D::instance();

    int blockNum = (grphic.shdwWidth * grphic.shdwHeight + 255) / 256;
    resetShadowMapKernel<<<blockNum, 256>>>(
        grphic.shadowDepth, grphic.shdwWidth, grphic.shdwHeight
    );
    cudaDeviceSynchronize();
}

void FragmentShader::createShadowMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Mesh3D &mesh = grphic.mesh;

    dim3 blockSize(8, 32);

    size_t blockNumTile = (grphic.shdwTileNum + blockSize.x - 1) / blockSize.x;
    size_t blockNumFace = (mesh.faces.size / 3 + blockSize.y - 1) / blockSize.y;
    dim3 blockNum(blockNumTile, blockNumFace);

    createShadowMapKernel<<<blockNum, blockSize>>>(
        mesh.world.x, mesh.world.y, mesh.world.z,
        mesh.faces.v, mesh.faces.size / 3,
        grphic.shadowDepth, grphic.shdwWidth, grphic.shdwHeight,
        grphic.shdwTileNumX, grphic.shdwTileNumY, grphic.shdwTileSizeX, grphic.shdwTileSizeY
    );
    cudaDeviceSynchronize();
}

void FragmentShader::applyShadowMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    applyShadowMapKernel<<<buffer.blockNum, buffer.blockSize>>>(
        buffer.active,
        buffer.world.x, buffer.world.y, buffer.world.z,
        buffer.normal.x, buffer.normal.y, buffer.normal.z,
        buffer.color.x, buffer.color.y, buffer.color.z, buffer.color.w,
        buffer.width, buffer.height,

        grphic.shadowDepth, grphic.shdwWidth, grphic.shdwHeight
    );
    cudaDeviceSynchronize();
}

void FragmentShader::customShader() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    customShaderKernel<<<buffer.blockNum, buffer.blockSize>>>(
        buffer.active, buffer.faceID, buffer.depth,
        buffer.bary.x, buffer.bary.y, buffer.bary.z,
        buffer.world.x, buffer.world.y, buffer.world.z,
        buffer.texture.x, buffer.texture.y,
        buffer.normal.x, buffer.normal.y, buffer.normal.z,
        buffer.color.x, buffer.color.y, buffer.color.z, buffer.color.w,
        buffer.width, buffer.height
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
    // Vec3f lightDir = light.dir * -1;
    Vec3f lightDir = light.dir - Vec3f(buffWx[i], buffWy[i], buffWz[i]);
    Vec3f n = Vec3f(buffNx[i], buffNy[i], buffNz[i]);

    // Calculate the cosine of the angle between the normal and the light direction
    float dot = n * lightDir;
    
    float cosA = dot / (n.mag() * lightDir.mag());
    if (cosA < 0) cosA = -cosA;

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

__global__ void resetShadowMapKernel(
    float *shadowDepth, int shdwWidth, int shdwHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= shdwWidth * shdwHeight) return;

    shadowDepth[i] = 100000;
}

__global__ void createShadowMapKernel(
    const float *worldX, const float *worldY, const float *worldZ,
    const ULLInt *faceWs, ULLInt numFs,
    float *shadowDepth, int shdwWidth, int shdwHeight,
    int shdwTileNumX, int shdwTileNumY, int shdwTileSizeX, int shdwTileSizeY
) {
    ULLInt tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    ULLInt fIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (tIdx >= shdwTileNumX * shdwTileNumY || fIdx >= numFs) return;

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    ULLInt fw0 = faceWs[idx0];
    ULLInt fw1 = faceWs[idx1];
    ULLInt fw2 = faceWs[idx2];

    float sx0 = (worldX[fw0] / 20 + 1) * shdwWidth / 2;
    float sx1 = (worldX[fw1] / 20 + 1) * shdwWidth / 2;
    float sx2 = (worldX[fw2] / 20 + 1) * shdwWidth / 2;

    float sy0 = (worldY[fw0] / 20 + 1) * shdwHeight / 2;
    float sy1 = (worldY[fw1] / 20 + 1) * shdwHeight / 2;
    float sy2 = (worldY[fw2] / 20 + 1) * shdwHeight / 2;

    float sz0 = worldZ[fw0];
    float sz1 = worldZ[fw1];
    float sz2 = worldZ[fw2];

    int tX = tIdx % shdwTileNumX;
    int tY = tIdx / shdwTileNumX;

    int shdwMinX = tX * shdwTileSizeX;
    int shdwMaxX = shdwMinX + shdwTileSizeX;
    int shdwMinY = tY * shdwTileSizeY;
    int shdwMaxY = shdwMinY + shdwTileSizeY;

    // Bound the shadow map
    int minX = min(min(sx0, sx1), sx2);
    int maxX = max(max(sx0, sx1), sx2);
    int minY = min(fmin(sy0, sy1), sy2);
    int maxY = max(max(sy0, sy1), sy2);

    // If bounding box is outside the tile, return
    if (minX > shdwMaxX || maxX < shdwMinX || minY > shdwMaxY || maxY < shdwMinY) return;

    // Clip based on the tile
    minX = max(minX, shdwMinX);
    maxX = min(maxX, shdwMaxX);
    minY = max(minY, shdwMinY);
    maxY = min(maxY, shdwMaxY);

    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
        int sIdx = x + y * shdwWidth;

        Vec3f bary = Vec3f::bary(
            Vec2f(x, y), Vec2f(sx0, sy0), Vec2f(sx1, sy1), Vec2f(sx2, sy2)
        );
        // Out of bound => Ignore
        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float zDepth = bary.x * sz0 + bary.y * sz1 + bary.z * sz2;

        if (atomicMinFloat(&shadowDepth[sIdx], zDepth)) {
            shadowDepth[sIdx] = zDepth;
        }
    }
}

__global__ void applyShadowMapKernel(
    bool *buffActive,
    float *buffWx, float *buffWy, float *buffWz,
    float *buffNx, float *buffNy, float *buffNz,
    float *buffCr, float *buffCg, float *buffCb, float *buffCa,
    int buffWidth, int buffHeight,

    float *shadowDepth, int shdwWidth, int shdwHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    float sfx = (buffWx[i] / 20 + 1) * shdwWidth / 2;
    float sfy = (buffWy[i] / 20 + 1) * shdwHeight / 2;
    float sfz = buffWz[i];

    // Create slight offset based on the normal direction with the light
    sfx += buffNx[i] * 0.8;
    sfy += buffNy[i] * 0.8;
    sfz += buffNz[i] * 0.8;

    int sx = int(sfx);
    int sy = int(sfy);

    if (sx < 0 || sx >= shdwWidth ||
        sy < 0 || sy >= shdwHeight) return;

    // Get the index of the shadow map
    int sIdx = sx + sy * shdwWidth;

    // If the fragment is closer than the shadow map, ignore
    if (sfz <= shadowDepth[sIdx] + 0.0001) return;

    // Apply the shadow
    buffCr[i] *= 0.15;
    buffCg[i] *= 0.15;
    buffCb[i] *= 0.15;
}

__global__ void customShaderKernel(
    bool *buffActive, ULLInt *buffFaceId, float *buffDepth,
    float *buffBrx, float *buffBry, float *buffBrz, // Bary
    float *buffWx, float *buffWy, float *buffWz, // World
    float *buffTu, float *buffTv, // Texture
    float *buffNx, float *buffNy, float *buffNz, // Normal
    float *buffCr, float *buffCg, float *buffCb, float *buffCa, // Color
    int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    ULLInt fIdx = buffFaceId[i];

    // If fId even, red, else blue
    float red = fIdx % 2 == 0 ? 255 : 100;
    float blue = fIdx % 2 == 0 ? 100 : 255;

    buffCr[i] = red;
    buffCg[i] = 100;
    buffCb[i] = blue;
    buffCa[i] = 255;
}