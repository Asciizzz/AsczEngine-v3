#include <FragmentShader.cuh>

// ======================== Static functions ========================

void FragmentShader::applyMaterial() { // Beta
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buff = grphic.buffer;
    Mesh3D &mesh = grphic.mesh;

    applyMaterialKernel<<<buff.blockNum, buff.blockSize>>>(
        // Mesh material
        mesh.m.ka.x, mesh.m.ka.y, mesh.m.ka.z,
        mesh.m.kd.x, mesh.m.kd.y, mesh.m.kd.z,
        mesh.m.ks.x, mesh.m.ks.y, mesh.m.ks.z,
        mesh.m.mkd.x,
        // Mesh texture
        mesh.t.tx.x, mesh.t.tx.y, mesh.t.tx.z,
        mesh.t.wh.x, mesh.t.wh.y, mesh.t.of.x,
        // Buffer
        buff.active, buff.matID,
        buff.color.x, buff.color.y, buff.color.z, buff.color.w,
        buff.texture.x, buff.texture.y,
        buff.width, buff.height
    );
    cudaDeviceSynchronize();
}

void FragmentShader::phongShading() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buff = grphic.buffer;

    phongShadingKernel<<<buff.blockNum, buff.blockSize>>>(
        buff.active,
        buff.world.x, buff.world.y, buff.world.z,
        buff.texture.x, buff.texture.y,
        buff.normal.x, buff.normal.y, buff.normal.z,
        buff.color.x, buff.color.y, buff.color.z, buff.color.w,
        buff.width, buff.height,

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
    size_t blockNumFace = (mesh.f.count + blockSize.y - 1) / blockSize.y;
    dim3 blockNum(blockNumTile, blockNumFace);

    createShadowMapKernel<<<blockNum, blockSize>>>(
        mesh.v.w.x, mesh.v.w.y, mesh.v.w.z,
        mesh.f.v, mesh.f.count,
        grphic.shadowDepth, grphic.shdwWidth, grphic.shdwHeight,
        grphic.shdwTileNumX, grphic.shdwTileNumY, grphic.shdwTileSizeX, grphic.shdwTileSizeY
    );
    cudaDeviceSynchronize();
}

void FragmentShader::applyShadowMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buff = grphic.buffer;

    applyShadowMapKernel<<<buff.blockNum, buff.blockSize>>>(
        buff.active,
        buff.world.x, buff.world.y, buff.world.z,
        buff.normal.x, buff.normal.y, buff.normal.z,
        buff.color.x, buff.color.y, buff.color.z, buff.color.w,
        buff.width, buff.height,

        grphic.shadowDepth, grphic.shdwWidth, grphic.shdwHeight
    );
    cudaDeviceSynchronize();
}

void FragmentShader::customShader() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buff = grphic.buffer;

    customShaderKernel<<<buff.blockNum, buff.blockSize>>>(
        buff.active, buff.faceID, buff.depth,
        buff.bary.x, buff.bary.y, buff.bary.z,
        buff.world.x, buff.world.y, buff.world.z,
        buff.texture.x, buff.texture.y,
        buff.normal.x, buff.normal.y, buff.normal.z,
        buff.color.x, buff.color.y, buff.color.z, buff.color.w,
        buff.width, buff.height
    );
    cudaDeviceSynchronize();
}

// ======================== Kernels ========================

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
    bool *bActive, LLInt *bMat,
    float *bCr, float *bCg, float *bCb, float *bCa,
    float *bTu, float *bTv,
    int bWidth, int bHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bWidth * bHeight || !bActive[i]) return;

    LLInt matIdx = bMat[i];
    if (matIdx < 0) {
        // Set default
        bCr[i] = 200;
        bCg[i] = 200;
        bCb[i] = 200;
        bCa[i] = 255;
        return;
    }

    bCr[i] = kdr[matIdx] * 255;
    bCg[i] = kdg[matIdx] * 255;
    bCb[i] = kdb[matIdx] * 255;
    bCa[i] = 255;

    LLInt texIdx = mkd[matIdx];
    if (texIdx >= 0) {
        // Warp the texture modulo 1
        float warpx = bTu[i] - floorf(bTu[i]);
        float warpy = bTv[i] - floorf(bTv[i]);

        int x = warpx * txw[texIdx];
        int y = warpy * txh[texIdx];

        int tIdx = x + y * txw[texIdx];
        tIdx += txof[texIdx];

        bCr[i] = txr[tIdx];
        bCg[i] = txg[tIdx];
        bCb[i] = txb[tIdx];
    }
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

    if (buffNx[i] == 0 &&
        buffNy[i] == 0 &&
        buffNz[i] == 0) return;

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

    // Find remainder of fIdx / 12
    /* The 12 color rotation:

    Red
    Green
    Blue
    Yellow
    Cyan
    Magenta
    Orange
    Purple
    Pink
    Brown
    White
    Gray (black is literally invisible)

    */

    int colorIdx = fIdx % 12;
    Vec4f color;
    if (colorIdx == 0) color = Vec4f(255, 80, 80, 255);
    else if (colorIdx == 1) color = Vec4f(80, 255, 80, 255);
    else if (colorIdx == 2) color = Vec4f(80, 80, 255, 255);
    else if (colorIdx == 3) color = Vec4f(255, 255, 80, 255);
    else if (colorIdx == 4) color = Vec4f(80, 255, 255, 255);
    else if (colorIdx == 5) color = Vec4f(255, 80, 255, 255);
    else if (colorIdx == 6) color = Vec4f(255, 160, 80, 255);
    else if (colorIdx == 7) color = Vec4f(160, 80, 255, 255);
    else if (colorIdx == 8) color = Vec4f(255, 80, 160, 255);
    else if (colorIdx == 9) color = Vec4f(160, 80, 80, 255);
    else if (colorIdx == 10) color = Vec4f(255, 255, 255, 255);
    else if (colorIdx == 11) color = Vec4f(160, 160, 160, 255);

    buffCr[i] = color.x;
    buffCg[i] = color.y;
    buffCb[i] = color.z;
    buffCa[i] = color.w;
}