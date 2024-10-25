#include <VertexShader.cuh>

// Render pipeline

void VertexShader::cameraProjection() {
    Graphic3D &grphic = Graphic3D::instance();
    Camera3D &camera = grphic.camera;
    Mesh3D &mesh = grphic.mesh;

    size_t gridSize = (mesh.world.size + 256 - 1) / 256;
    cameraProjectionKernel<<<gridSize, 256>>>(
        mesh.screen.x, mesh.screen.y, mesh.screen.z, mesh.screen.w,
        mesh.world.x, mesh.world.y, mesh.world.z,
        camera.mvp, mesh.world.size
    );
    cudaDeviceSynchronize();
}

void VertexShader::createRuntimeFaces() {
    Graphic3D &grphic = Graphic3D::instance();
    Mesh3D &mesh = grphic.mesh;

    cudaMemset(grphic.d_faceCounter, 0, sizeof(ULLInt));

    size_t gridSize = (mesh.faces.size / 3 + 256 - 1) / 256;

    createRuntimeFacesKernel<<<gridSize, 256>>>(
        mesh.screen.x, mesh.screen.y, mesh.screen.z, mesh.screen.w,
        mesh.world.x, mesh.world.y, mesh.world.z,
        mesh.normal.x, mesh.normal.y, mesh.normal.z,
        mesh.texture.x, mesh.texture.y,
        mesh.color.x, mesh.color.y, mesh.color.z, mesh.color.w,
        mesh.faces.v, mesh.faces.t, mesh.faces.n, mesh.faces.size / 3,

        grphic.runtimeFaces.sx, grphic.runtimeFaces.sy, grphic.runtimeFaces.sz, grphic.runtimeFaces.sw,
        grphic.runtimeFaces.wx, grphic.runtimeFaces.wy, grphic.runtimeFaces.wz,
        grphic.runtimeFaces.tu, grphic.runtimeFaces.tv,
        grphic.runtimeFaces.nx, grphic.runtimeFaces.ny, grphic.runtimeFaces.nz,
        grphic.runtimeFaces.cr, grphic.runtimeFaces.cg, grphic.runtimeFaces.cb, grphic.runtimeFaces.ca,
        grphic.d_faceCounter
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&grphic.faceCounter, grphic.d_faceCounter, sizeof(ULLInt), cudaMemcpyDeviceToHost);
}

void VertexShader::createDepthMapBeta() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    // 1 million elements per batch
    int chunkNum = (grphic.faceCounter + grphic.chunkSize - 1) / grphic.chunkSize;

    dim3 blockSize(16, 16);

    for (int i = 0; i < chunkNum; i++) {
        size_t currentChunkSize = (i == chunkNum - 1) ?
            grphic.faceCounter - i * grphic.chunkSize : grphic.chunkSize;
        size_t blockNumTile = (grphic.tileNum + blockSize.x - 1) / blockSize.x;
        size_t blockNumFace = (currentChunkSize + blockSize.y - 1) / blockSize.y;
        dim3 blockNum(blockNumTile, blockNumFace);

        createDepthMapKernel<<<blockNum, blockSize, 0, grphic.faceStreams[i]>>>(
            grphic.runtimeFaces.sx + i * grphic.chunkSize,
            grphic.runtimeFaces.sy + i * grphic.chunkSize,
            grphic.runtimeFaces.sz + i * grphic.chunkSize,
            grphic.runtimeFaces.sw + i * grphic.chunkSize,
            currentChunkSize,

            buffer.active, buffer.depth, buffer.faceID, buffer.bary,
            buffer.width, buffer.height,
            grphic.tileNumX, grphic.tileNumY,
            grphic.tileWidth, grphic.tileHeight
        );
    }

    // Synchronize all streams
    for (int i = 0; i < chunkNum; i++) {
        cudaStreamSynchronize(grphic.faceStreams[i]);
    }
}

void VertexShader::rasterization() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    rasterizationKernel<<<buffer.blockNum, buffer.blockSize>>>(
        grphic.runtimeFaces.wx, grphic.runtimeFaces.wy, grphic.runtimeFaces.wz,
        grphic.runtimeFaces.tu, grphic.runtimeFaces.tv,
        grphic.runtimeFaces.nx, grphic.runtimeFaces.ny, grphic.runtimeFaces.nz,
        grphic.runtimeFaces.cr, grphic.runtimeFaces.cg, grphic.runtimeFaces.cb, grphic.runtimeFaces.ca,
        buffer.faceID,
        buffer.world, buffer.texture, buffer.normal, buffer.color,
        buffer.active, buffer.bary, buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

// Camera projection (MVP) kernel
__global__ void cameraProjectionKernel(
    float *screenX, float *screenY, float *screenZ, float *screenW,
    float *worldX, float *worldY, float *worldZ,
    Mat4f mvp, ULLInt numWs
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numWs) return;

    Vec4f v4(worldX[i], worldY[i], worldZ[i], 1);
    Vec4f t4 = mvp * v4;
    bool inside = (
        t4.x >= -t4.w && t4.x <= t4.w &&
        t4.y >= -t4.w && t4.y <= t4.w &&
        t4.z >= 0 && t4.z <= t4.w
    );

    Vec3f t3 = t4.toVec3f(); // Convert to NDC [-1, 1]

    screenX[i] = t3.x;
    screenY[i] = t3.y;
    screenZ[i] = t3.z;
    screenW[i] = inside ? 1 : 0;
}

// Create runtime faces
__global__ void createRuntimeFacesKernel(
    float *screenX, float *screenY, float *screenZ, float *screenW,
    float *worldX, float *worldY, float *worldZ,
    float *normalX, float *normalY, float *normalZ,
    float *textureX, float *textureY,
    float *colorX, float *colorY, float *colorZ, float *colorW,
    ULLInt *faceWs, ULLInt *faceTs, ULLInt *faceNs, ULLInt numFs,

    float *runtimeSx, float *runtimeSy, float *runtimeSz, float *runtimeSw,
    float *runtimeWx, float *runtimeWy, float *runtimeWz,
    float *runtimeTu, float *runtimeTv,
    float *runtimeNx, float *runtimeNy, float *runtimeNz,
    float *runtimeCr, float *runtimeCg, float *runtimeCb, float *runtimeCa,
    ULLInt *faceCounter
) {
    ULLInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= numFs) return;

    ULLInt fIdx0 = fIdx * 3;
    ULLInt fIdx1 = fIdx * 3 + 1;
    ULLInt fIdx2 = fIdx * 3 + 2;

    ULLInt fw0 = faceWs[fIdx0];
    ULLInt fw1 = faceWs[fIdx1];
    ULLInt fw2 = faceWs[fIdx2];

    ULLInt ft0 = faceTs[fIdx0];
    ULLInt ft1 = faceTs[fIdx1];
    ULLInt ft2 = faceTs[fIdx2];

    ULLInt fn0 = faceNs[fIdx0];
    ULLInt fn1 = faceNs[fIdx1];
    ULLInt fn2 = faceNs[fIdx2];

    float sw0 = screenW[fw0];
    float sw1 = screenW[fw1];
    float sw2 = screenW[fw2];

    if (sw0 <= 0 && sw1 <= 0 && sw2 <= 0) return;

    ULLInt idx0 = atomicAdd(faceCounter, 1) * 3;
    ULLInt idx1 = idx0 + 1;
    ULLInt idx2 = idx0 + 2;

    runtimeSx[idx0] = screenX[fw0]; runtimeSx[idx1] = screenX[fw1]; runtimeSx[idx2] = screenX[fw2];
    runtimeSy[idx0] = screenY[fw0]; runtimeSy[idx1] = screenY[fw1]; runtimeSy[idx2] = screenY[fw2];
    runtimeSz[idx0] = screenZ[fw0]; runtimeSz[idx1] = screenZ[fw1]; runtimeSz[idx2] = screenZ[fw2];
    runtimeSw[idx0] = sw0; runtimeSw[idx1] = sw1; runtimeSw[idx2] = sw2;

    runtimeWx[idx0] = worldX[fw0]; runtimeWx[idx1] = worldX[fw1]; runtimeWx[idx2] = worldX[fw2];
    runtimeWy[idx0] = worldY[fw0]; runtimeWy[idx1] = worldY[fw1]; runtimeWy[idx2] = worldY[fw2];
    runtimeWz[idx0] = worldZ[fw0]; runtimeWz[idx1] = worldZ[fw1]; runtimeWz[idx2] = worldZ[fw2];

    runtimeTu[idx0] = textureX[ft0]; runtimeTu[idx1] = textureX[ft1]; runtimeTu[idx2] = textureX[ft2];
    runtimeTv[idx0] = textureY[ft0]; runtimeTv[idx1] = textureY[ft1]; runtimeTv[idx2] = textureY[ft2];

    runtimeNx[idx0] = normalX[fn0]; runtimeNx[idx1] = normalX[fn1]; runtimeNx[idx2] = normalX[fn2];
    runtimeNy[idx0] = normalY[fn0]; runtimeNy[idx1] = normalY[fn1]; runtimeNy[idx2] = normalY[fn2];
    runtimeNz[idx0] = normalZ[fn0]; runtimeNz[idx1] = normalZ[fn1]; runtimeNz[idx2] = normalZ[fn2];

    runtimeCr[idx0] = colorX[fw0]; runtimeCr[idx1] = colorX[fw1]; runtimeCr[idx2] = colorX[fw2];
    runtimeCg[idx0] = colorY[fw0]; runtimeCg[idx1] = colorY[fw1]; runtimeCg[idx2] = colorY[fw2];
    runtimeCb[idx0] = colorZ[fw0]; runtimeCb[idx1] = colorZ[fw1]; runtimeCb[idx2] = colorZ[fw2];
    runtimeCa[idx0] = colorW[fw0]; runtimeCa[idx1] = colorW[fw1]; runtimeCa[idx2] = colorW[fw2];
}

// Depth map creation
__global__ void createDepthMapKernel(
    float *runtimeSx, float *runtimeSy, float *runtimeSz, float *runtimeSw, ULLInt faceCounter,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight,
    int tileNumX, int tileNumY, int tileWidth, int tileHeight
) {
    ULLInt tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    ULLInt fIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdx >= tileNumX * tileNumY || fIdx >= faceCounter) return;

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    Vec4f p0 = Vec4f(runtimeSx[idx0], runtimeSy[idx0], runtimeSz[idx0], runtimeSw[idx0]);
    Vec4f p1 = Vec4f(runtimeSx[idx1], runtimeSy[idx1], runtimeSz[idx1], runtimeSw[idx1]);
    Vec4f p2 = Vec4f(runtimeSx[idx2], runtimeSy[idx2], runtimeSz[idx2], runtimeSw[idx2]);

    // Entirely outside the frustum
    if (p0.w != 1 && p1.w != 1 && p2.w != 1) return;

    p0.x = (p0.x + 1) * buffWidth / 2;
    p0.y = (1 - p0.y) * buffHeight / 2;
    p1.x = (p1.x + 1) * buffWidth / 2;
    p1.y = (1 - p1.y) * buffHeight / 2;
    p2.x = (p2.x + 1) * buffWidth / 2;
    p2.y = (1 - p2.y) * buffHeight / 2;

    // Buffer bounding box based on the tile

    int tX = tIdx % tileNumX;
    int tY = tIdx / tileNumX;

    int bufferMinX = tX * tileWidth;
    int bufferMaxX = bufferMinX + tileWidth;
    int bufferMinY = tY * tileHeight;
    int bufferMaxY = bufferMinY + tileHeight;

    // Bounding box
    int minX = min(min(p0.x, p1.x), p2.x);
    int maxX = max(max(p0.x, p1.x), p2.x);
    int minY = min(min(p0.y, p1.y), p2.y);
    int maxY = max(max(p0.y, p1.y), p2.y);

    // // If bounding box is outside the tile area, return
    if (minX > bufferMaxX ||
        maxX < bufferMinX ||
        minY > bufferMaxY ||
        maxY < bufferMinY) return;

    // Clip the bounding box based on the buffer
    minX = max(minX, bufferMinX);
    maxX = min(maxX, bufferMaxX);
    minY = max(minY, bufferMinY);
    maxY = min(maxY, bufferMaxY);

    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
        int bIdx = x + y * buffWidth;

        Vec3f bary = Vec3f::bary(
            Vec2f(x, y),
            Vec2f(p0.x, p0.y),
            Vec2f(p1.x, p1.y),
            Vec2f(p2.x, p2.y)
        );

        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float zDepth = bary.x * p0.z + bary.y * p1.z + bary.z * p2.z;

        if (atomicMinFloat(&buffDepth[bIdx], zDepth)) {
            buffDepth[bIdx] = zDepth;
            buffActive[bIdx] = true;
            buffFaceId[bIdx] = fIdx;
            buffBary[bIdx] = bary;
        }
    }
}

__global__ void rasterizationKernel(
    float *runtimeWx, float *runtimeWy, float *runtimeWz,
    float *runtimeTu, float *runtimeTv,
    float *runtimeNx, float *runtimeNy, float *runtimeNz,
    float *runtimeCr, float *runtimeCg, float *runtimeCb, float *runtimeCa,
    ULLInt *buffFaceId,
    Vec3f *buffWorld, Vec2f *buffTexture, Vec3f *buffNormal, Vec4f *buffColor,
    bool *buffActive, Vec3f *buffBary, int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    ULLInt fIdx = buffFaceId[i];

    // Set vertex, texture, and normal indices

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    // Get barycentric coordinates
    float alp = buffBary[i].x;
    float bet = buffBary[i].y;
    float gam = buffBary[i].z;

    // Set world position
    Vec3f w0 = Vec3f(runtimeWx[idx0], runtimeWy[idx0], runtimeWz[idx0]);
    Vec3f w1 = Vec3f(runtimeWx[idx1], runtimeWy[idx1], runtimeWz[idx1]);
    Vec3f w2 = Vec3f(runtimeWx[idx2], runtimeWy[idx2], runtimeWz[idx2]);
    buffWorld[i] = w0 * alp + w1 * bet + w2 * gam;

    // Set texture
    Vec2f t0 = Vec2f(runtimeTu[idx0], runtimeTv[idx0]);
    Vec2f t1 = Vec2f(runtimeTu[idx1], runtimeTv[idx1]);
    Vec2f t2 = Vec2f(runtimeTu[idx2], runtimeTv[idx2]);
    buffTexture[i] = t0 * alp + t1 * bet + t2 * gam;

    // Set normal
    Vec3f n0 = Vec3f(runtimeNx[idx0], runtimeNy[idx0], runtimeNz[idx0]);
    Vec3f n1 = Vec3f(runtimeNx[idx1], runtimeNy[idx1], runtimeNz[idx1]);
    Vec3f n2 = Vec3f(runtimeNx[idx2], runtimeNy[idx2], runtimeNz[idx2]);
    n0.norm(); n1.norm(); n2.norm();
    buffNormal[i] = n0 * alp + n1 * bet + n2 * gam;
    buffNormal[i].norm();

    // Set color
    Vec4f c0 = Vec4f(runtimeCr[idx0], runtimeCg[idx0], runtimeCb[idx0], runtimeCa[idx0]);
    Vec4f c1 = Vec4f(runtimeCr[idx1], runtimeCg[idx1], runtimeCb[idx1], runtimeCa[idx1]);
    Vec4f c2 = Vec4f(runtimeCr[idx2], runtimeCg[idx2], runtimeCb[idx2], runtimeCa[idx2]);
    buffColor[i] = c0 * alp + c1 * bet + c2 * gam;
}