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

        grphic.rtFaces.sx, grphic.rtFaces.sy, grphic.rtFaces.sz, grphic.rtFaces.sw,
        grphic.rtFaces.wx, grphic.rtFaces.wy, grphic.rtFaces.wz,
        grphic.rtFaces.tu, grphic.rtFaces.tv,
        grphic.rtFaces.nx, grphic.rtFaces.ny, grphic.rtFaces.nz,
        grphic.rtFaces.cr, grphic.rtFaces.cg, grphic.rtFaces.cb, grphic.rtFaces.ca,
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
            grphic.rtFaces.sx + i * grphic.chunkSize,
            grphic.rtFaces.sy + i * grphic.chunkSize,
            grphic.rtFaces.sz + i * grphic.chunkSize,
            grphic.rtFaces.sw + i * grphic.chunkSize,
            currentChunkSize,

            buffer.active, buffer.depth, buffer.faceID,
            buffer.bary.x, buffer.bary.y, buffer.bary.z,
            buffer.width, buffer.height,
            grphic.tileNumX, grphic.tileNumY,
            grphic.tileSizeX, grphic.tileSizeY
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
        grphic.rtFaces.sw,
        grphic.rtFaces.wx, grphic.rtFaces.wy, grphic.rtFaces.wz,
        grphic.rtFaces.tu, grphic.rtFaces.tv,
        grphic.rtFaces.nx, grphic.rtFaces.ny, grphic.rtFaces.nz,
        grphic.rtFaces.cr, grphic.rtFaces.cg, grphic.rtFaces.cb, grphic.rtFaces.ca,
        buffer.active, buffer.faceID,
        buffer.bary.x, buffer.bary.y, buffer.bary.z,
        buffer.world.x, buffer.world.y, buffer.world.z,
        buffer.texture.x, buffer.texture.y,
        buffer.normal.x, buffer.normal.y, buffer.normal.z,
        buffer.color.x, buffer.color.y, buffer.color.z, buffer.color.w,
        buffer.width, buffer.height
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

    screenX[i] = t4.x;
    screenY[i] = t4.y;
    screenZ[i] = t4.z;
    screenW[i] = t4.w;
}

// A bunch of placeholder functions
struct Intersect {
    bool near = false, far = false;
    bool left = false, right = false;
    bool up = false, down = false;
};

__device__ Intersect intersect(Vec4f s1, Vec4f s2) {
    Intersect inter;

    if ((s1.z < -s1.w && s2.z > -s2.w) || (s1.z > -s1.w && s2.z < -s2.w)) inter.near = true;
    if ((s1.z < s1.w && s2.z > s2.w) || (s1.z > s1.w && s2.z < s2.w)) inter.far = true;

    if ((s1.x < -s1.w && s2.x > -s2.w) || (s1.x > -s1.w && s2.x < -s2.w)) inter.left = true;
    if ((s1.x < s1.w && s2.x > s2.w) || (s1.x > s1.w && s2.x < s2.w)) inter.right = true;

    if ((s1.y < -s1.w && s2.y > -s2.w) || (s1.y > -s1.w && s2.y < -s2.w)) inter.down = true;
    if ((s1.y < s1.w && s2.y > s2.w) || (s1.y > s1.w && s2.y < s2.w)) inter.up = true;

    return inter;
}

// Create runtime faces
__global__ void createRuntimeFacesKernel(
    const float *screenX, const float *screenY, const float *screenZ, const float *screenW,
    const float *worldX, const float *worldY, const float *worldZ,
    const float *normalX, const float *normalY, const float *normalZ,
    const float *textureX, const float *textureY,
    const float *colorX, const float *colorY, const float *colorZ, float *colorW,
    const ULLInt *faceWs, const ULLInt *faceTs, const ULLInt *faceNs, ULLInt numFs,

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

    ULLInt fw[3] = {faceWs[fIdx0], faceWs[fIdx1], faceWs[fIdx2]};
    ULLInt ft[3] = {faceTs[fIdx0], faceTs[fIdx1], faceTs[fIdx2]};
    ULLInt fn[3] = {faceNs[fIdx0], faceNs[fIdx1], faceNs[fIdx2]};

    // If all W are negative, the face is behind the camera => Ignore
    if (screenW[fw[0]] < 0 && screenW[fw[1]] < 0 && screenW[fw[2]] < 0) return;

    // Find plane diretion
    bool left[3] = {screenX[fw[0]] < -screenW[fw[0]], screenX[fw[1]] < -screenW[fw[1]], screenX[fw[2]] < -screenW[fw[2]]};
    bool right[3] = {screenX[fw[0]] > screenW[fw[0]], screenX[fw[1]] > screenW[fw[1]], screenX[fw[2]] > screenW[fw[2]]};
    bool up[3] = {screenY[fw[0]] > screenW[fw[0]], screenY[fw[1]] > screenW[fw[1]], screenY[fw[2]] > screenW[fw[2]]};
    bool down[3] = {screenY[fw[0]] < -screenW[fw[0]], screenY[fw[1]] < -screenW[fw[1]], screenY[fw[2]] < -screenW[fw[2]]};
    bool far[3] = {screenZ[fw[0]] > screenW[fw[0]], screenZ[fw[1]] > screenW[fw[1]], screenZ[fw[2]] > screenW[fw[2]]};
    bool near[3] = {screenZ[fw[0]] < -screenW[fw[0]], screenZ[fw[1]] < -screenW[fw[1]], -screenZ[fw[2]] < screenW[fw[2]]};

    // All vertices lie on one side of the frustum's planes
    bool allLeft = left[0] && left[1] && left[2];
    bool allRight = right[0] && right[1] && right[2];
    bool allUp = up[0] && up[1] && up[2];
    bool allDown = down[0] && down[1] && down[2];
    bool allFar = far[0] && far[1] && far[2];
    bool allNear = near[0] && near[1] && near[2];
    if (allLeft || allRight || allUp || allDown || allFar || allNear) return;

    ULLInt idx0 = atomicAdd(faceCounter, 1) * 3;
    ULLInt idx1 = idx0 + 1;
    ULLInt idx2 = idx0 + 2;

    runtimeSx[idx0] = screenX[fw[0]]; runtimeSx[idx1] = screenX[fw[1]]; runtimeSx[idx2] = screenX[fw[2]];
    runtimeSy[idx0] = screenY[fw[0]]; runtimeSy[idx1] = screenY[fw[1]]; runtimeSy[idx2] = screenY[fw[2]];
    runtimeSz[idx0] = screenZ[fw[0]]; runtimeSz[idx1] = screenZ[fw[1]]; runtimeSz[idx2] = screenZ[fw[2]];
    runtimeSw[idx0] = screenW[fw[0]]; runtimeSw[idx1] = screenW[fw[1]]; runtimeSw[idx2] = screenW[fw[2]];

    runtimeWx[idx0] = worldX[fw[0]]; runtimeWx[idx1] = worldX[fw[1]]; runtimeWx[idx2] = worldX[fw[2]];
    runtimeWy[idx0] = worldY[fw[0]]; runtimeWy[idx1] = worldY[fw[1]]; runtimeWy[idx2] = worldY[fw[2]];
    runtimeWz[idx0] = worldZ[fw[0]]; runtimeWz[idx1] = worldZ[fw[1]]; runtimeWz[idx2] = worldZ[fw[2]];

    runtimeTu[idx0] = textureX[ft[0]]; runtimeTu[idx1] = textureX[ft[1]]; runtimeTu[idx2] = textureX[ft[2]];
    runtimeTv[idx0] = textureY[ft[0]]; runtimeTv[idx1] = textureY[ft[1]]; runtimeTv[idx2] = textureY[ft[2]];

    runtimeNx[idx0] = normalX[fn[0]]; runtimeNx[idx1] = normalX[fn[1]]; runtimeNx[idx2] = normalX[fn[2]];
    runtimeNy[idx0] = normalY[fn[0]]; runtimeNy[idx1] = normalY[fn[1]]; runtimeNy[idx2] = normalY[fn[2]];
    runtimeNz[idx0] = normalZ[fn[0]]; runtimeNz[idx1] = normalZ[fn[1]]; runtimeNz[idx2] = normalZ[fn[2]];

    runtimeCr[idx0] = colorX[fw[0]]; runtimeCr[idx1] = colorX[fw[1]]; runtimeCr[idx2] = colorX[fw[2]];
    runtimeCg[idx0] = colorY[fw[0]]; runtimeCg[idx1] = colorY[fw[1]]; runtimeCg[idx2] = colorY[fw[2]];
    runtimeCb[idx0] = colorZ[fw[0]]; runtimeCb[idx1] = colorZ[fw[1]]; runtimeCb[idx2] = colorZ[fw[2]];
    runtimeCa[idx0] = colorW[fw[0]]; runtimeCa[idx1] = colorW[fw[1]]; runtimeCa[idx2] = colorW[fw[2]];
}

// Depth map creation
__global__ void createDepthMapKernel(
    const float *runtimeSx, const float *runtimeSy, const float *runtimeSz, const float *runtimeSw, ULLInt faceCounter,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId,
    float *buffBaryX, float *buffBaryY, float *buffBaryZ,
    int buffWidth, int buffHeight, int tileNumX, int tileNumY, int tileSizeX, int tileSizeY
) {
    ULLInt tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    ULLInt fIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdx >= tileNumX * tileNumY || fIdx >= faceCounter) return;

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    float sx0 = (runtimeSx[idx0] / runtimeSw[idx0] + 1) * buffWidth / 2;
    float sx1 = (runtimeSx[idx1] / runtimeSw[idx1] + 1) * buffWidth / 2;
    float sx2 = (runtimeSx[idx2] / runtimeSw[idx2] + 1) * buffWidth / 2;

    float sy0 = (1 - runtimeSy[idx0] / runtimeSw[idx0]) * buffHeight / 2;
    float sy1 = (1 - runtimeSy[idx1] / runtimeSw[idx1]) * buffHeight / 2;
    float sy2 = (1 - runtimeSy[idx2] / runtimeSw[idx2]) * buffHeight / 2;

    float sz0 = runtimeSz[idx0] / runtimeSw[idx0];
    float sz1 = runtimeSz[idx1] / runtimeSw[idx1];
    float sz2 = runtimeSz[idx2] / runtimeSw[idx2];

    // Buffer bounding box based on the tile

    int tX = tIdx % tileNumX;
    int tY = tIdx / tileNumX;

    int bufferMinX = tX * tileSizeX;
    int bufferMaxX = bufferMinX + tileSizeX;
    int bufferMinY = tY * tileSizeY;
    int bufferMaxY = bufferMinY + tileSizeY;

    // Bounding box
    int minX = min(min(sx0, sx1), sx2);
    int maxX = max(max(sx0, sx1), sx2);
    int minY = min(min(sy0, sy1), sy2);
    int maxY = max(max(sy0, sy1), sy2);

    // If bounding box not in tile area, return
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
            Vec2f(x, y), Vec2f(sx0, sy0), Vec2f(sx1, sy1), Vec2f(sx2, sy2)
        );
        // Out of bound => Ignore
        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float zDepth = bary.x * sz0 + bary.y * sz1 + bary.z * sz2;

        if (atomicMinFloat(&buffDepth[bIdx], zDepth)) {
            buffDepth[bIdx] = zDepth;
            buffActive[bIdx] = true;
            buffFaceId[bIdx] = fIdx;

            buffBaryX[bIdx] = bary.x;
            buffBaryY[bIdx] = bary.y;
            buffBaryZ[bIdx] = bary.z;
        }
    }
}

__global__ void rasterizationKernel(
    const float *runtimeSw,
    const float *runtimeWx, const float *runtimeWy, const float *runtimeWz,
    const float *runtimeTu, const float *runtimeTv,
    const float *runtimeNx, const float *runtimeNy, const float *runtimeNz,
    const float *runtimeCr, const float *runtimeCg, const float *runtimeCb, const float *runtimeCa,

    const bool *buffActive, const ULLInt *buffFaceId,
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

    // Set vertex, texture, and normal (with correct perspective correction)

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    // Get barycentric coordinates
    float alp = buffBrx[i];
    float bet = buffBry[i];
    float gam = buffBrz[i];

    // Get homogenous 1/w
    float homo1divW = alp / runtimeSw[idx0] + bet / runtimeSw[idx1] + gam / runtimeSw[idx2];

    // Set world position
    float wx_sw = runtimeWx[idx0] / runtimeSw[idx0] * alp + runtimeWx[idx1] / runtimeSw[idx1] * bet + runtimeWx[idx2] / runtimeSw[idx2] * gam;
    float wy_sw = runtimeWy[idx0] / runtimeSw[idx0] * alp + runtimeWy[idx1] / runtimeSw[idx1] * bet + runtimeWy[idx2] / runtimeSw[idx2] * gam;
    float wz_sw = runtimeWz[idx0] / runtimeSw[idx0] * alp + runtimeWz[idx1] / runtimeSw[idx1] * bet + runtimeWz[idx2] / runtimeSw[idx2] * gam;

    buffWx[i] = wx_sw / homo1divW;
    buffWy[i] = wy_sw / homo1divW;
    buffWz[i] = wz_sw / homo1divW;

    // Set texture
    float tu_sw = runtimeTu[idx0] / runtimeSw[idx0] * alp + runtimeTu[idx1] / runtimeSw[idx1] * bet + runtimeTu[idx2] / runtimeSw[idx2] * gam;
    float tv_sw = runtimeTv[idx0] / runtimeSw[idx0] * alp + runtimeTv[idx1] / runtimeSw[idx1] * bet + runtimeTv[idx2] / runtimeSw[idx2] * gam;

    buffTu[i] = tu_sw / homo1divW;
    buffTv[i] = tv_sw / homo1divW;

    // Set normal
    float nx_sw = runtimeNx[idx0] / runtimeSw[idx0] * alp + runtimeNx[idx1] / runtimeSw[idx1] * bet + runtimeNx[idx2] / runtimeSw[idx2] * gam;
    float ny_sw = runtimeNy[idx0] / runtimeSw[idx0] * alp + runtimeNy[idx1] / runtimeSw[idx1] * bet + runtimeNy[idx2] / runtimeSw[idx2] * gam;
    float nz_sw = runtimeNz[idx0] / runtimeSw[idx0] * alp + runtimeNz[idx1] / runtimeSw[idx1] * bet + runtimeNz[idx2] / runtimeSw[idx2] * gam;

    buffNx[i] = nx_sw / homo1divW;
    buffNy[i] = ny_sw / homo1divW;
    buffNz[i] = nz_sw / homo1divW;
    float mag = sqrt( // Normalize the normal
        buffNx[i] * buffNx[i] +
        buffNy[i] * buffNy[i] +
        buffNz[i] * buffNz[i]
    );
    buffNx[i] /= mag;
    buffNy[i] /= mag;
    buffNz[i] /= mag; 

    // Set color
    float cr_sw = runtimeCr[idx0] / runtimeSw[idx0] * alp + runtimeCr[idx1] / runtimeSw[idx1] * bet + runtimeCr[idx2] / runtimeSw[idx2] * gam;
    float cg_sw = runtimeCg[idx0] / runtimeSw[idx0] * alp + runtimeCg[idx1] / runtimeSw[idx1] * bet + runtimeCg[idx2] / runtimeSw[idx2] * gam;
    float cb_sw = runtimeCb[idx0] / runtimeSw[idx0] * alp + runtimeCb[idx1] / runtimeSw[idx1] * bet + runtimeCb[idx2] / runtimeSw[idx2] * gam;
    float ca_sw = runtimeCa[idx0] / runtimeSw[idx0] * alp + runtimeCa[idx1] / runtimeSw[idx1] * bet + runtimeCa[idx2] / runtimeSw[idx2] * gam;

    buffCr[i] = cr_sw / homo1divW;
    buffCg[i] = cg_sw / homo1divW;
    buffCb[i] = cb_sw / homo1divW;
    buffCa[i] = ca_sw / homo1divW;
}