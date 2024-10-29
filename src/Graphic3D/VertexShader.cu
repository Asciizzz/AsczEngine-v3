#include <VertexShader.cuh>

// Render pipeline

void VertexShader::createRuntimeFaces() {
    Graphic3D &grphic = Graphic3D::instance();
    Mesh3D &mesh = grphic.mesh;

    cudaMemset(grphic.d_faceCount, 0, sizeof(ULLInt));

    size_t gridSize = (mesh.faces.size / 3 + 255) / 256;

    createRuntimeFacesKernel<<<gridSize, 256>>>(
        mesh.world.x, mesh.world.y, mesh.world.z,
        mesh.normal.x, mesh.normal.y, mesh.normal.z,
        mesh.texture.x, mesh.texture.y,
        mesh.color.x, mesh.color.y, mesh.color.z, mesh.color.w,
        mesh.faces.v, mesh.faces.t, mesh.faces.n, mesh.faces.size / 3,

        grphic.rtFaces.wx, grphic.rtFaces.wy, grphic.rtFaces.wz,
        grphic.rtFaces.tu, grphic.rtFaces.tv,
        grphic.rtFaces.nx, grphic.rtFaces.ny, grphic.rtFaces.nz,
        grphic.rtFaces.cr, grphic.rtFaces.cg, grphic.rtFaces.cb, grphic.rtFaces.ca,
        grphic.d_faceCount
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&grphic.faceCount, grphic.d_faceCount, sizeof(ULLInt), cudaMemcpyDeviceToHost);
}

void VertexShader::frustumClipping() {
    Graphic3D &grphic = Graphic3D::instance();
    Camera3D &camera = grphic.camera;

    cudaMemset(grphic.d_clip1Count, 0, sizeof(ULLInt));
    cudaMemset(grphic.d_clip2Count, 0, sizeof(ULLInt));

    size_t gridSize;

    // Clip near plane
    gridSize = (grphic.faceCount + 255) / 256;
    clipFrustumKernel<<<gridSize, 256>>>(
        grphic.rtFaces.wx, grphic.rtFaces.wy, grphic.rtFaces.wz,
        grphic.rtFaces.tu, grphic.rtFaces.tv,
        grphic.rtFaces.nx, grphic.rtFaces.ny, grphic.rtFaces.nz,
        grphic.rtFaces.cr, grphic.rtFaces.cg, grphic.rtFaces.cb, grphic.rtFaces.ca,
        grphic.d_faceCount,

        grphic.clip1.wx, grphic.clip1.wy, grphic.clip1.wz,
        grphic.clip1.tu, grphic.clip1.tv,
        grphic.clip1.nx, grphic.clip1.ny, grphic.clip1.nz,
        grphic.clip1.cr, grphic.clip1.cg, grphic.clip1.cb, grphic.clip1.ca,
        grphic.d_clip1Count,

        camera.nearPlane
    );
    cudaDeviceSynchronize();
    cudaMemcpy(&grphic.clip1Count, grphic.d_clip1Count, sizeof(ULLInt), cudaMemcpyDeviceToHost);

    // Clip far plane
    gridSize = (grphic.clip1Count + 255) / 256;
    clipFrustumKernel<<<gridSize, 256>>>(
        grphic.clip1.wx, grphic.clip1.wy, grphic.clip1.wz,
        grphic.clip1.tu, grphic.clip1.tv,
        grphic.clip1.nx, grphic.clip1.ny, grphic.clip1.nz,
        grphic.clip1.cr, grphic.clip1.cg, grphic.clip1.cb, grphic.clip1.ca,
        grphic.d_clip1Count,

        grphic.clip2.wx, grphic.clip2.wy, grphic.clip2.wz,
        grphic.clip2.tu, grphic.clip2.tv,
        grphic.clip2.nx, grphic.clip2.ny, grphic.clip2.nz,
        grphic.clip2.cr, grphic.clip2.cg, grphic.clip2.cb, grphic.clip2.ca,
        grphic.d_clip2Count,

        camera.farPlane
    );
    cudaDeviceSynchronize();
    cudaMemcpy(&grphic.clip2Count, grphic.d_clip2Count, sizeof(ULLInt), cudaMemcpyDeviceToHost);
}

void VertexShader::cameraProjection() {
    Graphic3D &grphic = Graphic3D::instance();
    Camera3D &camera = grphic.camera;

    size_t gridSize = (grphic.clip2Count * 3 + 255) / 256;
    cameraProjectionKernel<<<gridSize, 256>>>(
        grphic.clip2.sx, grphic.clip2.sy, grphic.clip2.sz, grphic.clip2.sw,
        grphic.clip2.wx, grphic.clip2.wy, grphic.clip2.wz,
        camera.mvp, grphic.clip2Count * 3
    );
    cudaDeviceSynchronize();
}

void VertexShader::createDepthMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    // Split the faces into chunks
    size_t chunkNum = (grphic.clip2Count + grphic.faceChunkSize - 1) 
                    /  grphic.faceChunkSize;

    dim3 blockSize(16, 32);
    for (size_t i = 0; i < chunkNum; i++) {
        size_t faceOffset = grphic.faceChunkSize * i;

        size_t curFaceCount = (i == chunkNum - 1) ?
            grphic.clip2Count - faceOffset : grphic.faceChunkSize;
        size_t blockNumTile = (grphic.tileNum + blockSize.x - 1) / blockSize.x;
        size_t blockNumFace = (curFaceCount + blockSize.y - 1) / blockSize.y;
        dim3 blockNum(blockNumTile, blockNumFace);

        createDepthMapKernel<<<blockNum, blockSize>>>(
            grphic.clip2.sx, grphic.clip2.sy,
            grphic.clip2.sz, grphic.clip2.sw,
            curFaceCount, faceOffset,

            buffer.active, buffer.depth, buffer.faceID,
            buffer.bary.x, buffer.bary.y, buffer.bary.z,
            buffer.width, buffer.height,
            grphic.tileNumX, grphic.tileNumY,
            grphic.tileSizeX, grphic.tileSizeY
        );
        cudaDeviceSynchronize();
    }
}

void VertexShader::rasterization() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    rasterizationKernel<<<buffer.blockNum, buffer.blockSize>>>(
        grphic.clip2.sw,
        grphic.clip2.wx, grphic.clip2.wy, grphic.clip2.wz,
        grphic.clip2.tu, grphic.clip2.tv,
        grphic.clip2.nx, grphic.clip2.ny, grphic.clip2.nz,
        grphic.clip2.cr, grphic.clip2.cg, grphic.clip2.cb, grphic.clip2.ca,

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

    screenX[i] = -t4.x;
    screenY[i] = t4.y;
    screenZ[i] = t4.z;
    screenW[i] = t4.w;
}

// Create runtime faces
__global__ void createRuntimeFacesKernel(
    const float *worldX, const float *worldY, const float *worldZ,
    const float *normalX, const float *normalY, const float *normalZ,
    const float *textureX, const float *textureY,
    const float *colorX, const float *colorY, const float *colorZ, float *colorW,
    const ULLInt *faceWs, const ULLInt *faceTs, const ULLInt *faceNs, ULLInt numFs,

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

    ULLInt idx0 = atomicAdd(faceCounter, 1) * 3;
    ULLInt idx1 = idx0 + 1;
    ULLInt idx2 = idx0 + 2;

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

// Frustum culling
__global__ void clipFrustumKernel(
    const float *preWx, const float *preWy, const float *preWz,
    const float *preTu, const float *preTv,
    const float *preNx, const float *preNy, const float *preNz,
    const float *preCr, const float *preCg, const float *preCb, const float *preCa,
    ULLInt *preCounter,

    float *postWx, float *postWy, float *postWz,
    float *postTu, float *postTv,
    float *postNx, float *postNy, float *postNz,
    float *postCr, float *postCg, float *postCb, float *postCa,
    ULLInt *postCounter,

    Plane3D plane
) {
    ULInt preIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (preIdx >= *preCounter) return;

    ULInt preIdx0 = preIdx * 3;
    ULInt preIdx1 = preIdx * 3 + 1;
    ULInt preIdx2 = preIdx * 3 + 2;

    Vec3f rtWs[3] = {
        Vec3f(preWx[preIdx0], preWy[preIdx0], preWz[preIdx0]),
        Vec3f(preWx[preIdx1], preWy[preIdx1], preWz[preIdx1]),
        Vec3f(preWx[preIdx2], preWy[preIdx2], preWz[preIdx2])
    };
    float side[3] = {
        plane.equation(rtWs[0]),
        plane.equation(rtWs[1]),
        plane.equation(rtWs[2])
    };

    // If all behind, return
    if (side[0] < 0 && side[1] < 0 && side[2] < 0) return;

    Vec2f rtTs[3] = {
        Vec2f(preTu[preIdx0], preTv[preIdx0]),
        Vec2f(preTu[preIdx1], preTv[preIdx1]),
        Vec2f(preTu[preIdx2], preTv[preIdx2])
    };
    Vec3f rtNs[3] = {
        Vec3f(preNx[preIdx0], preNy[preIdx0], preNz[preIdx0]),
        Vec3f(preNx[preIdx1], preNy[preIdx1], preNz[preIdx1]),
        Vec3f(preNx[preIdx2], preNy[preIdx2], preNz[preIdx2])
    };
    Vec4f rtCs[3] = {
        Vec4f(preCr[preIdx0], preCg[preIdx0], preCb[preIdx0], preCa[preIdx0]),
        Vec4f(preCr[preIdx1], preCg[preIdx1], preCb[preIdx1], preCa[preIdx1]),
        Vec4f(preCr[preIdx2], preCg[preIdx2], preCb[preIdx2], preCa[preIdx2])
    };

    // If all infront, copy
    if (side[0] >= 0 && side[1] >= 0 && side[2] >= 0) {
        ULInt idx0 = atomicAdd(postCounter, 1) * 3;
        ULInt idx1 = idx0 + 1;
        ULInt idx2 = idx0 + 2;

        postWx[idx0] = rtWs[0].x; postWx[idx1] = rtWs[1].x; postWx[idx2] = rtWs[2].x;
        postWy[idx0] = rtWs[0].y; postWy[idx1] = rtWs[1].y; postWy[idx2] = rtWs[2].y;
        postWz[idx0] = rtWs[0].z; postWz[idx1] = rtWs[1].z; postWz[idx2] = rtWs[2].z;

        postTu[idx0] = rtTs[0].x; postTu[idx1] = rtTs[1].x; postTu[idx2] = rtTs[2].x;
        postTv[idx0] = rtTs[0].y; postTv[idx1] = rtTs[1].y; postTv[idx2] = rtTs[2].y;

        postNx[idx0] = rtNs[0].x; postNx[idx1] = rtNs[1].x; postNx[idx2] = rtNs[2].x;
        postNy[idx0] = rtNs[0].y; postNy[idx1] = rtNs[1].y; postNy[idx2] = rtNs[2].y;
        postNz[idx0] = rtNs[0].z; postNz[idx1] = rtNs[1].z; postNz[idx2] = rtNs[2].z;

        postCr[idx0] = rtCs[0].x; postCr[idx1] = rtCs[1].x; postCr[idx2] = rtCs[2].x;
        postCg[idx0] = rtCs[0].y; postCg[idx1] = rtCs[1].y; postCg[idx2] = rtCs[2].y;
        postCb[idx0] = rtCs[0].z; postCb[idx1] = rtCs[1].z; postCb[idx2] = rtCs[2].z;
        postCa[idx0] = rtCs[0].w; postCa[idx1] = rtCs[1].w; postCa[idx2] = rtCs[2].w;

        return;
    }

    // Everything will be interpolated
    Vec3f newWs[4];
    Vec2f newTs[4];
    Vec3f newNs[4];
    Vec4f newCs[4];

    int newVcount = 0;

    /* Explaination

    We will go AB, BC, CA, with the first vertices being the anchor

    Both in front: append A
    Both in back: ignore
    A in front, B in back: append A and intersect AB
    B in front, A in back: append intersection

    Note: front and back here are relative to the frustum plane
    not to just the near plane
    */

    for (int a = 0; a < 3; a++) {
        int b = (a + 1) % 3;

        // Find plane side
        float sideA = side[a];
        float sideB = side[b];

        if (sideA < 0 && sideB < 0) continue;

        if (sideA >= 0 && sideB >= 0) {
            newWs[newVcount] = rtWs[a];
            newTs[newVcount] = rtTs[a];
            newNs[newVcount] = rtNs[a];
            newCs[newVcount] = rtCs[a];
            newVcount++;
            continue;
        }

        // Find intersection
        float tFact = -sideA / (sideB - sideA);

        Vec3f w = rtWs[a] + (rtWs[b] - rtWs[a]) * tFact;
        Vec2f t = rtTs[a] + (rtTs[b] - rtTs[a]) * tFact;
        Vec3f n = rtNs[a] + (rtNs[b] - rtNs[a]) * tFact;
        Vec4f c = rtCs[a] + (rtCs[b] - rtCs[a]) * tFact;

        if (sideA > 0) {
            // Append A
            newWs[newVcount] = rtWs[a];
            newTs[newVcount] = rtTs[a];
            newNs[newVcount] = rtNs[a];
            newCs[newVcount] = rtCs[a];
            newVcount++;
            // Append intersection
            newWs[newVcount] = w;
            newTs[newVcount] = t;
            newNs[newVcount] = n;
            newCs[newVcount] = c;
            newVcount++;
        } else {
            newWs[newVcount] = w;
            newTs[newVcount] = t;
            newNs[newVcount] = n;
            newCs[newVcount] = c;
            newVcount++;
        }
    }

    // If 4 point: create 2 faces A B C, A C D
    for (int i = 0; i < newVcount - 2; i++) {
        ULInt idx0 = atomicAdd(postCounter, 1) * 3;
        ULInt idx1 = idx0 + 1;
        ULInt idx2 = idx0 + 2;

        postWx[idx0] = newWs[0].x; postWx[idx1] = newWs[i + 1].x; postWx[idx2] = newWs[i + 2].x;
        postWy[idx0] = newWs[0].y; postWy[idx1] = newWs[i + 1].y; postWy[idx2] = newWs[i + 2].y;
        postWz[idx0] = newWs[0].z; postWz[idx1] = newWs[i + 1].z; postWz[idx2] = newWs[i + 2].z;

        postTu[idx0] = newTs[0].x; postTu[idx1] = newTs[i + 1].x; postTu[idx2] = newTs[i + 2].x;
        postTv[idx0] = newTs[0].y; postTv[idx1] = newTs[i + 1].y; postTv[idx2] = newTs[i + 2].y;

        postNx[idx0] = newNs[0].x; postNx[idx1] = newNs[i + 1].x; postNx[idx2] = newNs[i + 2].x;
        postNy[idx0] = newNs[0].y; postNy[idx1] = newNs[i + 1].y; postNy[idx2] = newNs[i + 2].y;
        postNz[idx0] = newNs[0].z; postNz[idx1] = newNs[i + 1].z; postNz[idx2] = newNs[i + 2].z;

        postCr[idx0] = newCs[0].x; postCr[idx1] = newCs[i + 1].x; postCr[idx2] = newCs[i + 2].x;
        postCg[idx0] = newCs[0].y; postCg[idx1] = newCs[i + 1].y; postCg[idx2] = newCs[i + 2].y;
        postCb[idx0] = newCs[0].z; postCb[idx1] = newCs[i + 1].z; postCb[idx2] = newCs[i + 2].z;
        postCa[idx0] = newCs[0].w; postCa[idx1] = newCs[i + 1].w; postCa[idx2] = newCs[i + 2].w;
    }
}

// Depth map creation
__global__ void createDepthMapKernel(
    const float *runtimeSx, const float *runtimeSy, const float *runtimeSz, const float *runtimeSw,
    ULLInt faceCounter, ULLInt faceOffset,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId,
    float *buffBaryX, float *buffBaryY, float *buffBaryZ,
    int buffWidth, int buffHeight, int tileNumX, int tileNumY, int tileSizeX, int tileSizeY
) {
    ULLInt tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    ULLInt fIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdx >= tileNumX * tileNumY || fIdx >= faceCounter) return;
    fIdx += faceOffset;

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
        // Ignore if out of bound
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