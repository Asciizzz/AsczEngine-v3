#include <VertexShader.cuh>

// Static functions
__device__ bool VertexShader::insideFrustum(const Vec4f &v) {
    return  v.x >= -v.w && v.x <= v.w &&
            v.y >= -v.w && v.y <= v.w &&
            v.z >= -v.w && v.z <= v.w;
}

// Render pipeline

void VertexShader::cameraProjection() {
    Graphic3D &grphic = Graphic3D::instance();
    Camera3D &camera = grphic.camera;
    Mesh3D &mesh = grphic.mesh;

    size_t gridSize = (mesh.world.size + 255) / 256;

    cameraProjectionKernel<<<gridSize, 256>>>(
        mesh.world.x, mesh.world.y, mesh.world.z,
        mesh.screen.x, mesh.screen.y, mesh.screen.z, mesh.screen.w,
        camera.mvp, mesh.world.size
    );
}

void VertexShader::createRuntimeFaces() {
    Graphic3D &grphic = Graphic3D::instance();
    Mesh3D &mesh = grphic.mesh;

    size_t gridSize = (mesh.faces.size / 3 + 255) / 256;

    cudaMemset(grphic.d_rtCount, 0, sizeof(ULLInt));
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
        grphic.d_rtCount
    );
    cudaDeviceSynchronize();
    cudaMemcpy(&grphic.rtCount, grphic.d_rtCount, sizeof(ULLInt), cudaMemcpyDeviceToHost);

    return;

    // Debugging
    float *wx = new float[grphic.rtCount * 3];
    float *wy = new float[grphic.rtCount * 3];
    float *wz = new float[grphic.rtCount * 3];

    cudaMemcpy(wx, grphic.rtFaces.wx, sizeof(float) * grphic.rtCount * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(wy, grphic.rtFaces.wy, sizeof(float) * grphic.rtCount * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(wz, grphic.rtFaces.wz, sizeof(float) * grphic.rtCount * 3, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < grphic.rtCount * 3; i++) {
        std::cout << "(" << wx[i] << " " << wy[i] << " " << wz[i] << ") -";
    }
    std::cout << std::endl;

    delete[] wx, wy, wz;
}

void VertexShader::createDepthMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    // Split the faces into chunks

    size_t chunkNum = (grphic.rtCount + grphic.faceChunkSize - 1) 
                    /  grphic.faceChunkSize;

    dim3 blockSize(16, 32);
    for (size_t i = 0; i < chunkNum; i++) {
        size_t chunkOffset = grphic.faceChunkSize * i;

        size_t curFaceCount = (i == chunkNum - 1) ?
            grphic.rtCount - chunkOffset : grphic.faceChunkSize;

        size_t blockNumTile = (grphic.tileNum + blockSize.x - 1) / blockSize.x;
        size_t blockNumFace = (curFaceCount + blockSize.y - 1) / blockSize.y;
        dim3 blockNum(blockNumTile, blockNumFace);

        createDepthMapKernel<<<blockNum, blockSize>>>(
            grphic.rtFaces.sx, grphic.rtFaces.sy,
            grphic.rtFaces.sz, grphic.rtFaces.sw,
            curFaceCount, chunkOffset,

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

/////////////////////////////////////////////////////////////////////////

// Camera projection
__global__ void cameraProjectionKernel(
    const float *worldX, const float *worldY, const float *worldZ,
    float *screenX, float *screenY, float *screenZ, float *screenW,
    Mat4f mvp, ULLInt numVs
) {
    ULLInt vIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vIdx >= numVs) return;

    Vec4f screen = mvp * Vec4f(worldX[vIdx], worldY[vIdx], worldZ[vIdx], 1);

    screenX[vIdx] = -screen.x; // Flip X
    screenY[vIdx] = screen.y;
    screenZ[vIdx] = screen.z;
    screenW[vIdx] = screen.w;
}

// Create runtime faces
__global__ void createRuntimeFacesKernel(
    // Orginal mesh data
    const float *screenX, const float *screenY, const float *screenZ, const float *screenW,
    const float *worldX, const float *worldY, const float *worldZ,
    const float *normalX, const float *normalY, const float *normalZ,
    const float *textureX, const float *textureY,
    const float *colorX, const float *colorY, const float *colorZ, float *colorW,
    const ULLInt *faceWs, const ULLInt *faceTs, const ULLInt *faceNs, ULLInt numFs,

    // Runtime faces
    float *rtSx, float *rtSy, float *rtSz, float *rtSw,
    float *rtWx, float *rtWy, float *rtWz,
    float *rtTu, float *rtTv,
    float *rtNx, float *rtNy, float *rtNz,
    float *rtCr, float *rtCg, float *rtCb, float *rtCa,
    ULLInt *rtCount
) {
    ULLInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= numFs) return;

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    ULLInt fw[3] = {faceWs[idx0], faceWs[idx1], faceWs[idx2]};
    ULLInt ft[3] = {faceTs[idx0], faceTs[idx1], faceTs[idx2]};
    ULLInt fn[3] = {faceNs[idx0], faceNs[idx1], faceNs[idx2]};

    // Early culling (for outside the frustum)
    Vec4f rtSs[3] = {
        Vec4f(screenX[fw[0]], screenY[fw[0]], screenZ[fw[0]], screenW[fw[0]]),
        Vec4f(screenX[fw[1]], screenY[fw[1]], screenZ[fw[1]], screenW[fw[1]]),
        Vec4f(screenX[fw[2]], screenY[fw[2]], screenZ[fw[2]], screenW[fw[2]])
    };

    // If all outside, ignore
    bool allOutLeft = rtSs[0].x < -rtSs[0].w && rtSs[1].x < -rtSs[1].w && rtSs[2].x < -rtSs[2].w;
    bool allOutRight = rtSs[0].x > rtSs[0].w && rtSs[1].x > rtSs[1].w && rtSs[2].x > rtSs[2].w;
    bool allOutTop = rtSs[0].y < -rtSs[0].w && rtSs[1].y < -rtSs[1].w && rtSs[2].y < -rtSs[2].w;
    bool allOutBottom = rtSs[0].y > rtSs[0].w && rtSs[1].y > rtSs[1].w && rtSs[2].y > rtSs[2].w;
    bool allOutNear = rtSs[0].z < -rtSs[0].w && rtSs[1].z < -rtSs[1].w && rtSs[2].z < -rtSs[2].w;
    bool allOutFar = rtSs[0].z > rtSs[0].w && rtSs[1].z > rtSs[1].w && rtSs[2].z > rtSs[2].w;
    if (allOutLeft || allOutRight || allOutTop || allOutBottom || allOutNear || allOutFar) return;

    Vec3f rtWs[3] = {
        Vec3f(worldX[fw[0]], worldY[fw[0]], worldZ[fw[0]]),
        Vec3f(worldX[fw[1]], worldY[fw[1]], worldZ[fw[1]]),
        Vec3f(worldX[fw[2]], worldY[fw[2]], worldZ[fw[2]])
    };
    Vec2f rtTs[3] = {
        Vec2f(textureX[ft[0]], textureY[ft[0]]),
        Vec2f(textureX[ft[1]], textureY[ft[1]]),
        Vec2f(textureX[ft[2]], textureY[ft[2]])
    };
    Vec3f rtNs[3] = {
        Vec3f(normalX[fn[0]], normalY[fn[0]], normalZ[fn[0]]),
        Vec3f(normalX[fn[1]], normalY[fn[1]], normalZ[fn[1]]),
        Vec3f(normalX[fn[2]], normalY[fn[2]], normalZ[fn[2]])
    };
    Vec4f rtCs[3] = {
        Vec4f(colorX[fw[0]], colorY[fw[0]], colorZ[fw[0]], colorW[fw[0]]),
        Vec4f(colorX[fw[1]], colorY[fw[1]], colorZ[fw[1]], colorW[fw[1]]),
        Vec4f(colorX[fw[2]], colorY[fw[2]], colorZ[fw[2]], colorW[fw[2]])
    };

    // If all inside, return
    bool inside1 = VertexShader::insideFrustum(rtSs[0]);
    bool inside2 = VertexShader::insideFrustum(rtSs[1]);
    bool inside3 = VertexShader::insideFrustum(rtSs[2]);
    if (inside1 && inside2 && inside3) {
        ULLInt idx0 = atomicAdd(rtCount, 1) * 3;
        ULLInt idx1 = idx0 + 1;
        ULLInt idx2 = idx0 + 2;

        rtSx[idx0] = rtSs[0].x; rtSx[idx1] = rtSs[1].x; rtSx[idx2] = rtSs[2].x;
        rtSy[idx0] = rtSs[0].y; rtSy[idx1] = rtSs[1].y; rtSy[idx2] = rtSs[2].y;
        rtSz[idx0] = rtSs[0].z; rtSz[idx1] = rtSs[1].z; rtSz[idx2] = rtSs[2].z;
        rtSw[idx0] = rtSs[0].w; rtSw[idx1] = rtSs[1].w; rtSw[idx2] = rtSs[2].w;

        rtWx[idx0] = rtWs[0].x; rtWx[idx1] = rtWs[1].x; rtWx[idx2] = rtWs[2].x;
        rtWy[idx0] = rtWs[0].y; rtWy[idx1] = rtWs[1].y; rtWy[idx2] = rtWs[2].y;
        rtWz[idx0] = rtWs[0].z; rtWz[idx1] = rtWs[1].z; rtWz[idx2] = rtWs[2].z;

        rtTu[idx0] = rtTs[0].x; rtTu[idx1] = rtTs[1].x; rtTu[idx2] = rtTs[2].x;
        rtTv[idx0] = rtTs[0].y; rtTv[idx1] = rtTs[1].y; rtTv[idx2] = rtTs[2].y;

        rtNx[idx0] = rtNs[0].x; rtNx[idx1] = rtNs[1].x; rtNx[idx2] = rtNs[2].x;
        rtNy[idx0] = rtNs[0].y; rtNy[idx1] = rtNs[1].y; rtNy[idx2] = rtNs[2].y;
        rtNz[idx0] = rtNs[0].z; rtNz[idx1] = rtNs[1].z; rtNz[idx2] = rtNs[2].z;

        rtCr[idx0] = rtCs[0].x; rtCr[idx1] = rtCs[1].x; rtCr[idx2] = rtCs[2].x;
        rtCg[idx0] = rtCs[0].y; rtCg[idx1] = rtCs[1].y; rtCg[idx2] = rtCs[2].y;
        rtCb[idx0] = rtCs[0].z; rtCb[idx1] = rtCs[1].z; rtCb[idx2] = rtCs[2].z;
        rtCa[idx0] = rtCs[0].w; rtCa[idx1] = rtCs[1].w; rtCa[idx2] = rtCs[2].w;

        return;
    }

    Vertex vertices[4];
    int newVcount = 0;

    /* Explaination

    We will go AB, BC, CA, with the first vertices being the anchor

    Both in front: append A
    Both in back: ignore
    A in front, B in back: append A and intersect AB
    B in front, A in back: append intersection

    Note: front and back here are relative to the frustum plane
    not to just the near plane

    Editor Note:

    Currently clipping is performed in actual 3D space
    We want to perform it in clip space instead
    */


    // These will be used for perspective correction in the interpolation
    Vec4f sDivW[3] = { rtSs[0]/rtSs[0].w, rtSs[1]/rtSs[1].w, rtSs[2]/rtSs[2].w };
    Vec3f wDivW[3] = { rtWs[0]/rtSs[0].w, rtWs[1]/rtSs[1].w, rtWs[2]/rtSs[2].w };
    Vec2f tDivW[3] = { rtTs[0]/rtSs[0].w, rtTs[1]/rtSs[1].w, rtTs[2]/rtSs[2].w };
    Vec3f nDivW[3] = { rtNs[0]/rtSs[0].w, rtNs[1]/rtSs[1].w, rtNs[2]/rtSs[2].w };
    Vec4f cDivW[3] = { rtCs[0]/rtSs[0].w, rtCs[1]/rtSs[1].w, rtCs[2]/rtSs[2].w };

    for (int a = 0; a < 3; a++) {
        int b = (a + 1) % 3;

        if (rtSs[a].z < -rtSs[a].w && rtSs[b].z < -rtSs[b].w) continue;

        if (rtSs[a].z >= -rtSs[a].w && rtSs[b].z >= -rtSs[b].w) {
            vertices[newVcount].screen = rtSs[a];
            vertices[newVcount].world = rtWs[a];
            vertices[newVcount].texture = rtTs[a];
            vertices[newVcount].normal = rtNs[a];
            vertices[newVcount].color = rtCs[a];
            newVcount++;
            continue;
        }

        // Find interpolation factor
        float tFact =
            (-1 - rtSs[a].z/rtSs[a].w) /
            (rtSs[b].z/rtSs[b].w - rtSs[a].z/rtSs[a].w);

        float homo1DivW = 1/rtSs[a].w + (1/rtSs[b].w - 1/rtSs[a].w) * tFact;

        Vec4f s_w = sDivW[a] + (sDivW[b] - sDivW[a]) * tFact;
        Vec3f w_w = wDivW[a] + (wDivW[b] - wDivW[a]) * tFact;
        Vec2f t_w = tDivW[a] + (tDivW[b] - tDivW[a]) * tFact;
        Vec3f n_w = nDivW[a] + (nDivW[b] - nDivW[a]) * tFact;
        Vec4f c_w = cDivW[a] + (cDivW[b] - cDivW[a]) * tFact;

        Vec4f s = s_w / homo1DivW;
        Vec3f w = w_w / homo1DivW;
        Vec2f t = t_w / homo1DivW;
        Vec3f n = n_w / homo1DivW;
        Vec4f c = c_w / homo1DivW;

        if (rtSs[a].z >= -rtSs[a].w) {
            // Append A
            vertices[newVcount].screen = rtSs[a];
            vertices[newVcount].world = rtWs[a];
            vertices[newVcount].texture = rtTs[a];
            vertices[newVcount].normal = rtNs[a];
            vertices[newVcount].color = rtCs[a];
            newVcount++;
        }

        vertices[newVcount].screen = s;
        vertices[newVcount].world = w;
        vertices[newVcount].texture = t;
        vertices[newVcount].normal = n;
        vertices[newVcount].color = c;
        newVcount++;
    }

    if (newVcount < 3) return;

    // If 4 point: create 2 faces A B C, A C D
    for (int i = 0; i < newVcount - 2; i++) {
        ULLInt idx0 = atomicAdd(rtCount, 1) * 3;
        ULLInt idx1 = idx0 + 1;
        ULLInt idx2 = idx0 + 2;

        rtSx[idx0] = vertices[0].screen.x; rtSx[idx1] = vertices[i + 1].screen.x; rtSx[idx2] = vertices[i + 2].screen.x;
        rtSy[idx0] = vertices[0].screen.y; rtSy[idx1] = vertices[i + 1].screen.y; rtSy[idx2] = vertices[i + 2].screen.y;
        rtSz[idx0] = vertices[0].screen.z; rtSz[idx1] = vertices[i + 1].screen.z; rtSz[idx2] = vertices[i + 2].screen.z;
        rtSw[idx0] = vertices[0].screen.w; rtSw[idx1] = vertices[i + 1].screen.w; rtSw[idx2] = vertices[i + 2].screen.w;

        rtWx[idx0] = vertices[0].world.x; rtWx[idx1] = vertices[i + 1].world.x; rtWx[idx2] = vertices[i + 2].world.x;
        rtWy[idx0] = vertices[0].world.y; rtWy[idx1] = vertices[i + 1].world.y; rtWy[idx2] = vertices[i + 2].world.y;
        rtWz[idx0] = vertices[0].world.z; rtWz[idx1] = vertices[i + 1].world.z; rtWz[idx2] = vertices[i + 2].world.z;

        rtTu[idx0] = vertices[0].texture.x; rtTu[idx1] = vertices[i + 1].texture.x; rtTu[idx2] = vertices[i + 2].texture.x;
        rtTv[idx0] = vertices[0].texture.y; rtTv[idx1] = vertices[i + 1].texture.y; rtTv[idx2] = vertices[i + 2].texture.y;

        rtNx[idx0] = vertices[0].normal.x; rtNx[idx1] = vertices[i + 1].normal.x; rtNx[idx2] = vertices[i + 2].normal.x;
        rtNy[idx0] = vertices[0].normal.y; rtNy[idx1] = vertices[i + 1].normal.y; rtNy[idx2] = vertices[i + 2].normal.y;
        rtNz[idx0] = vertices[0].normal.z; rtNz[idx1] = vertices[i + 1].normal.z; rtNz[idx2] = vertices[i + 2].normal.z;

        rtCr[idx0] = vertices[0].color.x; rtCr[idx1] = vertices[i + 1].color.x; rtCr[idx2] = vertices[i + 2].color.x;
        rtCg[idx0] = vertices[0].color.y; rtCg[idx1] = vertices[i + 1].color.y; rtCg[idx2] = vertices[i + 2].color.y;
        rtCb[idx0] = vertices[0].color.z; rtCb[idx1] = vertices[i + 1].color.z; rtCb[idx2] = vertices[i + 2].color.z;
        rtCa[idx0] = vertices[0].color.w; rtCa[idx1] = vertices[i + 1].color.w; rtCa[idx2] = vertices[i + 2].color.w;
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