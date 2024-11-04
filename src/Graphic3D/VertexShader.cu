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

    ULLInt gridSize = (mesh.faces.size / 3 + 255) / 256;

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
        grphic.rtFaces.active, grphic.d_rtCount
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
    ULLInt rtSize = grphic.rtFaces.size / 3;

    ULLInt chunkNum = (rtSize + grphic.faceChunkSize - 1) 
                    /  grphic.faceChunkSize;

    dim3 blockSize(16, 32);
    for (ULLInt i = 0; i < chunkNum; i++) {
        ULLInt chunkOffset = grphic.faceChunkSize * i;

        ULLInt curFaceCount = (i == chunkNum - 1) ?
            rtSize - chunkOffset : grphic.faceChunkSize;

        ULLInt blockNumTile = (grphic.tileNum + blockSize.x - 1) / blockSize.x;
        ULLInt blockNumFace = (curFaceCount + blockSize.y - 1) / blockSize.y;
        dim3 blockNum(blockNumTile, blockNumFace);

        createDepthMapKernel<<<blockNum, blockSize>>>(
            grphic.rtFaces.active,
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
    const float *wx, const float *wy, const float *wz,
    float *sx, float *sy, float *sz, float *sw,
    Mat4f mvp, ULLInt numVs
) {
    ULLInt vIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vIdx >= numVs) return;

    Vec4f screen = mvp * Vec4f(wx[vIdx], wy[vIdx], wz[vIdx], 1);

    sx[vIdx] = -screen.x;
    sy[vIdx] = screen.y;
    sz[vIdx] = screen.z;
    sw[vIdx] = screen.w;
}

__global__ void createRuntimeFacesKernel(
    // Orginal mesh data
    const float *sx, const float *sy, const float *sz, const float *sw,
    const float *wx, const float *wy, const float *wz,
    const float *nx, const float *ny, const float *nz,
    const float *tu, const float *tv,
    const float *cr, const float *cg, const float *cb, const float *ca,
    const ULLInt *fWs, const ULLInt *fTs, const ULLInt *fNs, ULLInt numFs,

    // Runtime faces
    float *rtSx, float *rtSy, float *rtSz, float *rtSw,
    float *rtWx, float *rtWy, float *rtWz,
    float *rtTu, float *rtTv,
    float *rtNx, float *rtNy, float *rtNz,
    float *rtCr, float *rtCg, float *rtCb, float *rtCa,
    bool *rtActive, ULLInt *rtCount
) {
    ULLInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= numFs) return;

    // Reset Active
    rtActive[fIdx * 4] = false;
    rtActive[fIdx * 4 + 1] = false;
    rtActive[fIdx * 4 + 2] = false;
    rtActive[fIdx * 4 + 3] = false;

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    ULLInt fw[3] = {fWs[idx0], fWs[idx1], fWs[idx2]};
    ULLInt ft[3] = {fTs[idx0], fTs[idx1], fTs[idx2]};
    ULLInt fn[3] = {fNs[idx0], fNs[idx1], fNs[idx2]};

    // Early culling (for outside the frustum)
    Vec4f rtSs[3] = {
        Vec4f(sx[fw[0]], sy[fw[0]], sz[fw[0]], sw[fw[0]]),
        Vec4f(sx[fw[1]], sy[fw[1]], sz[fw[1]], sw[fw[1]]),
        Vec4f(sx[fw[2]], sy[fw[2]], sz[fw[2]], sw[fw[2]])
    };

    // If all on one side, ignore
    bool allOutLeft = rtSs[0].x < -rtSs[0].w && rtSs[1].x < -rtSs[1].w && rtSs[2].x < -rtSs[2].w;
    bool allOutRight = rtSs[0].x > rtSs[0].w && rtSs[1].x > rtSs[1].w && rtSs[2].x > rtSs[2].w;
    bool allOutTop = rtSs[0].y < -rtSs[0].w && rtSs[1].y < -rtSs[1].w && rtSs[2].y < -rtSs[2].w;
    bool allOutBottom = rtSs[0].y > rtSs[0].w && rtSs[1].y > rtSs[1].w && rtSs[2].y > rtSs[2].w;
    bool allOutNear = rtSs[0].z < -rtSs[0].w && rtSs[1].z < -rtSs[1].w && rtSs[2].z < -rtSs[2].w;
    bool allOutFar = rtSs[0].z > rtSs[0].w && rtSs[1].z > rtSs[1].w && rtSs[2].z > rtSs[2].w;
    bool allBehind = rtSs[0].w < 0 && rtSs[1].w < 0 && rtSs[2].w < 0;
    if (allOutLeft || allOutRight || allOutTop || allOutBottom || allOutNear || allOutFar || allBehind)
        return;

    Vec3f rtWs[3] = {
        Vec3f(wx[fw[0]], wy[fw[0]], wz[fw[0]]),
        Vec3f(wx[fw[1]], wy[fw[1]], wz[fw[1]]),
        Vec3f(wx[fw[2]], wy[fw[2]], wz[fw[2]])
    };
    Vec2f rtTs[3] = {
        Vec2f(tu[ft[0]], tv[ft[0]]),
        Vec2f(tu[ft[1]], tv[ft[1]]),
        Vec2f(tu[ft[2]], tv[ft[2]])
    };
    Vec3f rtNs[3] = {
        Vec3f(nx[fn[0]], ny[fn[0]], nz[fn[0]]),
        Vec3f(nx[fn[1]], ny[fn[1]], nz[fn[1]]),
        Vec3f(nx[fn[2]], ny[fn[2]], nz[fn[2]])
    };
    Vec4f rtCs[3] = {
        Vec4f(cr[fw[0]], cg[fw[0]], cb[fw[0]], ca[fw[0]]),
        Vec4f(cr[fw[1]], cg[fw[1]], cb[fw[1]], ca[fw[1]]),
        Vec4f(cr[fw[2]], cg[fw[2]], cb[fw[2]], ca[fw[2]])
    };

    // If all inside, return
    bool inside1 = VertexShader::insideFrustum(rtSs[0]);
    bool inside2 = VertexShader::insideFrustum(rtSs[1]);
    bool inside3 = VertexShader::insideFrustum(rtSs[2]);
    if (inside1 && inside2 && inside3) {
        ULLInt idx0 = fIdx * 12;
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

        rtActive[fIdx * 4] = true;

        return;
    }

    int temp1Count = 3;
    Vec4f tempS1[4] = { rtSs[0], rtSs[1], rtSs[2] };
    Vec3f tempW1[4] = { rtWs[0], rtWs[1], rtWs[2] };
    Vec2f tempT1[4] = { rtTs[0], rtTs[1], rtTs[2] };
    Vec3f tempN1[4] = { rtNs[0], rtNs[1], rtNs[2] };
    Vec4f tempC1[4] = { rtCs[0], rtCs[1], rtCs[2] };

    int temp2Count = 0;
    Vec4f tempS2[4];
    Vec3f tempW2[4];
    Vec2f tempT2[4];
    Vec3f tempN2[4];
    Vec4f tempC2[4];

    // Clip to near plane
    for (int a = 0; a < temp1Count; a++) {
        int b = (a + 1) % temp1Count;

        float swA = tempS1[a].w;
        float swB = tempS1[b].w;
        float szA = tempS1[a].z;
        float szB = tempS1[b].z;

        if (szA < -swA && szB < -swB) continue;

        if (szA >= -swA && szB >= -swB) {
            tempS2[temp2Count] = tempS1[a];
            tempW2[temp2Count] = tempW1[a];
            tempT2[temp2Count] = tempT1[a];
            tempN2[temp2Count] = tempN1[a];
            tempC2[temp2Count] = tempC1[a];
            temp2Count++;
            continue;
        }

        float tFact = (-1 - szA/swA) / (szB/swB - szA/swA);
        Vec4f s_w = tempS1[a]/swA + (tempS1[b]/swB - tempS1[a]/swA) * tFact;
        Vec3f w_w = tempW1[a]/swA + (tempW1[b]/swB - tempW1[a]/swA) * tFact;
        Vec2f t_w = tempT1[a]/swA + (tempT1[b]/swB - tempT1[a]/swA) * tFact;
        Vec3f n_w = tempN1[a]/swA + (tempN1[b]/swB - tempN1[a]/swA) * tFact;
        Vec4f c_w = tempC1[a]/swA + (tempC1[b]/swB - tempC1[a]/swA) * tFact;

        float homo1DivW = 1/swA + (1/swB - 1/swA) * tFact;
        Vec4f s = s_w / homo1DivW;
        Vec3f w = w_w / homo1DivW;
        Vec2f t = t_w / homo1DivW;
        Vec3f n = n_w / homo1DivW;
        Vec4f c = c_w / homo1DivW;

        if (szA >= -swA) {
            tempS2[temp2Count] = tempS1[a];
            tempW2[temp2Count] = tempW1[a];
            tempT2[temp2Count] = tempT1[a];
            tempN2[temp2Count] = tempN1[a];
            tempC2[temp2Count] = tempC1[a];
            temp2Count++;
        }

        tempS2[temp2Count] = s;
        tempW2[temp2Count] = w;
        tempT2[temp2Count] = t;
        tempN2[temp2Count] = n;
        tempC2[temp2Count] = c;
        temp2Count++;
    }
    if (temp2Count < 3) return;

    // n points <=> n - 2 faces
    for (int i = 0; i < temp2Count - 2; i++) {
        ULLInt idx0 = fIdx * 12 + i * 3;
        ULLInt idx1 = idx0 + 1;
        ULLInt idx2 = idx0 + 2;

        rtSx[idx0] = tempS2[0].x; rtSx[idx1] = tempS2[i + 1].x; rtSx[idx2] = tempS2[i + 2].x;
        rtSy[idx0] = tempS2[0].y; rtSy[idx1] = tempS2[i + 1].y; rtSy[idx2] = tempS2[i + 2].y;
        rtSz[idx0] = tempS2[0].z; rtSz[idx1] = tempS2[i + 1].z; rtSz[idx2] = tempS2[i + 2].z;
        rtSw[idx0] = tempS2[0].w; rtSw[idx1] = tempS2[i + 1].w; rtSw[idx2] = tempS2[i + 2].w;

        rtWx[idx0] = tempW2[0].x; rtWx[idx1] = tempW2[i + 1].x; rtWx[idx2] = tempW2[i + 2].x;
        rtWy[idx0] = tempW2[0].y; rtWy[idx1] = tempW2[i + 1].y; rtWy[idx2] = tempW2[i + 2].y;
        rtWz[idx0] = tempW2[0].z; rtWz[idx1] = tempW2[i + 1].z; rtWz[idx2] = tempW2[i + 2].z;

        rtTu[idx0] = tempT2[0].x; rtTu[idx1] = tempT2[i + 1].x; rtTu[idx2] = tempT2[i + 2].x;
        rtTv[idx0] = tempT2[0].y; rtTv[idx1] = tempT2[i + 1].y; rtTv[idx2] = tempT2[i + 2].y;

        rtNx[idx0] = tempN2[0].x; rtNx[idx1] = tempN2[i + 1].x; rtNx[idx2] = tempN2[i + 2].x;
        rtNy[idx0] = tempN2[0].y; rtNy[idx1] = tempN2[i + 1].y; rtNy[idx2] = tempN2[i + 2].y;
        rtNz[idx0] = tempN2[0].z; rtNz[idx1] = tempN2[i + 1].z; rtNz[idx2] = tempN2[i + 2].z;

        rtCr[idx0] = tempC2[0].x; rtCr[idx1] = tempC2[i + 1].x; rtCr[idx2] = tempC2[i + 2].x;
        rtCg[idx0] = tempC2[0].y; rtCg[idx1] = tempC2[i + 1].y; rtCg[idx2] = tempC2[i + 2].y;
        rtCb[idx0] = tempC2[0].z; rtCb[idx1] = tempC2[i + 1].z; rtCb[idx2] = tempC2[i + 2].z;
        rtCa[idx0] = tempC2[0].w; rtCa[idx1] = tempC2[i + 1].w; rtCa[idx2] = tempC2[i + 2].w;

        rtActive[fIdx * 4 + i] = true;
    }
}

// Depth map creation
__global__ void createDepthMapKernel(
    const bool *runtimeActive,
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

    if (!runtimeActive[fIdx]) return;

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    float sw0 = runtimeSw[idx0];
    float sw1 = runtimeSw[idx1];
    float sw2 = runtimeSw[idx2];

    float bx0 = (runtimeSx[idx0] / sw0 + 1) * buffWidth / 2;
    float bx1 = (runtimeSx[idx1] / sw1 + 1) * buffWidth / 2;
    float bx2 = (runtimeSx[idx2] / sw2 + 1) * buffWidth / 2;

    float by0 = (1 - runtimeSy[idx0] / sw0) * buffHeight / 2;
    float by1 = (1 - runtimeSy[idx1] / sw1) * buffHeight / 2;
    float by2 = (1 - runtimeSy[idx2] / sw2) * buffHeight / 2;

    float bz0 = (runtimeSz[idx0] / sw0 + 1) / 2;
    float bz1 = (runtimeSz[idx1] / sw1 + 1) / 2;
    float bz2 = (runtimeSz[idx2] / sw2 + 1) / 2;

    // Buffer bounding box based on the tile

    int tX = tIdx % tileNumX;
    int tY = tIdx / tileNumX;

    int bufferMinX = tX * tileSizeX;
    int bufferMaxX = bufferMinX + tileSizeX;
    int bufferMinY = tY * tileSizeY;
    int bufferMaxY = bufferMinY + tileSizeY;

    // Bounding box
    int minX = min(min(bx0, bx1), bx2);
    int maxX = max(max(bx0, bx1), bx2);
    int minY = min(min(by0, by1), by2);
    int maxY = max(max(by0, by1), by2);

    // If bounding box not in tile area, return
    if (minX > bufferMaxX ||
        maxX < bufferMinX ||
        minY > bufferMaxY ||
        maxY < bufferMinY) return;

    // Clip the bounding box based on the buffer tile
    minX = max(minX, bufferMinX);
    maxX = min(maxX, bufferMaxX);
    minY = max(minY, bufferMinY);
    maxY = min(maxY, bufferMaxY);

    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
        int bIdx = x + y * buffWidth;

        Vec3f bary = Vec3f::bary(
            Vec2f(x, y), Vec2f(bx0, by0), Vec2f(bx1, by1), Vec2f(bx2, by2)
        );
        // Ignore if out of bound
        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float depth = bary.x * bz0 + bary.y * bz1 + bary.z * bz2; 

        if (atomicMinFloat(&buffDepth[bIdx], depth)) {
            buffDepth[bIdx] = depth;
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