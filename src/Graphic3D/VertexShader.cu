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

    ULLInt gridSize = (mesh.world.size + 255) / 256;
    cameraProjectionKernel<<<gridSize, 256>>>(
        mesh.world.x, mesh.world.y, mesh.world.z,
        mesh.screen.x, mesh.screen.y, mesh.screen.z, mesh.screen.w,
        camera.mvp, mesh.world.size
    );
}

void VertexShader::frustumCulling() {
    Graphic3D &grphic = Graphic3D::instance();
    Mesh3D &mesh = grphic.mesh;
    Face3D &faces = grphic.rtFaces;

    ULLInt gridSize = (mesh.faces.size / 3 + 255) / 256;
    frustumCullingKernel<<<gridSize, 256>>>(
        mesh.screen.x, mesh.screen.y, mesh.screen.z, mesh.screen.w,
        mesh.world.x, mesh.world.y, mesh.world.z,
        mesh.texture.x, mesh.texture.y,
        mesh.normal.x, mesh.normal.y, mesh.normal.z,
        mesh.color.x, mesh.color.y, mesh.color.z, mesh.color.w,
        mesh.faces.v, mesh.faces.t, mesh.faces.n, mesh.faces.size / 3,

        faces.sx, faces.sy, faces.sz, faces.sw,
        faces.wx, faces.wy, faces.wz,
        faces.tu, faces.tv,
        faces.nx, faces.ny, faces.nz,
        faces.cr, faces.cg, faces.cb, faces.ca,
        faces.active
    );
    cudaDeviceSynchronize();

    cudaMemset(grphic.d_rtCount, 0, sizeof(ULLInt));
    gridSize = (faces.size / 3 + 255) / 256;
    runtimeIndexingKernel<<<gridSize, 256>>>(
        faces.active, grphic.rtIndex, grphic.d_rtCount, faces.size / 3
    );
    cudaDeviceSynchronize();
    cudaMemcpy(&grphic.rtCount, grphic.d_rtCount, sizeof(ULLInt), cudaMemcpyDeviceToHost);
}

void VertexShader::createDepthMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;
    Face3D &faces = grphic.rtFaces;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    // Split the faces into chunks
    ULLInt rtSize = grphic.rtCount;

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
            grphic.rtIndex,
            faces.active, faces.sx, faces.sy, faces.sz, faces.sw,
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
    Face3D &faces = grphic.rtFaces;

    rasterizationKernel<<<buffer.blockNum, buffer.blockSize>>>(
        faces.sw,
        faces.wx, faces.wy, faces.wz,
        faces.tu, faces.tv,
        faces.nx, faces.ny, faces.nz,
        faces.cr, faces.cg, faces.cb, faces.ca,

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

__global__ void frustumCullingKernel(
    // Orginal mesh data
    const float *sx, const float *sy, const float *sz, const float *sw,
    const float *wx, const float *wy, const float *wz,
    const float *tu, const float *tv,
    const float *nx, const float *ny, const float *nz,
    const float *cr, const float *cg, const float *cb, const float *ca,
    const ULLInt *fWs, const ULLInt *fTs, const ULLInt *fNs, ULLInt numFs,

    // Runtime faces
    float *rtSx, float *rtSy, float *rtSz, float *rtSw,
    float *rtWx, float *rtWy, float *rtWz,
    float *rtTu, float *rtTv,
    float *rtNx, float *rtNy, float *rtNz,
    float *rtCr, float *rtCg, float *rtCb, float *rtCa,
    bool *rtActive
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
    Vec4f tempS1[6] = { rtSs[0], rtSs[1], rtSs[2] };
    Vec3f tempW1[6] = { rtWs[0], rtWs[1], rtWs[2] };
    Vec2f tempT1[6] = { rtTs[0], rtTs[1], rtTs[2] };
    Vec3f tempN1[6] = { rtNs[0], rtNs[1], rtNs[2] };
    Vec4f tempC1[6] = { rtCs[0], rtCs[1], rtCs[2] };

    int temp2Count = 0;
    Vec4f tempS2[6];
    Vec3f tempW2[6];
    Vec2f tempT2[6];
    Vec3f tempN2[6];
    Vec4f tempC2[6];

    // There are alot of repetition here, will fix it later

    // Clip to near plane
    for (int a = 0; a < temp1Count; a++) {
        int b = (a + 1) % temp1Count;

        float swA = tempS1[a].w, swB = tempS1[b].w;
        float szA = tempS1[a].z, szB = tempS1[b].z;

        if (szA < -swA && szB < -swB) continue;

        if (szA >= -swA) {
            tempS2[temp2Count] = tempS1[a];
            tempW2[temp2Count] = tempW1[a];
            tempT2[temp2Count] = tempT1[a];
            tempN2[temp2Count] = tempN1[a];
            tempC2[temp2Count] = tempC1[a];
            temp2Count++;

            if (szB >= -swB) continue;
        }

        float tFact = (-1 - szA/swA) / (szB/swB - szA/swA);
        Vec4f s_w = tempS1[a]/swA + (tempS1[b]/swB - tempS1[a]/swA) * tFact;
        Vec3f w_w = tempW1[a]/swA + (tempW1[b]/swB - tempW1[a]/swA) * tFact;
        Vec2f t_w = tempT1[a]/swA + (tempT1[b]/swB - tempT1[a]/swA) * tFact;
        Vec3f n_w = tempN1[a]/swA + (tempN1[b]/swB - tempN1[a]/swA) * tFact;
        Vec4f c_w = tempC1[a]/swA + (tempC1[b]/swB - tempC1[a]/swA) * tFact;

        float homo1DivW = 1/swA + (1/swB - 1/swA) * tFact;
        tempS2[temp2Count] = s_w / homo1DivW;
        tempW2[temp2Count] = w_w / homo1DivW;
        tempT2[temp2Count] = t_w / homo1DivW;
        tempN2[temp2Count] = n_w / homo1DivW;
        tempC2[temp2Count] = c_w / homo1DivW;
        temp2Count++;
    }
    if (temp2Count < 3) return;

    // Clip to left plane
    temp1Count = 0;
    for (int a = 0; a < temp2Count; a++) {
        int b = (a + 1) % temp2Count;

        float swA = tempS2[a].w, swB = tempS2[b].w;
        float sxA = tempS2[a].x, sxB = tempS2[b].x;

        if (sxA < -swA && sxB < -swB) continue;

        if (sxA >= -swA) {
            tempS1[temp1Count] = tempS2[a];
            tempW1[temp1Count] = tempW2[a];
            tempT1[temp1Count] = tempT2[a];
            tempN1[temp1Count] = tempN2[a];
            tempC1[temp1Count] = tempC2[a];
            temp1Count++;

            if (sxB >= -swB) continue;
        }

        float tFact = (-1 - sxA/swA) / (sxB/swB - sxA/swA);
        Vec4f s_w = tempS2[a]/swA + (tempS2[b]/swB - tempS2[a]/swA) * tFact;
        Vec3f w_w = tempW2[a]/swA + (tempW2[b]/swB - tempW2[a]/swA) * tFact;
        Vec2f t_w = tempT2[a]/swA + (tempT2[b]/swB - tempT2[a]/swA) * tFact;
        Vec3f n_w = tempN2[a]/swA + (tempN2[b]/swB - tempN2[a]/swA) * tFact;
        Vec4f c_w = tempC2[a]/swA + (tempC2[b]/swB - tempC2[a]/swA) * tFact;

        float homo1DivW = 1/swA + (1/swB - 1/swA) * tFact;
        tempS1[temp1Count] = s_w / homo1DivW;
        tempW1[temp1Count] = w_w / homo1DivW;
        tempT1[temp1Count] = t_w / homo1DivW;
        tempN1[temp1Count] = n_w / homo1DivW;
        tempC1[temp1Count] = c_w / homo1DivW;
        temp1Count++;
    }
    if (temp1Count < 3) return;

    // Clip to right plane
    temp2Count = 0;
    for (int a = 0; a < temp1Count; a++) {
        int b = (a + 1) % temp1Count;

        float swA = tempS1[a].w, swB = tempS1[b].w;
        float sxA = tempS1[a].x, sxB = tempS1[b].x;

        if (sxA > swA && sxB > swB) continue;

        if (sxA <= swA) {
            tempS2[temp2Count] = tempS1[a];
            tempW2[temp2Count] = tempW1[a];
            tempT2[temp2Count] = tempT1[a];
            tempN2[temp2Count] = tempN1[a];
            tempC2[temp2Count] = tempC1[a];
            temp2Count++;

            if (sxB <= swB) continue;
        }

        float tFact = (1 - sxA/swA) / (sxB/swB - sxA/swA);
        Vec4f s_w = tempS1[a]/swA + (tempS1[b]/swB - tempS1[a]/swA) * tFact;
        Vec3f w_w = tempW1[a]/swA + (tempW1[b]/swB - tempW1[a]/swA) * tFact;
        Vec2f t_w = tempT1[a]/swA + (tempT1[b]/swB - tempT1[a]/swA) * tFact;
        Vec3f n_w = tempN1[a]/swA + (tempN1[b]/swB - tempN1[a]/swA) * tFact;
        Vec4f c_w = tempC1[a]/swA + (tempC1[b]/swB - tempC1[a]/swA) * tFact;

        float homo1DivW = 1/swA + (1/swB - 1/swA) * tFact;
        tempS2[temp2Count] = s_w / homo1DivW;
        tempW2[temp2Count] = w_w / homo1DivW;
        tempT2[temp2Count] = t_w / homo1DivW;
        tempN2[temp2Count] = n_w / homo1DivW;
        tempC2[temp2Count] = c_w / homo1DivW;
        temp2Count++;
    }
    if (temp2Count < 3) return;

    // Clip to top plane
    temp1Count = 0;
    for (int a = 0; a < temp2Count; a++) {
        int b = (a + 1) % temp2Count;

        float swA = tempS2[a].w, swB = tempS2[b].w;
        float syA = tempS2[a].y, syB = tempS2[b].y;

        if (syA < -swA && syB < -swB) continue;

        if (syA >= -swA) {
            tempS1[temp1Count] = tempS2[a];
            tempW1[temp1Count] = tempW2[a];
            tempT1[temp1Count] = tempT2[a];
            tempN1[temp1Count] = tempN2[a];
            tempC1[temp1Count] = tempC2[a];
            temp1Count++;

            if (syB >= -swB) continue;
        }

        float tFact = (-1 - syA/swA) / (syB/swB - syA/swA);
        Vec4f s_w = tempS2[a]/swA + (tempS2[b]/swB - tempS2[a]/swA) * tFact;
        Vec3f w_w = tempW2[a]/swA + (tempW2[b]/swB - tempW2[a]/swA) * tFact;
        Vec2f t_w = tempT2[a]/swA + (tempT2[b]/swB - tempT2[a]/swA) * tFact;
        Vec3f n_w = tempN2[a]/swA + (tempN2[b]/swB - tempN2[a]/swA) * tFact;
        Vec4f c_w = tempC2[a]/swA + (tempC2[b]/swB - tempC2[a]/swA) * tFact;

        float homo1DivW = 1/swA + (1/swB - 1/swA) * tFact;
        tempS1[temp1Count] = s_w / homo1DivW;
        tempW1[temp1Count] = w_w / homo1DivW;
        tempT1[temp1Count] = t_w / homo1DivW;
        tempN1[temp1Count] = n_w / homo1DivW;
        tempC1[temp1Count] = c_w / homo1DivW;
        temp1Count++;
    }
    if (temp1Count < 3) return;

    // Clip to bottom plane
    temp2Count = 0;
    for (int a = 0; a < temp1Count; a++) {
        int b = (a + 1) % temp1Count;

        float swA = tempS1[a].w, swB = tempS1[b].w;
        float syA = tempS1[a].y, syB = tempS1[b].y;

        if (syA > swA && syB > swB) continue;

        if (syA <= swA) {
            tempS2[temp2Count] = tempS1[a];
            tempW2[temp2Count] = tempW1[a];
            tempT2[temp2Count] = tempT1[a];
            tempN2[temp2Count] = tempN1[a];
            tempC2[temp2Count] = tempC1[a];
            temp2Count++;

            if (syB <= swB) continue;
        }

        float tFact = (1 - syA/swA) / (syB/swB - syA/swA);
        Vec4f s_w = tempS1[a]/swA + (tempS1[b]/swB - tempS1[a]/swA) * tFact;
        Vec3f w_w = tempW1[a]/swA + (tempW1[b]/swB - tempW1[a]/swA) * tFact;
        Vec2f t_w = tempT1[a]/swA + (tempT1[b]/swB - tempT1[a]/swA) * tFact;
        Vec3f n_w = tempN1[a]/swA + (tempN1[b]/swB - tempN1[a]/swA) * tFact;
        Vec4f c_w = tempC1[a]/swA + (tempC1[b]/swB - tempC1[a]/swA) * tFact;

        float homo1DivW = 1/swA + (1/swB - 1/swA) * tFact;
        tempS2[temp2Count] = s_w / homo1DivW;
        tempW2[temp2Count] = w_w / homo1DivW;
        tempT2[temp2Count] = t_w / homo1DivW;
        tempN2[temp2Count] = n_w / homo1DivW;
        tempC2[temp2Count] = c_w / homo1DivW;
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

__global__ void runtimeIndexingKernel(
    const bool *rtActive, ULLInt *rtIndex, ULLInt *d_rtCount, ULLInt numFs
) {
    ULLInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= numFs) return;

    if (rtActive[fIdx]) {
        ULLInt idx = atomicAdd(d_rtCount, 1);
        rtIndex[idx] = fIdx;
    }
}

// Depth map creation
__global__ void createDepthMapKernel(
    const ULLInt *rtIndex,
    const bool *rtActive, const float *rtSx, const float *rtSy, const float *rtSz, const float *rtSw,
    ULLInt faceCounter, ULLInt faceOffset,
    bool *bActive, float *bDepth, ULLInt *bFaceId,
    float *bBrX, float *bBrY, float *bBrZ,
    int bWidth, int bHeight, int tNumX, int tNumY, int tSizeX, int tSizeY
) {
    ULLInt tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    ULLInt fIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdx >= tNumX * tNumY || fIdx >= faceCounter) return;
    fIdx += faceOffset;

    ULLInt rtIdx = rtIndex[fIdx];

    if (!rtActive[rtIdx]) return;

    ULLInt idx0 = rtIdx * 3;
    ULLInt idx1 = rtIdx * 3 + 1;
    ULLInt idx2 = rtIdx * 3 + 2;

    float sw0 = rtSw[idx0];
    float sw1 = rtSw[idx1];
    float sw2 = rtSw[idx2];

    float bx0 = (rtSx[idx0] / sw0 + 1) * bWidth / 2;
    float bx1 = (rtSx[idx1] / sw1 + 1) * bWidth / 2;
    float bx2 = (rtSx[idx2] / sw2 + 1) * bWidth / 2;

    float by0 = (1 - rtSy[idx0] / sw0) * bHeight / 2;
    float by1 = (1 - rtSy[idx1] / sw1) * bHeight / 2;
    float by2 = (1 - rtSy[idx2] / sw2) * bHeight / 2;

    float bz0 = (rtSz[idx0] / sw0 + 1) / 2;
    float bz1 = (rtSz[idx1] / sw1 + 1) / 2;
    float bz2 = (rtSz[idx2] / sw2 + 1) / 2;

    // Buffer bounding box based on the tile
    int tX = tIdx % tNumX;
    int tY = tIdx / tNumX;

    int bufferMinX = tX * tSizeX;
    int bufferMaxX = bufferMinX + tSizeX;
    int bufferMinY = tY * tSizeY;
    int bufferMaxY = bufferMinY + tSizeY;

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
        int bIdx = x + y * bWidth;

        Vec3f bary = Vec3f::bary(
            Vec2f(x, y), Vec2f(bx0, by0), Vec2f(bx1, by1), Vec2f(bx2, by2)
        );
        // Ignore if out of bound
        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float depth = bary.x * bz0 + bary.y * bz1 + bary.z * bz2; 

        if (atomicMinFloat(&bDepth[bIdx], depth)) {
            bDepth[bIdx] = depth;
            bActive[bIdx] = true;
            bFaceId[bIdx] = rtIdx;

            bBrX[bIdx] = bary.x;
            bBrY[bIdx] = bary.y;
            bBrZ[bIdx] = bary.z;
        }
    }
}

__global__ void rasterizationKernel(
    const float *rtSw,
    const float *rtWx, const float *rtWy, const float *rtWz,
    const float *rtTu, const float *rtTv,
    const float *rtNx, const float *rtNy, const float *rtNz,
    const float *rtCr, const float *rtCg, const float *rtCb, const float *rtCa,

    const bool *bActive, const ULLInt *bFaceId,
    float *bBrx, float *bBry, float *bBrz, // Bary
    float *bWx, float *bWy, float *bWz, // World
    float *bTu, float *bTv, // Texture
    float *bNx, float *bNy, float *bNz, // Normal
    float *bCr, float *bCg, float *bCb, float *bCa, // Color
    int bWidth, int bHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bWidth * bHeight || !bActive[i]) return;

    ULLInt fIdx = bFaceId[i];

    ULLInt idx0 = fIdx * 3;
    ULLInt idx1 = fIdx * 3 + 1;
    ULLInt idx2 = fIdx * 3 + 2;

    // Get barycentric coordinates
    float alp = bBrx[i];
    float bet = bBry[i];
    float gam = bBrz[i];

    // Get homogenous 1/w
    float homo1divW = alp / rtSw[idx0] + bet / rtSw[idx1] + gam / rtSw[idx2];

    // Set world position
    float wx_sw = rtWx[idx0] / rtSw[idx0] * alp + rtWx[idx1] / rtSw[idx1] * bet + rtWx[idx2] / rtSw[idx2] * gam;
    float wy_sw = rtWy[idx0] / rtSw[idx0] * alp + rtWy[idx1] / rtSw[idx1] * bet + rtWy[idx2] / rtSw[idx2] * gam;
    float wz_sw = rtWz[idx0] / rtSw[idx0] * alp + rtWz[idx1] / rtSw[idx1] * bet + rtWz[idx2] / rtSw[idx2] * gam;

    bWx[i] = wx_sw / homo1divW;
    bWy[i] = wy_sw / homo1divW;
    bWz[i] = wz_sw / homo1divW;

    // Set texture
    float tu_sw = rtTu[idx0] / rtSw[idx0] * alp + rtTu[idx1] / rtSw[idx1] * bet + rtTu[idx2] / rtSw[idx2] * gam;
    float tv_sw = rtTv[idx0] / rtSw[idx0] * alp + rtTv[idx1] / rtSw[idx1] * bet + rtTv[idx2] / rtSw[idx2] * gam;

    bTu[i] = tu_sw / homo1divW;
    bTv[i] = tv_sw / homo1divW;

    // Set normal
    float nx_sw = rtNx[idx0] / rtSw[idx0] * alp + rtNx[idx1] / rtSw[idx1] * bet + rtNx[idx2] / rtSw[idx2] * gam;
    float ny_sw = rtNy[idx0] / rtSw[idx0] * alp + rtNy[idx1] / rtSw[idx1] * bet + rtNy[idx2] / rtSw[idx2] * gam;
    float nz_sw = rtNz[idx0] / rtSw[idx0] * alp + rtNz[idx1] / rtSw[idx1] * bet + rtNz[idx2] / rtSw[idx2] * gam;

    bNx[i] = nx_sw / homo1divW;
    bNy[i] = ny_sw / homo1divW;
    bNz[i] = nz_sw / homo1divW;
    float mag = sqrt( // Normalize the normal
        bNx[i] * bNx[i] +
        bNy[i] * bNy[i] +
        bNz[i] * bNz[i]
    );
    bNx[i] /= mag;
    bNy[i] /= mag;
    bNz[i] /= mag; 

    // Set color
    float cr_sw = rtCr[idx0] / rtSw[idx0] * alp + rtCr[idx1] / rtSw[idx1] * bet + rtCr[idx2] / rtSw[idx2] * gam;
    float cg_sw = rtCg[idx0] / rtSw[idx0] * alp + rtCg[idx1] / rtSw[idx1] * bet + rtCg[idx2] / rtSw[idx2] * gam;
    float cb_sw = rtCb[idx0] / rtSw[idx0] * alp + rtCb[idx1] / rtSw[idx1] * bet + rtCb[idx2] / rtSw[idx2] * gam;
    float ca_sw = rtCa[idx0] / rtSw[idx0] * alp + rtCa[idx1] / rtSw[idx1] * bet + rtCa[idx2] / rtSw[idx2] * gam;

    bCr[i] = cr_sw / homo1divW;
    bCg[i] = cg_sw / homo1divW;
    bCb[i] = cb_sw / homo1divW;
    bCa[i] = ca_sw / homo1divW;
}