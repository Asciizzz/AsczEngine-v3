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

    size_t gridSize = (mesh.faces.size / 3 + 255) / 256;

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

void VertexShader::frustumCulling() {
    Graphic3D &grphic = Graphic3D::instance();
    Camera3D &camera = grphic.camera;

    cudaMemset(grphic.d_cullCounter, 0, sizeof(ULLInt));

    size_t gridSize = (grphic.faceCounter + 255) / 256;

    frustumCullingKernel<<<gridSize, 256>>>(
        grphic.rtFaces.sx, grphic.rtFaces.sy, grphic.rtFaces.sz, grphic.rtFaces.sw,
        grphic.rtFaces.wx, grphic.rtFaces.wy, grphic.rtFaces.wz,
        grphic.rtFaces.tu, grphic.rtFaces.tv,
        grphic.rtFaces.nx, grphic.rtFaces.ny, grphic.rtFaces.nz,
        grphic.rtFaces.cr, grphic.rtFaces.cg, grphic.rtFaces.cb, grphic.rtFaces.ca,
        grphic.faceCounter,

        grphic.cullFaces.sx, grphic.cullFaces.sy, grphic.cullFaces.sz, grphic.cullFaces.sw,
        grphic.cullFaces.wx, grphic.cullFaces.wy, grphic.cullFaces.wz,
        grphic.cullFaces.tu, grphic.cullFaces.tv,
        grphic.cullFaces.nx, grphic.cullFaces.ny, grphic.cullFaces.nz,
        grphic.cullFaces.cr, grphic.cullFaces.cg, grphic.cullFaces.cb, grphic.cullFaces.ca,
        grphic.d_cullCounter,

        camera.plane
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&grphic.cullCounter, grphic.d_cullCounter, sizeof(ULLInt), cudaMemcpyDeviceToHost);
}

void VertexShader::createDepthMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    // Split the faces into chunks
    size_t chunkNum = (grphic.cullCounter + grphic.faceChunkSize - 1) 
                    /  grphic.faceChunkSize;

    dim3 blockSize(16, 32);
    for (size_t i = 0; i < chunkNum; i++) {
        size_t faceOffset = grphic.faceChunkSize * i;

        size_t curFaceCount = (i == chunkNum - 1) ?
            grphic.cullCounter - faceOffset : grphic.faceChunkSize;
        size_t blockNumTile = (grphic.tileNum + blockSize.x - 1) / blockSize.x;
        size_t blockNumFace = (curFaceCount + blockSize.y - 1) / blockSize.y;
        dim3 blockNum(blockNumTile, blockNumFace);

        createDepthMapKernel<<<blockNum, blockSize>>>(
            grphic.cullFaces.sx, grphic.cullFaces.sy,
            grphic.cullFaces.sz, grphic.cullFaces.sw,
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
        grphic.cullFaces.sw,
        grphic.cullFaces.wx, grphic.cullFaces.wy, grphic.cullFaces.wz,
        grphic.cullFaces.tu, grphic.cullFaces.tv,
        grphic.cullFaces.nx, grphic.cullFaces.ny, grphic.cullFaces.nz,
        grphic.cullFaces.cr, grphic.cullFaces.cg, grphic.cullFaces.cb, grphic.cullFaces.ca,

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

// Frustum culling
__global__ void frustumCullingKernel(
    const float *runtimeSx, const float *runtimeSy, const float *runtimeSz, const float *runtimeSw,
    const float *runtimeWx, const float *runtimeWy, const float *runtimeWz,
    const float *runtimeTu, const float *runtimeTv,
    const float *runtimeNx, const float *runtimeNy, const float *runtimeNz,
    const float *runtimeCr, const float *runtimeCg, const float *runtimeCb, const float *runtimeCa,
    ULLInt faceCounter,

    float *cullSx, float *cullSy, float *cullSz, float *cullSw,
    float *cullWx, float *cullWy, float *cullWz,
    float *cullTu, float *cullTv,
    float *cullNx, float *cullNy, float *cullNz,
    float *cullCr, float *cullCg, float *cullCb, float *cullCa,
    ULLInt *cullCounter,

    Plane3D plane
) {
    ULInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= faceCounter) return;

    ULInt fIdx0 = fIdx * 3;
    ULInt fIdx1 = fIdx * 3 + 1;
    ULInt fIdx2 = fIdx * 3 + 2;

    Vec4f rtSs[3] = {
        Vec4f(runtimeSx[fIdx0], runtimeSy[fIdx0], runtimeSz[fIdx0], runtimeSw[fIdx0]),
        Vec4f(runtimeSx[fIdx1], runtimeSy[fIdx1], runtimeSz[fIdx1], runtimeSw[fIdx1]),
        Vec4f(runtimeSx[fIdx2], runtimeSy[fIdx2], runtimeSz[fIdx2], runtimeSw[fIdx2])
    };
    Vec3f rtWs[3] = {
        Vec3f(runtimeWx[fIdx0], runtimeWy[fIdx0], runtimeWz[fIdx0]),
        Vec3f(runtimeWx[fIdx1], runtimeWy[fIdx1], runtimeWz[fIdx1]),
        Vec3f(runtimeWx[fIdx2], runtimeWy[fIdx2], runtimeWz[fIdx2])
    };
    Vec2f rtTs[3] = {
        Vec2f(runtimeTu[fIdx0], runtimeTv[fIdx0]),
        Vec2f(runtimeTu[fIdx1], runtimeTv[fIdx1]),
        Vec2f(runtimeTu[fIdx2], runtimeTv[fIdx2])
    };
    Vec3f rtNs[3] = {
        Vec3f(runtimeNx[fIdx0], runtimeNy[fIdx0], runtimeNz[fIdx0]),
        Vec3f(runtimeNx[fIdx1], runtimeNy[fIdx1], runtimeNz[fIdx1]),
        Vec3f(runtimeNx[fIdx2], runtimeNy[fIdx2], runtimeNz[fIdx2])
    };
    Vec4f rtCs[3] = {
        Vec4f(runtimeCr[fIdx0], runtimeCg[fIdx0], runtimeCb[fIdx0], runtimeCa[fIdx0]),
        Vec4f(runtimeCr[fIdx1], runtimeCg[fIdx1], runtimeCb[fIdx1], runtimeCa[fIdx1]),
        Vec4f(runtimeCr[fIdx2], runtimeCg[fIdx2], runtimeCb[fIdx2], runtimeCa[fIdx2])
    };

    // Everything will be interpolated
    Vec4f newSs[4];
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
    */

    for (int a = 0; a < 3; a++) {
        int b = (a + 1) % 3;

        // Find plane side
        float sideA = plane.equation(rtWs[a]);
        float sideB = plane.equation(rtWs[b]);

        if (sideA < 0 && sideB < 0) continue;

        if (sideA >= 0 && sideB >= 0) {
            newSs[newVcount] = rtSs[a];
            newWs[newVcount] = rtWs[a];
            newTs[newVcount] = rtTs[a];
            newNs[newVcount] = rtNs[a];
            newCs[newVcount] = rtCs[a];
            newVcount++;
            continue;
        }

        // Find intersection
        float tFact = -sideA / (sideB - sideA);

        Vec4f s = rtSs[a] + (rtSs[b] - rtSs[a]) * tFact;
        Vec3f w = rtWs[a] + (rtWs[b] - rtWs[a]) * tFact;
        Vec2f t = rtTs[a] + (rtTs[b] - rtTs[a]) * tFact;
        Vec3f n = rtNs[a] + (rtNs[b] - rtNs[a]) * tFact;
        Vec4f c = rtCs[a] + (rtCs[b] - rtCs[a]) * tFact;

        if (sideA > 0) {
            newSs[newVcount] = rtSs[a];
            newWs[newVcount] = rtWs[a];
            newTs[newVcount] = rtTs[a];
            newNs[newVcount] = rtNs[a];
            newCs[newVcount] = rtCs[a];
            newVcount++;
            newSs[newVcount] = s;
            newWs[newVcount] = w;
            newTs[newVcount] = t;
            newNs[newVcount] = n;
            newCs[newVcount] = c;
            newVcount++;
        } else {
            newSs[newVcount] = s;
            newWs[newVcount] = w;
            newTs[newVcount] = t;
            newNs[newVcount] = n;
            newCs[newVcount] = c;
            newVcount++;
        }
    }

    // If 4 point: create 2 faces A B C, A C D
    for (int i = 0; i < newVcount - 2; i++) {
        ULInt idx0 = atomicAdd(cullCounter, 1) * 3;
        ULInt idx1 = idx0 + 1;
        ULInt idx2 = idx0 + 2;

        cullSx[idx0] = newSs[0].x; cullSx[idx1] = newSs[i + 1].x; cullSx[idx2] = newSs[i + 2].x;
        cullSy[idx0] = newSs[0].y; cullSy[idx1] = newSs[i + 1].y; cullSy[idx2] = newSs[i + 2].y;
        cullSz[idx0] = newSs[0].z; cullSz[idx1] = newSs[i + 1].z; cullSz[idx2] = newSs[i + 2].z;
        cullSw[idx0] = newSs[0].w; cullSw[idx1] = newSs[i + 1].w; cullSw[idx2] = newSs[i + 2].w;

        cullWx[idx0] = newWs[0].x; cullWx[idx1] = newWs[i + 1].x; cullWx[idx2] = newWs[i + 2].x;
        cullWy[idx0] = newWs[0].y; cullWy[idx1] = newWs[i + 1].y; cullWy[idx2] = newWs[i + 2].y;
        cullWz[idx0] = newWs[0].z; cullWz[idx1] = newWs[i + 1].z; cullWz[idx2] = newWs[i + 2].z;

        cullTu[idx0] = newTs[0].x; cullTu[idx1] = newTs[i + 1].x; cullTu[idx2] = newTs[i + 2].x;
        cullTv[idx0] = newTs[0].y; cullTv[idx1] = newTs[i + 1].y; cullTv[idx2] = newTs[i + 2].y;

        cullNx[idx0] = newNs[0].x; cullNx[idx1] = newNs[i + 1].x; cullNx[idx2] = newNs[i + 2].x;
        cullNy[idx0] = newNs[0].y; cullNy[idx1] = newNs[i + 1].y; cullNy[idx2] = newNs[i + 2].y;
        cullNz[idx0] = newNs[0].z; cullNz[idx1] = newNs[i + 1].z; cullNz[idx2] = newNs[i + 2].z;

        cullCr[idx0] = newCs[0].x; cullCr[idx1] = newCs[i + 1].x; cullCr[idx2] = newCs[i + 2].x;
        cullCg[idx0] = newCs[0].y; cullCg[idx1] = newCs[i + 1].y; cullCg[idx2] = newCs[i + 2].y;
        cullCb[idx0] = newCs[0].z; cullCb[idx1] = newCs[i + 1].z; cullCb[idx2] = newCs[i + 2].z;
        cullCa[idx0] = newCs[0].w; cullCa[idx1] = newCs[i + 1].w; cullCa[idx2] = newCs[i + 2].w;
    }

    return;

    // For the time being we just gonna copy the face
    ULInt idx0 = atomicAdd(cullCounter, 1) * 3;
    ULInt idx1 = idx0 + 1;
    ULInt idx2 = idx0 + 2;

    cullSx[idx0] = runtimeSx[fIdx0]; cullSx[idx1] = runtimeSx[fIdx1]; cullSx[idx2] = runtimeSx[fIdx2];
    cullSy[idx0] = runtimeSy[fIdx0]; cullSy[idx1] = runtimeSy[fIdx1]; cullSy[idx2] = runtimeSy[fIdx2];
    cullSz[idx0] = runtimeSz[fIdx0]; cullSz[idx1] = runtimeSz[fIdx1]; cullSz[idx2] = runtimeSz[fIdx2];
    cullSw[idx0] = runtimeSw[fIdx0]; cullSw[idx1] = runtimeSw[fIdx1]; cullSw[idx2] = runtimeSw[fIdx2];

    cullWx[idx0] = runtimeWx[fIdx0]; cullWx[idx1] = runtimeWx[fIdx1]; cullWx[idx2] = runtimeWx[fIdx2];
    cullWy[idx0] = runtimeWy[fIdx0]; cullWy[idx1] = runtimeWy[fIdx1]; cullWy[idx2] = runtimeWy[fIdx2];
    cullWz[idx0] = runtimeWz[fIdx0]; cullWz[idx1] = runtimeWz[fIdx1]; cullWz[idx2] = runtimeWz[fIdx2];

    cullTu[idx0] = runtimeTu[fIdx0]; cullTu[idx1] = runtimeTu[fIdx1]; cullTu[idx2] = runtimeTu[fIdx2];
    cullTv[idx0] = runtimeTv[fIdx0]; cullTv[idx1] = runtimeTv[fIdx1]; cullTv[idx2] = runtimeTv[fIdx2];

    cullNx[idx0] = runtimeNx[fIdx0]; cullNx[idx1] = runtimeNx[fIdx1]; cullNx[idx2] = runtimeNx[fIdx2];
    cullNy[idx0] = runtimeNy[fIdx0]; cullNy[idx1] = runtimeNy[fIdx1]; cullNy[idx2] = runtimeNy[fIdx2];
    cullNz[idx0] = runtimeNz[fIdx0]; cullNz[idx1] = runtimeNz[fIdx1]; cullNz[idx2] = runtimeNz[fIdx2];

    cullCr[idx0] = runtimeCr[fIdx0]; cullCr[idx1] = runtimeCr[fIdx1]; cullCr[idx2] = runtimeCr[fIdx2];
    cullCg[idx0] = runtimeCg[fIdx0]; cullCg[idx1] = runtimeCg[fIdx1]; cullCg[idx2] = runtimeCg[fIdx2];
    cullCb[idx0] = runtimeCb[fIdx0]; cullCb[idx1] = runtimeCb[fIdx1]; cullCb[idx2] = runtimeCb[fIdx2];
    cullCa[idx0] = runtimeCa[fIdx0]; cullCa[idx1] = runtimeCa[fIdx1]; cullCa[idx2] = runtimeCa[fIdx2];
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