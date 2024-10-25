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
        grphic.runtimeFaces, grphic.d_faceCounter
    );
    cudaDeviceSynchronize();

    cudaMemcpy(&grphic.faceCounter, grphic.d_faceCounter, sizeof(ULLInt), cudaMemcpyDeviceToHost);
}

void VertexShader::createDepthMap() {
    Graphic3D &grphic = Graphic3D::instance();
    Buffer3D &buffer = grphic.buffer;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    // 1 million elements per batch
    int chunkNum = (grphic.faceCounter + grphic.chunkSize - 1) / grphic.chunkSize;

    dim3 blockSize(8, 32);

    for (int i = 0; i < chunkNum; i++) {
        size_t currentChunkSize = (i == chunkNum - 1) ?
            grphic.faceCounter - i * grphic.chunkSize : grphic.chunkSize;
        size_t blockNumTile = (grphic.tileNum + blockSize.x - 1) / blockSize.x;
        size_t blockNumFace = (currentChunkSize + blockSize.y - 1) / blockSize.y;
        dim3 blockNum(blockNumTile, blockNumFace);

        createDepthMapKernel<<<blockNum, blockSize, 0, grphic.faceStreams[i]>>>(
            grphic.runtimeFaces + i * grphic.chunkSize, currentChunkSize,
            buffer.active, buffer.depth, buffer.faceID, buffer.bary,
            buffer.width, buffer.height, grphic.tileNumX, grphic.tileNumY,
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
        grphic.runtimeFaces, buffer.faceID,
        buffer.world, buffer.normal, buffer.texture, buffer.color,
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

    Vec3f t3 = t4.toVec3f(); // Convert to NDC [-1, 1]
    bool inside = (
        t3.x >= -1 && t3.x <= 1 &&
        t3.y >= -1 && t3.y <= 1 &&
        t3.z >= 0 && t3.z <= 1   
    );

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
    Face3D *runtimeFaces, ULLInt *faceCounter
) {
    ULLInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= numFs) return;

    ULLInt fw0 = faceWs[fIdx * 3];
    ULLInt fw1 = faceWs[fIdx * 3 + 1];
    ULLInt fw2 = faceWs[fIdx * 3 + 2];

    ULLInt ft0 = faceTs[fIdx * 3];
    ULLInt ft1 = faceTs[fIdx * 3 + 1];
    ULLInt ft2 = faceTs[fIdx * 3 + 2];

    ULLInt fn0 = faceNs[fIdx * 3];
    ULLInt fn1 = faceNs[fIdx * 3 + 1];
    ULLInt fn2 = faceNs[fIdx * 3 + 2];

    Vec4f p0(screenX[fw0], screenY[fw0], screenZ[fw0], screenW[fw0]);
    Vec4f p1(screenX[fw1], screenY[fw1], screenZ[fw1], screenW[fw1]);
    Vec4f p2(screenX[fw2], screenY[fw2], screenZ[fw2], screenW[fw2]);

    if (p0.w > 0 || p1.w > 0 || p2.w > 0) {
        ULLInt idx = atomicAdd(faceCounter, 1);

        runtimeFaces[idx].world[0] = Vec3f(worldX[fw0], worldY[fw0], worldZ[fw0]);
        runtimeFaces[idx].world[1] = Vec3f(worldX[fw1], worldY[fw1], worldZ[fw1]);
        runtimeFaces[idx].world[2] = Vec3f(worldX[fw2], worldY[fw2], worldZ[fw2]);

        runtimeFaces[idx].normal[0] = Vec3f(normalX[fn0], normalY[fn0], normalZ[fn0]);
        runtimeFaces[idx].normal[1] = Vec3f(normalX[fn1], normalY[fn1], normalZ[fn1]);
        runtimeFaces[idx].normal[2] = Vec3f(normalX[fn2], normalY[fn2], normalZ[fn2]);

        runtimeFaces[idx].texture[0] = Vec2f(textureX[ft0], textureY[ft0]);
        runtimeFaces[idx].texture[1] = Vec2f(textureX[ft1], textureY[ft1]);
        runtimeFaces[idx].texture[2] = Vec2f(textureX[ft2], textureY[ft2]);

        runtimeFaces[idx].color[0] = Vec4f(colorX[fw0], colorY[fw0], colorZ[fw0], colorW[fw0]);
        runtimeFaces[idx].color[1] = Vec4f(colorX[fw1], colorY[fw1], colorZ[fw1], colorW[fw1]);
        runtimeFaces[idx].color[2] = Vec4f(colorX[fw2], colorY[fw2], colorZ[fw2], colorW[fw2]);

        runtimeFaces[idx].screen[0] = p0;
        runtimeFaces[idx].screen[1] = p1;
        runtimeFaces[idx].screen[2] = p2;
    }
}

// Depth map creation
__global__ void createDepthMapKernel(
    Face3D *runtimeFaces, ULLInt faceCounter,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight,
    int tileNumX, int tileNumY, int tileWidth, int tileHeight
) {
    ULLInt tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    ULLInt fIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdx >= tileNumX * tileNumY || fIdx >= faceCounter) return;

    Vec4f p0 = runtimeFaces[fIdx].screen[0];
    Vec4f p1 = runtimeFaces[fIdx].screen[1];
    Vec4f p2 = runtimeFaces[fIdx].screen[2];

    // Entirely outside the frustum
    if (p0.w <= 0 && p1.w <= 0 && p2.w <= 0) return;

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

    // If bounding box is outside the tile area, return
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
        if (bIdx >= buffWidth * buffHeight) continue;

        Vec3f bary = Vec3f::bary(
            Vec2f(x + .5f, y + .5f),
            Vec2f(p0.x, p0.y),
            Vec2f(p1.x, p1.y),
            Vec2f(p2.x, p2.y)
        );

        if (bary.x < 0 || bary.y < 0 || bary.z < 0) continue;

        float zDepth = bary.x * p0.z + bary.y * p1.z + bary.z * p2.z;

        if (atomicMinFloat(&buffDepth[bIdx], zDepth)) {
            buffActive[bIdx] = true;
            buffDepth[bIdx] = zDepth;
            buffFaceId[bIdx] = fIdx;
            buffBary[bIdx] = bary;
        }
    }
}

__global__ void rasterizationKernel(
    Face3D *runtimeFaces, ULLInt *buffFaceId,
    Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture, Vec4f *buffColor,
    bool *buffActive, Vec3f *buffBary, int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    ULLInt fIdx = buffFaceId[i];

    // Set vertex, texture, and normal indices

    // Get barycentric coordinates
    float alp = buffBary[i].x;
    float bet = buffBary[i].y;
    float gam = buffBary[i].z;

    // Set color
    Vec4f c0 = runtimeFaces[fIdx].color[0];
    Vec4f c1 = runtimeFaces[fIdx].color[1];
    Vec4f c2 = runtimeFaces[fIdx].color[2];
    buffColor[i] = c0 * alp + c1 * bet + c2 * gam;

    // Set world position
    Vec3f w0 = runtimeFaces[fIdx].world[0];
    Vec3f w1 = runtimeFaces[fIdx].world[1];
    Vec3f w2 = runtimeFaces[fIdx].world[2];
    buffWorld[i] = w0 * alp + w1 * bet + w2 * gam;

    // Set normal
    Vec3f n0 = runtimeFaces[fIdx].normal[0];
    Vec3f n1 = runtimeFaces[fIdx].normal[1];
    Vec3f n2 = runtimeFaces[fIdx].normal[2];
    n0.norm(); n1.norm(); n2.norm();
    buffNormal[i] = n0 * alp + n1 * bet + n2 * gam;
    buffNormal[i].norm();

    // Set texture
    Vec2f t0 = runtimeFaces[fIdx].texture[0];
    Vec2f t1 = runtimeFaces[fIdx].texture[1];
    Vec2f t2 = runtimeFaces[fIdx].texture[2];
    buffTexture[i] = t0 * alp + t1 * bet + t2 * gam;
}