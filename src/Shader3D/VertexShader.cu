#include <VertexShader.cuh>

// Render functions

Vec4f VertexShader::toScreenSpace(Camera3D &camera, Vec3f world, int buffWidth, int buffHeight) {
    Vec4f v4 = world.toVec4f();
    Vec4f t4 = camera.mvp * v4;
    Vec3f t3 = t4.toVec3f(); // Convert to NDC [-1, 1]
    Vec4f p = t3.toVec4f();
    p.w = camera.isInsideFrustum(world);

    return p;
}

// Render pipeline

void VertexShader::cameraProjection() {
    Graphic3D &grphic = Graphic3D::instance();
    Camera3D &camera = grphic.camera;
    Buffer3D &buffer = grphic.buffer;
    Mesh3D &mesh = grphic.mesh;

    cameraProjectionKernel<<<mesh.blockNumWs, mesh.blockSize>>>(
        mesh.screen, mesh.world, camera, buffer.width, buffer.height, mesh.numWs
    );
    cudaDeviceSynchronize();
}

void VertexShader::createRuntimeFaces() {
    Graphic3D &grphic = Graphic3D::instance();
    Mesh3D &mesh = grphic.mesh;

    cudaMemset(grphic.d_faceCounter, 0, sizeof(ULLInt));
    createRuntimeFacesKernel<<<mesh.blockNumFs, mesh.blockSize>>>(
        mesh.screen, mesh.world, mesh.normal, mesh.texture, mesh.color,
        mesh.faceWs, mesh.faceNs, mesh.faceTs, mesh.numFs,
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
    Vec4f *screen, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numWs
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numWs) return;

    Vec4f p = VertexShader::toScreenSpace(camera, world[i], buffWidth, buffHeight);
    screen[i] = p;
}

// Create runtime faces
__global__ void createRuntimeFacesKernel(
    Vec4f *screen, Vec3f *world, Vec3f *normal, Vec2f *texture, Vec4f *color,
    Vec3ulli *faceWs, Vec3ulli *faceNs, Vec3ulli *faceTs, ULLInt numFs,
    Face3D *runtimeFaces, ULLInt *faceCounter
) {
    ULLInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= numFs) return;

    Vec3ulli fw = faceWs[fIdx];
    Vec3ulli fn = faceNs[fIdx];
    Vec3ulli ft = faceTs[fIdx];

    Vec4f p0 = screen[fw.x];
    Vec4f p1 = screen[fw.y];
    Vec4f p2 = screen[fw.z];

    if (p0.w > 0 || p1.w > 0 || p2.w > 0) {
        ULLInt idx = atomicAdd(faceCounter, 1);

        runtimeFaces[idx] = {
            {world[fw.x], world[fw.y], world[fw.z]},
            {normal[fn.x], normal[fn.y], normal[fn.z]},
            {texture[ft.x], texture[ft.y], texture[ft.z]},
            {color[fw.x], color[fw.y], color[fw.z]},
            {screen[fw.x], screen[fw.y], screen[fw.z]}
        };
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