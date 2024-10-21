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
    Graphic3D &graphic = Graphic3D::instance();
    Camera3D &camera = graphic.camera;
    Buffer3D &buffer = graphic.buffer;
    Mesh3D &mesh = graphic.mesh;

    cameraProjectionKernel<<<mesh.blockNumWs, mesh.blockSize>>>(
        mesh.screen, mesh.world, camera, buffer.width, buffer.height, mesh.numWs
    );
    cudaDeviceSynchronize();
}

void VertexShader::getVisibleFaces() {
    Graphic3D &graphic = Graphic3D::instance();
    Mesh3D &mesh = graphic.mesh;

    cudaMemset(mesh.numFsVisible, 0, sizeof(ULLInt));

    getVisibleFacesKernel<<<mesh.blockNumFs, mesh.blockSize>>>(
        mesh.screen, mesh.numWs,
        mesh.faces, mesh.numFs,
        mesh.facesVisible, mesh.numFsVisible
    );
    cudaDeviceSynchronize();
}

void VertexShader::createDepthMap() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;
    Mesh3D &mesh = graphic.mesh;

    buffer.clearBuffer();
    buffer.nightSky(); // Cool effect

    // Retrieve the visibleFace count from the device
    ULLInt visibleFaces;
    cudaMemcpy(&visibleFaces, mesh.numFsVisible, sizeof(ULLInt), cudaMemcpyDeviceToHost);

    dim3 blockSize(8, 32);
    ULLInt blockNumTile = (graphic.tileNum + blockSize.x - 1) / blockSize.x;
    ULLInt blockNumFace = (visibleFaces + blockSize.y - 1) / blockSize.y;
    dim3 blockNum(blockNumTile, blockNumFace);

    createDepthMapKernel<<<blockNum, blockSize>>>(
        mesh.screen, mesh.world, mesh.facesVisible, visibleFaces,
        buffer.active, buffer.depth, buffer.faceID, buffer.bary, buffer.width, buffer.height,
        graphic.tileNumX, graphic.tileNumY, graphic.tileWidth, graphic.tileHeight
    );
    cudaDeviceSynchronize();
}

void VertexShader::rasterization() {
    Graphic3D &graphic = Graphic3D::instance();
    Buffer3D &buffer = graphic.buffer;
    Mesh3D &mesh = graphic.mesh;

    rasterizationKernel<<<buffer.blockNum, buffer.blockSize>>>(
        mesh.world, buffer.world, mesh.wObjId, buffer.wObjId,
        mesh.normal, buffer.normal, mesh.nObjId, buffer.nObjId,
        mesh.texture, buffer.texture, mesh.tObjId, buffer.tObjId,
        mesh.color, buffer.color,
        mesh.faces, buffer.faceID, buffer.bary, buffer.bary,
        buffer.active, buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

// Kernels
__global__ void cameraProjectionKernel(
    Vec4f *screen, Vec3f *world, Camera3D camera, int buffWidth, int buffHeight, ULLInt numWs
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numWs) return;

    Vec4f p = VertexShader::toScreenSpace(camera, world[i], buffWidth, buffHeight);
    screen[i] = p;
}

// Find visible faces
__global__ void getVisibleFacesKernel(
    Vec4f *screen, ULLInt numWs,
    Vec3x3ulli *faces, ULLInt numFs,
    Vec3x3x1ulli *facesVisible, ULLInt *numFsVisible
) {
    ULLInt fIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (fIdx >= numFs) return;

    Vec3ulli fv = faces[fIdx].v;
    Vec4f p0 = screen[fv.x];
    Vec4f p1 = screen[fv.y];
    Vec4f p2 = screen[fv.z];

    // Entirely outside the frustum
    if (p0.w <= 0 && p1.w <= 0 && p2.w <= 0) return;

    // Set the face to be visible
    ULLInt idx = atomicAdd(numFsVisible, 1);
    facesVisible[idx].v = fv;
    facesVisible[idx].n = faces[fIdx].n;
    facesVisible[idx].t = faces[fIdx].t;
    facesVisible[idx].w = fIdx;
}

// Depth map creation
__global__ void createDepthMapKernel(
    Vec4f *screen, Vec3f *world, Vec3x3x1ulli *faces, ULLInt numFs,
    bool *buffActive, float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight,
    int tileNumX, int tileNumY, int tileWidth, int tileHeight
) {
    ULLInt tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    ULLInt fIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdx >= tileNumX * tileNumY || fIdx >= numFs) return;

    Vec3ulli fv = faces[fIdx].v;
    Vec4f p0 = screen[fv.x];
    Vec4f p1 = screen[fv.y];
    Vec4f p2 = screen[fv.z];

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
            buffFaceId[bIdx] = faces[fIdx].w;
            buffBary[bIdx] = bary;
        }
    }
}

__global__ void rasterizationKernel(
    Vec3f *world, Vec3f *buffWorld, UInt *wObjId, UInt *buffWObjId,
    Vec3f *normal, Vec3f *buffNormal, UInt *nObjId, UInt *buffNObjId,
    Vec2f *texture, Vec2f *buffTexture, UInt *tObjId, UInt *buffTObjId,
    Vec4f *color, Vec4f *buffColor,
    Vec3x3ulli *faces, ULLInt *buffFaceId, Vec3f *bary, Vec3f *buffBary,
    bool *buffActive, int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight || !buffActive[i]) return;

    ULLInt fIdx = buffFaceId[i];

    // Set vertex, texture, and normal indices
    Vec3ulli vIdx = faces[fIdx].v;
    Vec3ulli tIdx = faces[fIdx].t;
    Vec3ulli nIdx = faces[fIdx].n;

    // Get barycentric coordinates
    float alp = buffBary[i].x;
    float bet = buffBary[i].y;
    float gam = buffBary[i].z;

    // Set color
    Vec4f c0 = color[vIdx.x];
    Vec4f c1 = color[vIdx.y];
    Vec4f c2 = color[vIdx.z];
    buffColor[i] = c0 * alp + c1 * bet + c2 * gam;

    // Set world position
    Vec3f w0 = world[vIdx.x];
    Vec3f w1 = world[vIdx.y];
    Vec3f w2 = world[vIdx.z];
    buffWorld[i] = w0 * alp + w1 * bet + w2 * gam;

    // Set normal
    Vec3f n0 = normal[nIdx.x];
    Vec3f n1 = normal[nIdx.y];
    Vec3f n2 = normal[nIdx.z];
    n0.norm(); n1.norm(); n2.norm();
    buffNormal[i] = n0 * alp + n1 * bet + n2 * gam;
    buffNormal[i].norm();

    // Set texture
    Vec2f t0 = texture[tIdx.x];
    Vec2f t1 = texture[tIdx.y];
    Vec2f t2 = texture[tIdx.z];
    buffTexture[i] = t0 * alp + t1 * bet + t2 * gam;

    // Set obj Id
    buffWObjId[i] = wObjId[vIdx.x];
    buffNObjId[i] = nObjId[nIdx.x];
    buffTObjId[i] = tObjId[tIdx.x];
}