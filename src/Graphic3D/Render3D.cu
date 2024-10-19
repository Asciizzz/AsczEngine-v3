#include <Render3D.cuh>

void Render3D::setResolution(float w, float h) {
    res = {w, h};
    res_half = {w / 2, h / 2};
    camera.setResolution(w, h);
    buffer.resize(w, h, pixelSize);
}

void Render3D::allocateProjection() {
    cudaMalloc(&projection, mesh.numVs * sizeof(Vec4f));
}
void Render3D::freeProjection() {
    if (projection) cudaFree(projection);
}
void Render3D::resizeProjection() {
    freeProjection();
    allocateProjection();
}

// Pipeline

void Render3D::vertexProjection() {
    vertexProjectionKernel<<<mesh.blockNumVs, mesh.blockSize>>>(
        projection, mesh.world, camera, pixelSize, mesh.numVs
    );
    cudaDeviceSynchronize();
}

void Render3D::createDepthMap() {
    buffer.clearBuffer();

    // Currently very buggy
    for (int i = 0; i < 2; i++)
        createDepthMapKernel<<<mesh.blockNumFs, mesh.blockSize>>>(
            projection, mesh.faces, mesh.numFs,
            buffer.depth, buffer.faceID, buffer.bary, buffer.width, buffer.height
        );
    cudaDeviceSynchronize();
}

void Render3D::rasterization() {
    rasterizationKernel<<<buffer.blockCount, buffer.blockSize>>>(
        mesh.color, mesh.world, mesh.normal, mesh.texture, mesh.meshID, mesh.faces,
        buffer.color, buffer.world, buffer.normal, buffer.texture, buffer.meshID,
        buffer.faceID, buffer.bary, buffer.width, buffer.height
    );
    cudaDeviceSynchronize();
}

// Kernels

__global__ void vertexProjectionKernel(Vec4f *projection, Vec3f *world, Camera3D camera, int p_s, ULLInt numVs) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVs) return;

    Vec4f v4 = world[i].toVec4f();
    Vec4f t4 = camera.mvp * v4;
    Vec3f t3 = t4.toVec3f(); // Convert to NDC [-1, 1]

    // Convert to screen space
    t3.x = (t3.x + 1) * camera.res.x / p_s / 2;
    t3.y = (1 - t3.y) * camera.res.y / p_s / 2;

    Vec4f p = t3.toVec4f();
    p.w = camera.isInsideFrustum(world[i]);

    projection[i] = p;
}

__device__ bool atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old) > value;
}

__global__ void createDepthMapKernel(
    // Mesh data
    Vec4f *projection, Vec3uli *faces, ULLInt numFs,
    // Buffer data
    float *buffDepth, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli f = faces[i];
    Vec4f p0 = projection[f.x];
    Vec4f p1 = projection[f.y];
    Vec4f p2 = projection[f.z];

    if (p0.w <= 0 || p1.w <= 0 || p2.w <= 0) return;

    // Bounding box
    int minX = min(min(p0.x, p1.x), p2.x);
    int maxX = max(max(p0.x, p1.x), p2.x);
    int minY = min(min(p0.y, p1.y), p2.y);
    int maxY = max(max(p0.y, p1.y), p2.y);

    // Clip the bounding box
    minX = max(minX, 0);
    maxX = min(maxX, buffWidth - 1);
    minY = max(minY, 0);
    maxY = min(maxY, buffHeight - 1);

    for (int x = minX; x <= maxX; x++)
    for (int y = minY; y <= maxY; y++) {
        int bIdx = x + y * buffWidth;

        float alp = ((p1.y - p2.y) * (x - p2.x) + (p2.x - p1.x) * (y - p2.y)) /
                    ((p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y));
        float bet = ((p2.y - p0.y) * (x - p2.x) + (p0.x - p2.x) * (y - p2.y)) /
                    ((p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y));
        float gam = 1.0f - alp - bet;

        if (alp < 0 || bet < 0 || gam < 0) continue;

        float zDepth = alp * p0.z + bet * p1.z + gam * p2.z;

        if (atomicMinFloat(&buffDepth[bIdx], zDepth)) {
            buffDepth[bIdx] = zDepth;
            buffFaceId[bIdx] = i;
            buffBary[bIdx] = Vec3f(alp, bet, gam);
        }
    }
}

__global__ void rasterizationKernel(
    // Mesh data
    Vec4f *color, Vec3f *world, Vec3f *normal, Vec2f *texture, UInt *meshID, Vec3uli *faces,
    // Buffer data
    Vec4f *buffColor, Vec3f *buffWorld, Vec3f *buffNormal, Vec2f *buffTexture,
    UInt *buffMeshId, ULLInt *buffFaceId, Vec3f *buffBary, int buffWidth, int buffHeight
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffWidth * buffHeight) return;

    ULLInt fIdx = buffFaceId[i];

    // Set vertex index
    ULLInt vIdx0 = faces[fIdx].x;
    ULLInt vIdx1 = faces[fIdx].y;
    ULLInt vIdx2 = faces[fIdx].z;

    // Get barycentric coordinates
    float alp = buffBary[i].x;
    float bet = buffBary[i].y;
    float gam = buffBary[i].z;

    // Set color
    Vec4f c0 = color[vIdx0];
    Vec4f c1 = color[vIdx1];
    Vec4f c2 = color[vIdx2];
    buffColor[i] = c0 * alp + c1 * bet + c2 * gam;

    // Set world position
    Vec3f w0 = world[vIdx0];
    Vec3f w1 = world[vIdx1];
    Vec3f w2 = world[vIdx2];
    buffWorld[i] = w0 * alp + w1 * bet + w2 * gam;

    // Set normal
    Vec3f n0 = normal[vIdx0];
    Vec3f n1 = normal[vIdx1];
    Vec3f n2 = normal[vIdx2];
    n0.norm(); n1.norm(); n2.norm();
    buffNormal[i] = n0 * alp + n1 * bet + n2 * gam;

    // Set texture
    Vec2f t0 = texture[vIdx0];
    Vec2f t1 = texture[vIdx1];
    Vec2f t2 = texture[vIdx2];
    buffTexture[i] = t0 * alp + t1 * bet + t2 * gam;

    // Set mesh ID
    buffMeshId[i] = meshID[vIdx0];
}