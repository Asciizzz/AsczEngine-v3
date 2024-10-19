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

void Render3D::rasterizeFaces() {
    buffer.clearBuffer();

    // Currently very buggy
    // for (int i = 0; i < 2; i++)
        rasterizeFacesKernel<<<mesh.blockNumFs, mesh.blockSize>>>(
            projection, mesh.color, mesh.faces, mesh.numFs,
            buffer.depth, buffer.color, buffer.width, buffer.height
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

__global__ void rasterizeFacesKernel(
    // Mesh data
    Vec4f *projection, Vec4f *color, Vec3uli *faces, ULLInt numFs,
    // Buffer data
    float *buffDepth, Vec4f *buffColor, int buffWidth, int buffHeight
) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli f = faces[i];
    Vec4f p0 = projection[f.x];
    Vec4f p1 = projection[f.y];
    Vec4f p2 = projection[f.z];

    Vec4f c0 = color[f.x];
    Vec4f c1 = color[f.y];
    Vec4f c2 = color[f.z];

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

        float alpha = ((p1.y - p2.y) * (x - p2.x) + (p2.x - p1.x) * (y - p2.y)) /
                      ((p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y));
        float beta = ((p2.y - p0.y) * (x - p2.x) + (p0.x - p2.x) * (y - p2.y)) /
                     ((p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y));
        float gamma = 1.0f - alpha - beta;

        if (alpha < 0 || beta < 0 || gamma < 0) continue;

        float zDepth = alpha * p0.z + beta * p1.z + gamma * p2.z;

        if (atomicMinFloat(&buffDepth[bIdx], zDepth)) {
            buffDepth[bIdx] = zDepth;
            buffColor[bIdx] = c0 * alpha + c1 * beta + c2 * gamma;
        }
    }
}