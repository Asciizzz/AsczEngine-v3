#include <Buffer3D.cuh>

Buffer3D::Buffer3D() {}

void Buffer3D::resize(int width, int height, int pixelSize) {
    this->width = width / pixelSize;
    this->height = height / pixelSize;
    size = width * height;
    blockNum = (size + blockSize - 1) / blockSize;

    free(); // Free the previous buffer

    cudaMalloc(&active, size * sizeof(bool));
    cudaMalloc(&depth, size * sizeof(float));
    cudaMalloc(&color, size * sizeof(Vec4f));

    cudaMalloc(&world, size * sizeof(Vec3f));
    cudaMalloc(&normal, size * sizeof(Vec3f));
    cudaMalloc(&texture, size * sizeof(Vec2f));
    cudaMalloc(&wMeshId, size * sizeof(UInt));
    cudaMalloc(&nMeshId, size * sizeof(UInt));
    cudaMalloc(&tMeshId, size * sizeof(UInt));

    cudaMalloc(&faceID, size * sizeof(ULLInt));
    cudaMalloc(&bary, size * sizeof(Vec3f));
}

void Buffer3D::free() {
    if (active) cudaFree(active);
    if (depth) cudaFree(depth);
    if (color) cudaFree(color);
    if (world) cudaFree(world);
    if (normal) cudaFree(normal);
    if (texture) cudaFree(texture);
    if (wMeshId) cudaFree(wMeshId);
    if (nMeshId) cudaFree(nMeshId);
    if (tMeshId) cudaFree(tMeshId);
    if (faceID) cudaFree(faceID);
    if (bary) cudaFree(bary);
}

void Buffer3D::clearBuffer() {
    clearBufferKernel<<<blockNum, blockSize>>>(
        active, depth, color,
        world, normal, texture,
        wMeshId, nMeshId, tMeshId,
        faceID, bary, size
    );
    cudaDeviceSynchronize();
}

// Kernel for clearing the buffer
__global__ void clearBufferKernel(
    bool *active, float *depth, Vec4f *color,
    Vec3f *world, Vec3f *normal, Vec2f *texture,
    UInt *wMeshId, UInt *nMeshId, UInt *tMeshId,
    ULLInt *faceID, Vec3f *bary, int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    active[i] = false; // Inactive
    depth[i] = 1; // Furthest depth
    color[i] = Vec4f(); // Black

    world[i] = Vec3f(); // Limbo
    normal[i] = Vec3f(); // Limbo
    texture[i] = Vec2f(); // Limbo
    wMeshId[i] = NULL; // No mesh
    nMeshId[i] = NULL; // No mesh
    tMeshId[i] = NULL; // No mesh

    faceID[i] = NULL; // No face
    bary[i] = Vec3f(); // Limbo
}

// Night sky
void Buffer3D::nightSky() {
    nightSkyKernel<<<blockNum, blockSize>>>(color, width, height);
    cudaDeviceSynchronize();
}

__global__ void nightSkyKernel(Vec4f *color, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int x = i % width;
    int y = i / width;

    float ratioX = float(x) / float(width);
    float ratioY = float(y) / float(height);

    color[i] = Vec4f(0, 0, 0, 255);
    color[i].x = 4 * (1 - ratioY);
    color[i].y = 10 * (1 - ratioX);
    color[i].z = 20 * (1 - ratioY);
}