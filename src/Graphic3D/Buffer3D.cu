#include <Buffer3D.cuh>

Buffer3D::Buffer3D() {}

void Buffer3D::resize(int width, int height, int pixelSize) {
    buffWidth = width / pixelSize;
    buffHeight = height / pixelSize;
    buffSize = buffWidth * buffHeight;
    blockCount = (buffSize + blockSize - 1) / blockSize;

    free(); // Free the previous buffer

    // For depth checking
    cudaMalloc(&depth, buffSize * sizeof(float));
    // For lighting
    cudaMalloc(&color, buffSize * sizeof(Vec4f));
    cudaMalloc(&world, buffSize * sizeof(Vec3f));
    cudaMalloc(&normal, buffSize * sizeof(Vec3f));
    // For texture mapping
    cudaMalloc(&texture, buffSize * sizeof(Vec2f));
    cudaMalloc(&meshID, buffSize * sizeof(UInt));
}

void Buffer3D::free() {
    if (depth) cudaFree(depth);
    if (color) cudaFree(color);
    if (world) cudaFree(world);
    if (normal) cudaFree(normal);
    if (texture) cudaFree(texture);
    if (meshID) cudaFree(meshID);
}

void Buffer3D::clearBuffer() {
    clearBufferKernel<<<blockCount, blockSize>>>(
        depth, color, world, normal, texture, meshID, buffSize
    );
    cudaDeviceSynchronize();
}

// Kernel for clearing the buffer
__global__ void clearBufferKernel(
    float *depth,
    Vec4f *color,
    Vec3f *world,
    Vec3f *normal,
    Vec2f *texture,
    UInt *meshID,
    int buffSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffSize) return;

    depth[i] = INFINITY;
    color[i] = Vec4f(0, 0, 0, 0);
    world[i] = Vec3f(0, 0, 0);
    normal[i] = Vec3f(0, 0, 0);
    texture[i] = Vec2f(0, 0);
    meshID[i] = NULL;
}