#include <Buffer3D.cuh>

Buffer3D::Buffer3D() {}
Buffer3D::~Buffer3D() { free(); }

void Buffer3D::resize(int width, int height, int pixelSize) {
    this->width = width / pixelSize;
    this->height = height / pixelSize;
    size = width * height;
    blockCount = (size + blockSize - 1) / blockSize;

    free(); // Free the previous buffer

    // For depth checking
    cudaMalloc(&depth, size * sizeof(float));
    // For lighting
    cudaMalloc(&color, size * sizeof(Vec4f));
    cudaMalloc(&world, size * sizeof(Vec3f));
    cudaMalloc(&normal, size * sizeof(Vec3f));
    // For texture mapping
    cudaMalloc(&texture, size * sizeof(Vec2f));
    cudaMalloc(&meshID, size * sizeof(UInt));
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
        depth, color, world, normal, texture, meshID, size
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
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    depth[i] = 1;
    color[i] = Vec4f(0, 0, 0, 0);
    world[i] = Vec3f(0, 0, 0);
    normal[i] = Vec3f(0, 0, 0);
    texture[i] = Vec2f(0, 0);
    meshID[i] = NULL;
}