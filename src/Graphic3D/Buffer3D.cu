#include <Buffer3D.cuh>

Buffer3D::Buffer3D() {}

void Buffer3D::resize(int width, int height, int pixelSize) {
    buffWidth = width / pixelSize;
    buffHeight = height / pixelSize;
    buffSize = buffWidth * buffHeight;
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