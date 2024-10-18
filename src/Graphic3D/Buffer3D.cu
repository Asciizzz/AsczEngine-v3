#include <Buffer3D.cuh>

Buffer3D::Buffer3D() {}

void Buffer3D::resize(int width, int height, int pixelSize) {
    buffWidth = width / pixelSize;
    buffHeight = height / pixelSize;
    buffSize = buffWidth * buffHeight;

    free();

    cudaMalloc(&depth, buffSize * sizeof(float));
    cudaMalloc(&color, buffSize * sizeof(Vec3f));
    cudaMalloc(&normal, buffSize * sizeof(Vec3f));
    cudaMalloc(&world, buffSize * sizeof(Vec3f));
    cudaMalloc(&tex, buffSize * sizeof(Vec2f));
}

void Buffer3D::free() {
    if (depth) cudaFree(depth);
    if (color) cudaFree(color);
    if (normal) cudaFree(normal);
    if (world) cudaFree(world);
    if (tex) cudaFree(tex);
}