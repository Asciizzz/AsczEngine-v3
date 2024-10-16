#include <Buffer3D.cuh>

Buffer3D::Buffer3D(int width, int height) {
    cudaMalloc(&depth, width * height * sizeof(float));
    cudaMalloc(&normal, width * height * 3 * sizeof(float));
    cudaMalloc(&color, width * height * 3 * sizeof(float));
    cudaMalloc(&world, width * height * 3 * sizeof(float));
}