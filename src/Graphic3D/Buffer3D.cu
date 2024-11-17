#include <Buffer3D.cuh>

void Buffer3D::resize(int width, int height, int pixelSize) {
    this->width = width / pixelSize;
    this->height = height / pixelSize;
    size = width * height;
    blockNum = (size + blockSize - 1) / blockSize;

    free(); // Free the previous buffer

    cudaMalloc(&active, size * sizeof(bool));
    cudaMalloc(&depth, size * sizeof(float));
    cudaMalloc(&faceID, size * sizeof(ULLInt));
    cudaMalloc(&matID, size * sizeof(LLInt));
    bary.malloc(size);
    world.malloc(size);
    texture.malloc(size);
    normal.malloc(size);
    color.malloc(size);
}

void Buffer3D::free() {
    if (active) cudaFree(active);
    if (depth) cudaFree(depth);
    if (faceID) cudaFree(faceID);
    if (matID) cudaFree(matID);
    bary.free();
    world.free();
    texture.free();
    normal.free();
    color.free();
}

void Buffer3D::clearBuffer() {
    clearBufferKernel<<<blockNum, blockSize>>>(
        active, depth, faceID,
        bary.x, bary.y, bary.z,
        world.x, world.y, world.z,
        texture.x, texture.y,
        normal.x, normal.y, normal.z,
        color.x, color.y, color.z, color.w,
        size
    );
    cudaDeviceSynchronize();
}

__global__ void clearBufferKernel(
    bool *active, float *depth, ULLInt *faceID,
    float *brx, float *bry, float *brz, // Bary
    float *wx, float *wy, float *wz, // World
    float *tu, float *tv, // Texture
    float *nx, float *ny, float *nz, // Normal
    float *cr, float *cg, float *cb, float *ca, // Color
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    active[i] = false; // Inactive
    depth[i] = 1; // Furthest depth
    faceID[i] = NULL; // No face
    brx[i] = 0; bry[i] = 0; brz[i] = 0; // Limbo

    wx[i] = 0; wy[i] = 0; wz[i] = 0; // Limbo
    tu[i] = 0; tv[i] = 0; // Limbo
    nx[i] = 0; ny[i] = 0; nz[i] = 0; // Limbo
    cr[i] = 0; cg[i] = 0; cb[i] = 0; ca[i] = 0; // Limbo
}

// Night sky
void Buffer3D::defaultColor() {
    defaultColorKernel<<<blockNum, blockSize>>>(
        color.x, color.y, color.z, color.w, width, height
    );
    cudaDeviceSynchronize();
}

__global__ void defaultColorKernel(
    float *cr, float *cg, float *cb, float *ca, int width, int height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int x = i % width;
    int y = i / width;

    float ratioY = float(y) / float(height);

    // Sky up rgb(4 99 180) -> down rgb(4 135 227)
    cr[i] = 2;
    cg[i] = 10 + 16 * ratioY;
    cb[i] = 15 + 55 * ratioY;
    ca[i] = 255;
}