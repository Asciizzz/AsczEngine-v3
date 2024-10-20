#include <Graphic3D.cuh>

void Graphic3D::setResolution(float w, float h) {
    res = {w, h};
    res_half = {w / 2, h / 2};
    camera.setResolution(w, h);
    buffer.resize(w, h, pixelSize);
}

void Graphic3D::allocateProjection() {
    cudaMalloc(&projection, mesh.numWs * sizeof(Vec4f));
}
void Graphic3D::freeProjection() {
    if (projection) cudaFree(projection);
}
void Graphic3D::resizeProjection() {
    freeProjection();
    allocateProjection();
}

// Atomic functions
__device__ bool atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old) > value;
}