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

void Graphic3D::allocateEdges() {
    cudaMalloc(&edges, mesh.numFs * 3 * sizeof(Vec3x2uli));

    facesToEdgesKernel<<<mesh.blockNumFs, mesh.blockSize>>>(
        edges, mesh.faces, mesh.numFs
    );
    cudaDeviceSynchronize();
}
void Graphic3D::freeEdges() {
    if (edges) cudaFree(edges);
}
void Graphic3D::resizeEdges() {
    freeEdges();
    allocateEdges();
}

void Graphic3D::free() {
    mesh.free();
    buffer.free();
    freeProjection();
    freeEdges();
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

// Helpful kernels
__global__ void facesToEdgesKernel(
    Vec2uli *edges, Vec3x3uli *faces, ULLInt numFs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFs) return;

    Vec3uli v = faces[i].v;
    
    // Edge only contain world space indices
    edges[i * 3 + 0] = {v.x, v.y};
    edges[i * 3 + 1] = {v.y, v.z};
    edges[i * 3 + 2] = {v.z, v.x};
}