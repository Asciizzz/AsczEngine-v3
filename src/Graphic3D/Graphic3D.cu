#include <Graphic3D.cuh>

void Graphic3D::setResolution(float w, float h, float ps) {
    pixelSize = ps;
    res = {w, h};
    res_half = {w / 2, h / 2};
    camera.aspect = w / h;
    buffer.resize(w, h, pixelSize);
}

void Graphic3D::free() {
    mesh.free();
    buffer.free();
    freeProjection();
    freeEdges();

    freeShadow();
}

void Graphic3D::operator+=(Mesh3D &m) {
    mesh += m;
    m.free();
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

// BETA: Shadow Map
void Graphic3D::allocateShadow(int sw, int sh) {
    sWidth = sw;
    sHeight = sh;
    sSize = sw * sh;

    cudaMalloc(&shadowActive, sSize * sizeof(bool));
    cudaMalloc(&shadowDepth, sSize * sizeof(float));

    cudaMalloc(&lightProj, mesh.numWs * sizeof(Vec3f));
}

void Graphic3D::freeShadow() {
    if (shadowActive) cudaFree(shadowActive);
    if (shadowDepth) cudaFree(shadowDepth);
    if (lightProj) cudaFree(lightProj);
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