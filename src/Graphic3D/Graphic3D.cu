#include <Graphic3D.cuh>

void Graphic3D::setResolution(float w, float h, float ps) {
    pixelSize = ps;
    res = {w, h};
    res_half = {w / 2, h / 2};
    camera.aspect = w / h;
    buffer.resize(w, h, pixelSize);
}

void Graphic3D::setTileSize(int tw, int th) {
    tileWidth = tw;
    tileHeight = th;

    // Buffer W/H must be divisible by tile W/H, otherwise throw an error
    // It's a bit forceful, but it's better to have a consistent tile size
    // Otherwise the entire tile-based rasterization will be broken
    // Trust me, I've been there
    if (buffer.width % tileWidth != 0 || buffer.height % tileHeight != 0) {
        std::cerr << "Buffer W/H must be divisible by tile W/H" << std::endl;
        exit(1);
    }

    tileNumX = buffer.width / tileWidth;
    tileNumY = buffer.height / tileHeight;
    tileNum = tileNumX * tileNumY;
}

// Free everything
void Graphic3D::free() {
    mesh.free();
    buffer.free();
    freeGFaces();
}

// Append Mesh3D
void Graphic3D::operator+=(Mesh3D &m) {
    mesh += m;
    m.free();
}

// Visible Faces
void Graphic3D::mallocGFaces() {
    cudaMalloc(&d_numVisibFs, sizeof(ULLInt));
    cudaMalloc(&visibFWs, sizeof(Vec4ulli) * mesh.numFs);

    cudaMalloc(&faceAreas, sizeof(float) * mesh.numFs);
}
void Graphic3D::freeGFaces() {
    if (d_numVisibFs) cudaFree(d_numVisibFs);
    if (visibFWs) cudaFree(visibFWs);

    if (faceAreas) cudaFree(faceAreas);
}
void Graphic3D::resizeGFaces() {
    freeGFaces();
    mallocGFaces();
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