#include <Graphic3D.cuh>

#include <SFML/Graphics.hpp>
// Face3D

void Face3D::malloc(ULLInt size) {
    this->size = size;
    count = size / 3;

    // Data x3
    s.malloc(size);
    w.malloc(size);
    t.malloc(size);
    n.malloc(size);

    // Data x1
    cudaMalloc(&active, sizeof(bool) * count);
    cudaMalloc(&mat, sizeof(LLInt) * count);
    cudaMalloc(&area, sizeof(float) * count);
}
void Face3D::free() {
    size = 0;

    s.free();
    w.free();
    t.free();

    if (active) cudaFree(active);
    if (mat) cudaFree(mat);
    if (area) cudaFree(area);
}

// Graphic stuff below

void Graphic3D::setResolution(float w, float h, float ps) {
    pixelSize = ps;
    res = {w, h};
    res_half = {w / 2, h / 2};
    camera.aspect = w / h;
    buffer.resize(w, h, pixelSize);
}

// Free everything
void Graphic3D::free() {
    mesh.free();
    buffer.free();
    freeRuntimeFaces();
    destroyRuntimeStreams();

    freeShadowMap();
}

// Graphic faces (runtime)
void Graphic3D::mallocRuntimeFaces() {
    rtFaces.malloc(mesh.f.size * 4);

    cudaMalloc(&d_rtCount1, sizeof(ULLInt));
    cudaMalloc(&d_rtCount2, sizeof(ULLInt));
    cudaMalloc(&rtIndex1, sizeof(ULLInt) * rtFaces.count);
    cudaMalloc(&rtIndex2, sizeof(ULLInt) * rtFaces.count);
}
void Graphic3D::freeRuntimeFaces() {
    rtFaces.free();

    if (d_rtCount1) cudaFree(d_rtCount1);
    if (d_rtCount2) cudaFree(d_rtCount2);
    if (rtIndex1) cudaFree(rtIndex1);
    if (rtIndex2) cudaFree(rtIndex2);
}
void Graphic3D::resizeRuntimeFaces() {
    freeRuntimeFaces();
    mallocRuntimeFaces();
}

void Graphic3D::createRuntimeStreams() {
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&rtStreams[i]);
    }
}
void Graphic3D::destroyRuntimeStreams() {
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(rtStreams[i]);
    }
}

// =========================================================================
// ============================= BETAs SECTION =============================
// =========================================================================

// Beta: Shadow mapping
void Graphic3D::createShadowMap(int w, int h, int tw, int th) {
    shdwWidth = w;
    shdwHeight = h;
    shdwTileSizeX = tw;
    shdwTileSizeY = th;
    shdwTileNumX = w / tw;
    shdwTileNumY = h / th;
    shdwTileNum = shdwTileNumX * shdwTileNumY;

    cudaMalloc(&shadowDepth, sizeof(float) * w * h);
}

void Graphic3D::freeShadowMap() {
    if (shadowDepth) cudaFree(shadowDepth);
}