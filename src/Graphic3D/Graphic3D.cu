#include <Graphic3D.cuh>

#include <SFML/Graphics.hpp>
// Face3D

void Face3D::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&sx, sizeof(float) * size);
    cudaMalloc(&sy, sizeof(float) * size);
    cudaMalloc(&sz, sizeof(float) * size);
    cudaMalloc(&sw, sizeof(float) * size);
    cudaMalloc(&wx, sizeof(float) * size);
    cudaMalloc(&wy, sizeof(float) * size);
    cudaMalloc(&wz, sizeof(float) * size);
    cudaMalloc(&tu, sizeof(float) * size);
    cudaMalloc(&tv, sizeof(float) * size);
    cudaMalloc(&nx, sizeof(float) * size);
    cudaMalloc(&ny, sizeof(float) * size);
    cudaMalloc(&nz, sizeof(float) * size);
    cudaMalloc(&active, sizeof(bool) * size / 3);
    cudaMalloc(&mat, sizeof(LLInt) * size / 3);
    cudaMalloc(&area, sizeof(float) * size / 3);
}
void Face3D::free() {
    size = 0;

    if (sx) cudaFree(sx); if (sy) cudaFree(sy); if (sz) cudaFree(sz); if (sw) cudaFree(sw);
    if (wx) cudaFree(wx); if (wy) cudaFree(wy); if (wz) cudaFree(wz);
    if (tu) cudaFree(tu); if (tv) cudaFree(tv);
    if (nx) cudaFree(nx); if (ny) cudaFree(ny); if (nz) cudaFree(nz);

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
    cudaMalloc(&rtIndex1, sizeof(ULLInt) * rtFaces.size / 3);
    cudaMalloc(&rtIndex2, sizeof(ULLInt) * rtFaces.size / 3);
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