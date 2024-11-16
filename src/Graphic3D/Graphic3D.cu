#include <Graphic3D.cuh>

#include <SFML/Graphics.hpp>
// Face3D

void Face3D::malloc(ULLInt size) {
    this->size = size;
    cudaMalloc(&wx, sizeof(float) * size);
    cudaMalloc(&wy, sizeof(float) * size);
    cudaMalloc(&wz, sizeof(float) * size);
    cudaMalloc(&nx, sizeof(float) * size);
    cudaMalloc(&ny, sizeof(float) * size);
    cudaMalloc(&nz, sizeof(float) * size);
    cudaMalloc(&tu, sizeof(float) * size);
    cudaMalloc(&tv, sizeof(float) * size);
    cudaMalloc(&cr, sizeof(float) * size);
    cudaMalloc(&cg, sizeof(float) * size);
    cudaMalloc(&cb, sizeof(float) * size);
    cudaMalloc(&ca, sizeof(float) * size);
    cudaMalloc(&sx, sizeof(float) * size);
    cudaMalloc(&sy, sizeof(float) * size);
    cudaMalloc(&sz, sizeof(float) * size);
    cudaMalloc(&sw, sizeof(float) * size);
    cudaMalloc(&area, sizeof(float) * size / 3);
    cudaMalloc(&active, sizeof(bool) * size / 3);
}
void Face3D::free() {
    size = 0;
    if (wx) cudaFree(wx); if (wy) cudaFree(wy); if (wz) cudaFree(wz);
    if (nx) cudaFree(nx); if (ny) cudaFree(ny); if (nz) cudaFree(nz);
    if (tu) cudaFree(tu); if (tv) cudaFree(tv);
    if (cr) cudaFree(cr); if (cg) cudaFree(cg); if (cb) cudaFree(cb); if (ca) cudaFree(ca);
    if (sx) cudaFree(sx); if (sy) cudaFree(sy); if (sz) cudaFree(sz); if (sw) cudaFree(sw);
    if (area) cudaFree(area);
    if (active) cudaFree(active);
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

    freeTexture();
    freeShadowMap();
}

// Graphic faces (runtime)
void Graphic3D::mallocRuntimeFaces() {
    rtFaces.malloc(mesh.faces.size * 4);

    cudaMalloc(&d_rtCount1, sizeof(ULLInt));
    cudaMalloc(&d_rtCount2, sizeof(ULLInt));
    cudaMalloc(&d_rtCount3, sizeof(ULLInt));
    cudaMalloc(&d_rtCount4, sizeof(ULLInt));
    cudaMalloc(&rtIndex2, sizeof(ULLInt) * rtFaces.size / 3);
    cudaMalloc(&rtIndex3, sizeof(ULLInt) * rtFaces.size / 3);
    cudaMalloc(&rtIndex4, sizeof(ULLInt) * rtFaces.size / 3);
    cudaMalloc(&rtIndex1, sizeof(ULLInt) * rtFaces.size / 3);
}
void Graphic3D::freeRuntimeFaces() {
    rtFaces.free();

    if (d_rtCount1) cudaFree(d_rtCount1);
    if (d_rtCount2) cudaFree(d_rtCount2);
    if (d_rtCount3) cudaFree(d_rtCount3);
    if (d_rtCount4) cudaFree(d_rtCount4);
    if (rtIndex1) cudaFree(rtIndex1);
    if (rtIndex2) cudaFree(rtIndex2);
    if (rtIndex3) cudaFree(rtIndex3);
    if (rtIndex4) cudaFree(rtIndex4);
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

// BETA: Texture mapping
void Graphic3D::createTexture(const std::string &path) {
    sf::Image textureImage;
    if (!textureImage.loadFromFile(path)) {
        throw std::runtime_error("Failed to load texture image.");
    }

    textureWidth = textureImage.getSize().x;
    textureHeight = textureImage.getSize().y;

    std::vector<Vec3f> texture(textureWidth * textureHeight);

    for (int y = 0; y < textureHeight; y++) {
        for (int x = 0; x < textureWidth; x++) {
            sf::Color color = textureImage.getPixel(x, y);
            int idx = x + y * textureWidth;
            texture[idx] = {float(color.r), float(color.g), float(color.b)};
        }
    }

    if (textureSet) freeTexture();
    else textureSet = true;

    cudaMalloc(&d_texture, sizeof(Vec3f) * texture.size());
    cudaMemcpy(d_texture, texture.data(), sizeof(Vec3f) * texture.size(), cudaMemcpyHostToDevice);
}

void Graphic3D::freeTexture() {
    if (d_texture) cudaFree(d_texture);
}

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