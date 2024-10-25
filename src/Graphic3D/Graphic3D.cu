#include <Graphic3D.cuh>

#include <SFML/Graphics.hpp>
// Face3D

void Face3D::malloc(ULLInt size) {
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
}
void Face3D::free() {
    if (wx) cudaFree(wx); if (wy) cudaFree(wy); if (wz) cudaFree(wz);
    if (nx) cudaFree(nx); if (ny) cudaFree(ny); if (nz) cudaFree(nz);
    if (tu) cudaFree(tu); if (tv) cudaFree(tv);
    if (cr) cudaFree(cr); if (cg) cudaFree(cg); if (cb) cudaFree(cb); if (ca) cudaFree(ca);
    if (sx) cudaFree(sx); if (sy) cudaFree(sy); if (sz) cudaFree(sz); if (sw) cudaFree(sw);
}

// Graphic stuff below

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
    freeRuntimeFaces();
    freeFaceStreams();

    freeTexture();
}

// Graphic faces (runtime)
void Graphic3D::mallocRuntimeFaces() {
    cudaMalloc(&d_faceCounter, sizeof(ULLInt));
    // In the worst case scenario, each face when culled can be split into 4 faces
    rtFaces.malloc(mesh.faces.size * 4);
}
void Graphic3D::freeRuntimeFaces() {
    if (d_faceCounter) cudaFree(d_faceCounter);
    rtFaces.free();
}
void Graphic3D::resizeRuntimeFaces() {
    freeRuntimeFaces();
    mallocRuntimeFaces();
}

// Face stream for chunking very large number of faces
void Graphic3D::mallocFaceStreams() {
    chunkNum = (mesh.faces.size + chunkSize - 1) / chunkSize;

    // Stream for asynchronous execution (very helpful)
    faceStreams = (cudaStream_t*)malloc(chunkNum * sizeof(cudaStream_t));
    for (int i = 0; i < chunkNum; i++) {
        cudaStreamCreate(&faceStreams[i]);
    }
}
void Graphic3D::freeFaceStreams() {
    // for (int i = 0; i < chunkSize; i++) {
    //     if (faceStreams) cudaStreamDestroy(faceStreams[i]);
    // }
    // if (faceStreams) delete[] faceStreams;
}
void Graphic3D::resizeFaceStreams() {
    int newChunkNum = (mesh.faces.size + chunkSize - 1) / chunkSize;
    if (newChunkNum == chunkNum) return;

    freeFaceStreams();
    mallocFaceStreams();
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