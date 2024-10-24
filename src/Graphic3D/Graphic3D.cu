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
    freeFaceStreams();
}

// Append Mesh3D
void Graphic3D::appendMesh(Mesh3D &m, bool del) {
    mesh += m;
    if (del) m.free();
}

// Graphic faces (runtime)
void Graphic3D::mallocGFaces() {
    cudaMalloc(&d_faceCounter, sizeof(ULLInt));
    // In the worst case scenario, each face when culled can be split into 4 faces
    cudaMalloc(&runtimeFaces, sizeof(Face3D) * mesh.numFs * 4);
}
void Graphic3D::freeGFaces() {
    if (d_faceCounter) cudaFree(d_faceCounter);
    if (runtimeFaces) cudaFree(runtimeFaces);
}
void Graphic3D::resizeGFaces() {
    freeGFaces();
    mallocGFaces();
}

// Face stream for chunking very large number of faces
void Graphic3D::mallocFaceStreams() {
    chunkNum = (mesh.numFs + chunkSize - 1) / chunkSize;

    // Stream for asynchronous execution (very helpful)
    faceStreams = (cudaStream_t*)malloc(chunkNum * sizeof(cudaStream_t));
    for (int i = 0; i < chunkNum; i++) {
        cudaStreamCreate(&faceStreams[i]);
    }
}
void Graphic3D::freeFaceStreams() {
    for (int i = 0; i < chunkSize; i++) {
        if (faceStreams) cudaStreamDestroy(faceStreams[i]);
    }
    if (faceStreams) delete[] faceStreams;
}
void Graphic3D::resizeFaceStreams() {
    int newChunkNum = (mesh.numFs + chunkSize - 1) / chunkSize;
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