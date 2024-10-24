#include <Mesh3D.cuh>

// Mesh object

Mesh::Mesh(
    std::vector<float> wx, std::vector<float> wy, std::vector<float> wz,
    std::vector<float> nx, std::vector<float> ny, std::vector<float> nz,
    std::vector<float> tu, std::vector<float> tv,
    std::vector<float> cr, std::vector<float> cg, std::vector<float> cb, std::vector<float> ca,
    std::vector<ULLInt> fw, std::vector<ULLInt> ft, std::vector<ULLInt> fn
) : wx(wx), wy(wy), wz(wz), nx(nx), ny(ny), nz(nz), tu(tu), tv(tv),
    cr(cr), cg(cg), cb(cb), ca(ca), fw(fw), ft(ft), fn(fn) {}

Mesh::Mesh() {}

Vec3f Mesh::w3f(ULLInt i) { return Vec3f(wx[i], wy[i], wz[i]); }
Vec3f Mesh::n3f(ULLInt i) { return Vec3f(nx[i], ny[i], nz[i]); }
Vec2f Mesh::t2f(ULLInt i) { return Vec2f(tu[i], tv[i]); }
Vec4f Mesh::c4f(ULLInt i) { return Vec4f(cr[i], cg[i], cb[i], ca[i]); }
Vec3ulli Mesh::fw3ulli(ULLInt i) { return Vec3ulli(fw[i]); }
Vec3ulli Mesh::ft3ulli(ULLInt i) { return Vec3ulli(ft[i]); }
Vec3ulli Mesh::fn3ulli(ULLInt i) { return Vec3ulli(fn[i]); }

// Mesh3D

Mesh3D::Mesh3D(ULLInt numWs, ULLInt numNs, ULLInt numTs, ULLInt numFs) {
    mallocVertices(numWs, numNs, numTs);
    mallocFaces(numFs);
}

// Vertices allocation
void Mesh3D::mallocVertices(ULLInt numWs, ULLInt numNs, ULLInt numTs) {
    world.malloc(numWs);
    normal.malloc(numNs);
    texture.malloc(numTs);
    color.malloc(numWs);
    screen.malloc(numWs);
}
void Mesh3D::freeVertices() {
    world.free();
    normal.free();
    texture.free();
    color.free();
    screen.free();
}
void Mesh3D::resizeVertices(ULLInt numWs, ULLInt numTs, ULLInt numNs) {
    freeVertices();
    mallocVertices(numWs, numNs, numTs);
}

// Faces allocation
void Mesh3D::mallocFaces(ULLInt numFs) {
    // 1 Face = 3 indices
    faces.malloc(numFs * 3);
}
void Mesh3D::freeFaces() {
    faces.free();
}
void Mesh3D::resizeFaces(ULLInt numFs) {
    freeFaces();
    mallocFaces(numFs);
}

// Free all memory
void Mesh3D::free() {
    freeVertices();
    freeFaces();
}

// Append mesh obj to device mesh

void Mesh3D::operator+=(Mesh &mesh) {
    // Stream for async memory copy
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ULLInt offsetV = world.size;
    ULLInt offsetT = texture.size;
    ULLInt offsetN = normal.size;

    Vecptr3f newWorld;
    Vecptr3f newNormal;
    Vecptr2f newTexture;
    Vecptr4f newColor;
    Vecptr4ulli newFaces;
    ULLInt worldSize = mesh.wx.size();
    ULLInt normalSize = mesh.nx.size();
    ULLInt textureSize = mesh.tu.size();
    ULLInt colorSize = mesh.cr.size();
    ULLInt faceSize = mesh.fw.size();
    newWorld.malloc(worldSize);
    newNormal.malloc(normalSize);
    newTexture.malloc(textureSize);
    newColor.malloc(colorSize);
    newFaces.malloc(faceSize);

    cudaMemcpyAsync(newWorld.x, mesh.wx.data(), worldSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newWorld.y, mesh.wy.data(), worldSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newWorld.z, mesh.wz.data(), worldSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newNormal.x, mesh.nx.data(), normalSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newNormal.y, mesh.ny.data(), normalSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newNormal.z, mesh.nz.data(), normalSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newTexture.x, mesh.tu.data(), textureSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newTexture.y, mesh.tv.data(), textureSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newColor.x, mesh.cr.data(), colorSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newColor.y, mesh.cg.data(), colorSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newColor.z, mesh.cb.data(), colorSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newColor.w, mesh.ca.data(), colorSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newFaces.v, mesh.fw.data(), faceSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newFaces.t, mesh.ft.data(), faceSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newFaces.n, mesh.fn.data(), faceSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);

    ULLInt gridSize = (faceSize + 255) / 256;
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFaces.v, offsetV, faceSize);
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFaces.t, offsetT, faceSize);
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFaces.n, offsetN, faceSize);

    world += newWorld;
    normal += newNormal;
    texture += newTexture;
    color += newColor;
    faces += newFaces;

    screen.free();
    screen.malloc(world.size);
}

// Kernel for incrementing face indices
__global__ void incrementFaceIdxKernel(ULLInt *f, ULLInt offset, ULLInt numFs) { // BETA
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFs) f[idx] += offset;
}