#include <Mesh3D.cuh>
#include <Graphic3D.cuh> // For the Graphic.mesh object

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

void Mesh::translateIni(Vec3f t) {
    #pragma omp parallel for
    for (ULLInt i = 0; i < wx.size(); i++) {
        wx[i] += t.x; wy[i] += t.y; wz[i] += t.z;
    }
}
void Mesh::rotateIni(Vec3f origin, Vec3f rot, bool rotNormal) {
    #pragma omp parallel for
    for (ULLInt i = 0; i < wx.size(); i++) {
        Vec3f p = Vec3f(wx[i], wy[i], wz[i]);
        p.rotate(origin, rot);

        wx[i] = p.x; wy[i] = p.y; wz[i] = p.z;
    }

    if (!rotNormal) return;
    #pragma omp parallel for
    for (ULLInt i = 0; i < nx.size(); i++) {
        Vec3f n = Vec3f(nx[i], ny[i], nz[i]);
        n.rotate(Vec3f(), rot);

        n /= n.mag(); // Normalize

        nx[i] = n.x; ny[i] = n.y; nz[i] = n.z;
    }
}
void Mesh::scaleIni(Vec3f origin, Vec3f scl, bool sclNormal) {
    #pragma omp parallel for
    for (ULLInt i = 0; i < wx.size(); i++) {
        Vec3f p = Vec3f(wx[i], wy[i], wz[i]);
        p.scale(origin, scl);

        wx[i] = p.x; wy[i] = p.y; wz[i] = p.z;
    }

    if (!sclNormal) return;
    #pragma omp parallel for
    for (ULLInt i = 0; i < nx.size(); i++) {
        Vec3f n = Vec3f(nx[i], ny[i], nz[i]);
        n.scale(Vec3f(), scl);

        n /= n.mag(); // Normalize

        nx[i] = n.x; ny[i] = n.y; nz[i] = n.z;
    }
}

void Mesh::translateRuntime(Vec3f t) {
    Vec3f_ptr &world = Graphic3D::instance().mesh.world;

    ULLInt start = w_range.x, end = w_range.y;

    ULLInt numWs = end - start;
    ULLInt gridSize = (numWs + 255) / 256;

    translateKernel<<<gridSize, 256>>>(
        world.x + start, world.y + start, world.z + start,
        t.x, t.y, t.z, numWs
    );
    cudaDeviceSynchronize();
}
void Mesh::rotateRuntime(Vec3f origin, Vec3f rot) {
    Vec3f_ptr &world = Graphic3D::instance().mesh.world;
    Vec3f_ptr &normal = Graphic3D::instance().mesh.normal;

    ULLInt startW = w_range.x, endW = w_range.y;
    ULLInt startN = n_range.x, endN = n_range.y;

    ULLInt numWs = endW - startW;
    ULLInt numNs = endN - startN;

    ULLInt gridWsSize = (numWs + 255) / 256;
    ULLInt gridNsSize = (numNs + 255) / 256;

    rotateWsKernel<<<gridWsSize, 256>>>(
        world.x + startW, world.y + startW, world.z + startW,
        origin.x, origin.y, origin.z, rot.x, rot.y, rot.z, numWs
    );
    rotateNsKernel<<<gridNsSize, 256>>>(
        normal.x + startN, normal.y + startN, normal.z + startN,
        rot.x, rot.y, rot.z, numNs
    );
    cudaDeviceSynchronize();
}
void Mesh::scaleRuntime(Vec3f origin, Vec3f scl) {
    Vec3f_ptr &world = Graphic3D::instance().mesh.world;
    Vec3f_ptr &normal = Graphic3D::instance().mesh.normal;

    ULLInt startW = w_range.x, endW = w_range.y;
    ULLInt startN = n_range.x, endN = n_range.y;

    ULLInt numWs = endW - startW;
    ULLInt numNs = endN - startN;

    ULLInt gridWsSize = (numWs + 255) / 256;
    ULLInt gridNsSize = (numNs + 255) / 256;

    scaleWsKernel<<<gridWsSize, 256>>>(
        world.x + startW, world.y + startW, world.z + startW,
        origin.x, origin.y, origin.z, scl.x, scl.y, scl.z, numWs
    );
    scaleNsKernel<<<gridNsSize, 256>>>(
        normal.x + startN, normal.y + startN, normal.z + startN,
        scl.x, scl.y, scl.z, numNs
    );
    cudaDeviceSynchronize();
}

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

// Push
void Mesh3D::push(Mesh &mesh) {
    ULLInt offsetV = world.size;
    ULLInt offsetT = texture.size;
    ULLInt offsetN = normal.size;

    // Set the range of stuff
    mesh.w_range = {offsetV, offsetV + mesh.wx.size()};
    mesh.t_range = {offsetT, offsetT + mesh.tu.size()};
    mesh.n_range = {offsetN, offsetN + mesh.nx.size()};
    mesh.c_range = {offsetV, offsetV + mesh.cr.size()};

    mesh.fw_range = {faces.size, faces.size + mesh.fw.size()};
    mesh.ft_range = {faces.size, faces.size + mesh.ft.size()};
    mesh.fn_range = {faces.size, faces.size + mesh.fn.size()};

    Vec3f_ptr newWorld;
    Vec2f_ptr newTexture;
    Vec3f_ptr newNormal;
    Vec4f_ptr newColor;
    Vec4ulli_ptr newFaces;
    ULLInt worldSize = mesh.wx.size();
    ULLInt textureSize = mesh.tu.size();
    ULLInt normalSize = mesh.nx.size();
    ULLInt colorSize = mesh.cr.size();
    ULLInt faceSize = mesh.fw.size();
    newWorld.malloc(worldSize);
    newTexture.malloc(textureSize);
    newNormal.malloc(normalSize);
    newColor.malloc(colorSize);
    newFaces.malloc(faceSize);

    // Stream for async memory copy
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(newWorld.x, mesh.wx.data(), worldSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newWorld.y, mesh.wy.data(), worldSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newWorld.z, mesh.wz.data(), worldSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newTexture.x, mesh.tu.data(), textureSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newTexture.y, mesh.tv.data(), textureSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newNormal.x, mesh.nx.data(), normalSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newNormal.y, mesh.ny.data(), normalSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newNormal.z, mesh.nz.data(), normalSize * sizeof(float), cudaMemcpyHostToDevice, stream);

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

// Kernel for transforming vertices

__global__ void translateKernel(
    float *wx, float *wy, float *wz,
    float tx, float ty, float tz,
    ULLInt numWs
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWs) {
        wx[idx] += tx;
        wy[idx] += ty;
        wz[idx] += tz;
    }
}

__global__ void rotateWsKernel(
    float *wx, float *wy, float *wz,
    float ox, float oy, float oz,
    float rx, float ry, float rz, ULLInt numWs
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs) return;

    Vec3f w = {wx[idx], wy[idx], wz[idx]};
    w.rotate({ox, oy, oz}, {rx, ry, rz});
    wx[idx] = w.x; wy[idx] = w.y; wz[idx] = w.z;
}
__global__ void rotateNsKernel(
    float *nx, float *ny, float *nz,
    float rx, float ry, float rz, ULLInt numNs
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNs) return;

    Vec3f n = {nx[idx], ny[idx], nz[idx]};
    n.rotate({0, 0, 0}, {rx, ry, rz});
    n /= n.mag(); // Normalize
    nx[idx] = n.x; ny[idx] = n.y; nz[idx] = n.z;
}

__global__ void scaleWsKernel(
    float *wx, float *wy, float *wz,
    float ox, float oy, float oz,
    float sx, float sy, float sz, ULLInt numWs
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs) return;

    Vec3f w = {wx[idx], wy[idx], wz[idx]};
    w.scale({ox, oy, oz}, {sx, sy, sz});
    wx[idx] = w.x; wy[idx] = w.y; wz[idx] = w.z;
}

__global__ void scaleNsKernel(
    float *nx, float *ny, float *nz,
    float sx, float sy, float sz, ULLInt numNs
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNs) return;

    Vec3f n = {nx[idx], ny[idx], nz[idx]};
    n.scale({0, 0, 0}, {sx, sy, sz});
    n /= n.mag(); // Normalize
    nx[idx] = n.x; ny[idx] = n.y; nz[idx] = n.z;
}