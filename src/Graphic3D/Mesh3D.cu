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

void Mesh::push(Mesh &mesh) {
    for (ULLInt i = 0; i < mesh.wx.size(); i++) {
        wx.push_back(mesh.wx[i]);
        wy.push_back(mesh.wy[i]);
        wz.push_back(mesh.wz[i]);
    }
    for (ULLInt i = 0; i < mesh.nx.size(); i++) {
        nx.push_back(mesh.nx[i]);
        ny.push_back(mesh.ny[i]);
        nz.push_back(mesh.nz[i]);
    }
    for (ULLInt i = 0; i < mesh.tu.size(); i++) {
        tu.push_back(mesh.tu[i]);
        tv.push_back(mesh.tv[i]);
    }
    for (ULLInt i = 0; i < mesh.cr.size(); i++) {
        cr.push_back(mesh.cr[i]);
        cg.push_back(mesh.cg[i]);
        cb.push_back(mesh.cb[i]);
        ca.push_back(mesh.ca[i]);
    }

    // Increment face indices
    for (ULLInt i = 0; i < mesh.fw.size(); i++) {
        fw.push_back(mesh.fw[i] + wx.size());
        ft.push_back(mesh.ft[i] + tu.size());
        fn.push_back(mesh.fn[i] + nx.size());
    }
}

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

void Mesh::rotateIni(Vec3f origin, float r, short axis) {
    #pragma omp parallel for
    for (ULLInt i = 0; i < wx.size(); i++) {
        Vec3f p = Vec3f(wx[i], wy[i], wz[i]);

        switch (axis) {
            case 0: p.rotateX(origin, r); break;
            case 1: p.rotateY(origin, r); break;
            case 2: p.rotateZ(origin, r); break;
        }

        wx[i] = p.x; wy[i] = p.y; wz[i] = p.z;
    }

    #pragma omp parallel for
    for (ULLInt i = 0; i < nx.size(); i++) {
        Vec3f n = Vec3f(nx[i], ny[i], nz[i]);
        
        switch (axis) {
            case 0: n.rotateX(Vec3f(), r); break;
            case 1: n.rotateY(Vec3f(), r); break;
            case 2: n.rotateZ(Vec3f(), r); break;
        }

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

    translateMeshKernel<<<gridSize, 256>>>(
        world.x + start, world.y + start, world.z + start,
        t.x, t.y, t.z, numWs
    );
    cudaDeviceSynchronize();
}
void Mesh::rotateRuntime(Vec3f origin, float r, short axis) {
    Vec3f_ptr &world = Graphic3D::instance().mesh.world;
    Vec3f_ptr &normal = Graphic3D::instance().mesh.normal;

    ULLInt startW = w_range.x, endW = w_range.y;
    ULLInt startN = n_range.x, endN = n_range.y;

    ULLInt numWs = endW - startW;
    ULLInt numNs = endN - startN;

    ULLInt num = max(numWs, numNs);
    ULLInt gridSize = (num + 255) / 256;    

    rotateMeshKernel<<<gridSize, 256>>>(
        world.x + startW, world.y + startW, world.z + startW, numWs,
        normal.x + startN, normal.y + startN, normal.z + startN, numNs,
        origin.x, origin.y, origin.z, r, axis
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

    ULLInt num = max(numWs, numNs);
    ULLInt gridSize = (num + 255) / 256;

    scaleMeshKernel<<<gridSize, 256>>>(
        world.x + startW, world.y + startW, world.z + startW, numWs,
        normal.x + startN, normal.y + startN, normal.z + startN, numNs,
        origin.x, origin.y, origin.z, scl.x, scl.y, scl.z
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
void Mesh3D::push(std::vector<Mesh> &meshes) {
    for (Mesh &mesh : meshes) push(mesh);
}

// Kernel for incrementing face indices
__global__ void incrementFaceIdxKernel(ULLInt *f, ULLInt offset, ULLInt numFs) { // BETA
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFs) f[idx] += offset;
}

// Kernel for transforming vertices

__global__ void translateMeshKernel(
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

__global__ void rotateMeshKernel(
    float *wx, float *wy, float *wz, ULLInt numWs,
    float *nx, float *ny, float *nz, ULLInt numNs,
    float ox, float oy, float oz,
    float r, short axis
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numWs) {
        Vec3f p = Vec3f(wx[idx], wy[idx], wz[idx]);

        switch (axis) {
            case 0: p.rotateX(Vec3f(ox, oy, oz), r); break;
            case 1: p.rotateY(Vec3f(ox, oy, oz), r); break;
            case 2: p.rotateZ(Vec3f(ox, oy, oz), r); break;
        }

        wx[idx] = p.x;
        wy[idx] = p.y;
        wz[idx] = p.z;
    }

    if (idx < numNs) {
        Vec3f n = Vec3f(nx[idx], ny[idx], nz[idx]);

        switch (axis) {
            case 0: n.rotateX(Vec3f(), r); break;
            case 1: n.rotateY(Vec3f(), r); break;
            case 2: n.rotateZ(Vec3f(), r); break;
        }

        n /= n.mag(); // Normalize

        nx[idx] = n.x;
        ny[idx] = n.y;
        nz[idx] = n.z;
    }
}

__global__ void scaleMeshKernel(
    float *wx, float *wy, float *wz, ULLInt numWs,
    float *nx, float *ny, float *nz, ULLInt numNs,
    float ox, float oy, float oz,
    float sx, float sy, float sz
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numWs) {
        wx[idx] = (wx[idx] - ox) * sx + ox;
        wy[idx] = (wy[idx] - oy) * sy + oy;
        wz[idx] = (wz[idx] - oz) * sz + oz;
    }

    if (idx < numNs) {
        nx[idx] *= sx;
        ny[idx] *= sy;
        nz[idx] *= sz;
        // Normalize
        float mag = sqrt(
            nx[idx] * nx[idx] +
            ny[idx] * ny[idx] +
            nz[idx] * nz[idx]
        );
        nx[idx] /= mag;
        ny[idx] /= mag;
        nz[idx] /= mag;
    }
}