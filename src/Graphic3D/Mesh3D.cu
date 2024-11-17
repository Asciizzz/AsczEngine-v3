#include <Mesh3D.cuh>
#include <Graphic3D.cuh> // For the Graphic.mesh object

// Mesh object

Mesh::Mesh(
    std::vector<float> wx, std::vector<float> wy, std::vector<float> wz,
    std::vector<float> tu, std::vector<float> tv,
    std::vector<float> nx, std::vector<float> ny, std::vector<float> nz,
    std::vector<float> cr, std::vector<float> cg, std::vector<float> cb, std::vector<float> ca,
    std::vector<ULLInt> fw, std::vector<ULLInt> ft, std::vector<ULLInt> fn
) : wx(wx), wy(wy), wz(wz), tu(tu), tv(tv), nx(nx), ny(ny), nz(nz), 
    cr(cr), cg(cg), cb(cb), ca(ca), fw(fw), ft(ft), fn(fn) {}

Mesh::Mesh() {}

void Mesh::push(Mesh &mesh) {
    for (ULLInt i = 0; i < mesh.wx.size(); i++) {
        wx.push_back(mesh.wx[i]);
        wy.push_back(mesh.wy[i]);
        wz.push_back(mesh.wz[i]);
    }
    for (ULLInt i = 0; i < mesh.tu.size(); i++) {
        tu.push_back(mesh.tu[i]);
        tv.push_back(mesh.tv[i]);
    }
    for (ULLInt i = 0; i < mesh.nx.size(); i++) {
        nx.push_back(mesh.nx[i]);
        ny.push_back(mesh.ny[i]);
        nz.push_back(mesh.nz[i]);
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
Vec2f Mesh::t2f(ULLInt i) { return Vec2f(tu[i], tv[i]); }
Vec3f Mesh::n3f(ULLInt i) { return Vec3f(nx[i], ny[i], nz[i]); }
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
    Vec3f_ptr &w = Graphic3D::instance().mesh.w;

    ULLInt start = w_range.x, end = w_range.y;

    ULLInt numWs = end - start;
    ULLInt gridSize = (numWs + 255) / 256;

    translateMeshKernel<<<gridSize, 256>>>(
        w.x + start, w.y + start, w.z + start,
        t.x, t.y, t.z, numWs
    );
    cudaDeviceSynchronize();
}
void Mesh::rotateRuntime(Vec3f origin, float r, short axis) {
    Vec3f_ptr &w = Graphic3D::instance().mesh.w;
    Vec3f_ptr &n = Graphic3D::instance().mesh.n;

    ULLInt startW = w_range.x, endW = w_range.y;
    ULLInt startN = n_range.x, endN = n_range.y;

    ULLInt numWs = endW - startW;
    ULLInt numNs = endN - startN;

    ULLInt num = max(numWs, numNs);
    ULLInt gridSize = (num + 255) / 256;    

    rotateMeshKernel<<<gridSize, 256>>>(
        w.x + startW, w.y + startW, w.z + startW, numWs,
        n.x + startN, n.y + startN, n.z + startN, numNs,
        origin.x, origin.y, origin.z, r, axis
    );
    cudaDeviceSynchronize();
}
void Mesh::scaleRuntime(Vec3f origin, Vec3f scl) {
    Vec3f_ptr &w = Graphic3D::instance().mesh.w;
    Vec3f_ptr &n = Graphic3D::instance().mesh.n;

    ULLInt startW = w_range.x, endW = w_range.y;
    ULLInt startN = n_range.x, endN = n_range.y;

    ULLInt numWs = endW - startW;
    ULLInt numNs = endN - startN;

    ULLInt num = max(numWs, numNs);
    ULLInt gridSize = (num + 255) / 256;

    scaleMeshKernel<<<gridSize, 256>>>(
        w.x + startW, w.y + startW, w.z + startW, numWs,
        n.x + startN, n.y + startN, n.z + startN, numNs,
        origin.x, origin.y, origin.z, scl.x, scl.y, scl.z
    );
    cudaDeviceSynchronize();
}

// Mesh3D

Mesh3D::Mesh3D(ULLInt numWs, ULLInt numTs, ULLInt numNs, ULLInt numFs) {
    mallocVertices(numWs, numTs, numNs);
    mallocFaces(numFs);
}

// Vertices allocation
void Mesh3D::mallocVertices(ULLInt numWs, ULLInt numTs, ULLInt numNs) {
    s.malloc(numWs);
    w.malloc(numWs);
    t.malloc(numTs);
    n.malloc(numNs);
    c.malloc(numWs);
}
void Mesh3D::freeVertices() {
    s.free();
    w.free();
    n.free();
    t.free();
    c.free();
}
void Mesh3D::resizeVertices(ULLInt numWs, ULLInt numTs, ULLInt numNs) {
    freeVertices();
    mallocVertices(numWs, numNs, numTs);
}

// Faces allocation
void Mesh3D::mallocFaces(ULLInt numFs) {
    // 1 Face = 3 indices
    fvtn.malloc(numFs * 3);
}
void Mesh3D::freeFaces() {
    fvtn.free();
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
    ULLInt offsetV = w.size;
    ULLInt offsetT = t.size;
    ULLInt offsetN = n.size;

    // Set the range of stuff
    mesh.w_range = {offsetV, offsetV + mesh.wx.size()};
    mesh.t_range = {offsetT, offsetT + mesh.tu.size()};
    mesh.n_range = {offsetN, offsetN + mesh.nx.size()};
    mesh.c_range = {offsetV, offsetV + mesh.cr.size()};

    Vec3f_ptr newW;
    Vec2f_ptr newT;
    Vec3f_ptr newN;
    Vec4f_ptr newC;
    Vec3ulli_ptr newFvtn;
    ULLInt wSize = mesh.wx.size();
    ULLInt tSize = mesh.tu.size();
    ULLInt nSize = mesh.nx.size();
    ULLInt cSize = mesh.cr.size();
    ULLInt fvtnSize = mesh.fw.size();
    newW.malloc(wSize);
    newT.malloc(tSize);
    newN.malloc(nSize);
    newC.malloc(cSize);
    newFvtn.malloc(fvtnSize);

    // Stream for async memory copy
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(newW.x, mesh.wx.data(), wSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newW.y, mesh.wy.data(), wSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newW.z, mesh.wz.data(), wSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newT.x, mesh.tu.data(), tSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newT.y, mesh.tv.data(), tSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newN.x, mesh.nx.data(), nSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newN.y, mesh.ny.data(), nSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newN.z, mesh.nz.data(), nSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newC.x, mesh.cr.data(), cSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newC.y, mesh.cg.data(), cSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newC.z, mesh.cb.data(), cSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newC.w, mesh.ca.data(), cSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newFvtn.v, mesh.fw.data(), fvtnSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newFvtn.t, mesh.ft.data(), fvtnSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newFvtn.n, mesh.fn.data(), fvtnSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);

    // Increment face indices
    ULLInt gridSize = (fvtnSize + 255) / 256;
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFvtn.v, offsetV, fvtnSize);
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFvtn.t, offsetT, fvtnSize);
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFvtn.n, offsetN, fvtnSize);

    w += newW;
    t += newT;
    n += newN;
    c += newC;
    fvtn += newFvtn;

    s.free();
    s.malloc(w.size);
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