#include <Mesh3D.cuh>
#include <Graphic3D.cuh> // For the Graphic.mesh object

// ======================= Mesh object =======================

Mesh::Mesh() {}
Mesh::Mesh(
    VectF wx, VectF wy, VectF wz,
    VectF tu, VectF tv,
    VectF nx, VectF ny, VectF nz,
    VectULLI fw, VectLLI ft, VectLLI fn, VectLLI fm,
    VectF kdr, VectF kdg, VectF kdb
) : wx(wx), wy(wy), wz(wz),
    tu(tu), tv(tv),
    nx(nx), ny(ny), nz(nz),

    fw(fw), ft(ft), fn(fn), fm(fm),

    kdr(kdr), kdg(kdg), kdb(kdb)
{}

void Mesh::push(Mesh &mesh) {
    // Vertex data
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

    // Increment face indices
    for (ULLInt i = 0; i < mesh.fw.size(); i++) {
        fw.push_back(mesh.fw[i] + wx.size());

        if (mesh.ft[i] < 0) ft.push_back(-1);
        else ft.push_back(mesh.ft[i] + tu.size());

        if (mesh.fn[i] < 0) fn.push_back(-1);
        else fn.push_back(mesh.fn[i] + nx.size());

        if (mesh.fm[i] < 0) fm.push_back(-1);
        else fm.push_back(mesh.fm[i] + kdr.size());
    }

    // Material data
    for (ULLInt i = 0; i < mesh.kdr.size(); i++) {
        kdr.push_back(mesh.kdr[i]);
        kdg.push_back(mesh.kdg[i]);
        kdb.push_back(mesh.kdb[i]);
    }
}

Vec3f Mesh::w3f(ULLInt i) { return Vec3f(wx[i], wy[i], wz[i]); }
Vec2f Mesh::t2f(ULLInt i) { return Vec2f(tu[i], tv[i]); }
Vec3f Mesh::n3f(ULLInt i) { return Vec3f(nx[i], ny[i], nz[i]); }

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
    Vec3f_ptr &w = Graphic3D::instance().mesh.v.w;

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
    Vec3f_ptr &w = Graphic3D::instance().mesh.v.w;
    Vec3f_ptr &n = Graphic3D::instance().mesh.v.n;

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
    Vec3f_ptr &w = Graphic3D::instance().mesh.v.w;
    Vec3f_ptr &n = Graphic3D::instance().mesh.v.n;

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

// ======================= Vertex_ptr =======================


void Vertex_ptr::malloc(ULLInt size) {
    s.malloc(size);
    w.malloc(size);
    t.malloc(size);
    n.malloc(size);
}
void Vertex_ptr::free() {
    s.free();
    w.free();
    t.free();
    n.free();
}
void Vertex_ptr::operator+=(Vertex_ptr &vertex) {
    s += vertex.s;
    w += vertex.w;
    t += vertex.t;
    n += vertex.n;
}

// ======================= Face_ptr =======================


void Face_ptr::malloc(ULLInt size) {
    cudaMalloc(&v, size * sizeof(ULLInt));
    cudaMalloc(&t, size * sizeof(LLInt));
    cudaMalloc(&n, size * sizeof(LLInt));
    cudaMalloc(&m, size * sizeof(LLInt));
    this->size = size;
}
void Face_ptr::free() {
    if (v) cudaFree(v);
    if (t) cudaFree(t);
    if (n) cudaFree(n);
    if (m) cudaFree(m);
}
void Face_ptr::operator+=(Face_ptr &face) {
    ULLInt size = this->size + face.size;

    ULLInt *newV;
    LLInt *newT;
    LLInt *newN;
    LLInt *newM;
    cudaMalloc(&newV, size * sizeof(ULLInt));
    cudaMalloc(&newT, size * sizeof(LLInt));
    cudaMalloc(&newN, size * sizeof(LLInt));
    cudaMalloc(&newM, size * sizeof(LLInt));

    cudaMemcpy(newV, v, this->size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newT, t, this->size * sizeof(LLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newN, n, this->size * sizeof(LLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newM, m, this->size * sizeof(LLInt), cudaMemcpyDeviceToDevice);

    cudaMemcpy(newV + this->size, face.v, face.size * sizeof(ULLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newT + this->size, face.t, face.size * sizeof(LLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newN + this->size, face.n, face.size * sizeof(LLInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newM + this->size, face.m, face.size * sizeof(LLInt), cudaMemcpyDeviceToDevice);

    free();

    v = newV;
    t = newT;
    n = newN;
    m = newM;
    this->size = size;
}

// ======================= Material_ptr =======================


void Material_ptr::malloc(ULLInt size) {
    ka.malloc(size);
    kd.malloc(size);
    ks.malloc(size);
    this->size = size;
}
void Material_ptr::free() {
    ka.free();
    kd.free();
    ks.free();
}
void Material_ptr::operator+=(Material_ptr &material) {
    ka += material.ka;
    kd += material.kd;
    ks += material.ks;
    this->size += material.size;
}


// ======================= Mesh3D =======================

// Free
void Mesh3D::free() {
    v.free();
    f.free();
    m.free();
}

// Push
void Mesh3D::push(Mesh &mesh) {
    ULLInt offsetV = v.w.size;
    ULLInt offsetT = v.t.size;
    ULLInt offsetN = v.n.size;

    // Set the range of stuff
    mesh.w_range = {offsetV, offsetV + mesh.wx.size()};
    mesh.t_range = {offsetT, offsetT + mesh.tu.size()};
    mesh.n_range = {offsetN, offsetN + mesh.nx.size()};

    // Stream for async memory copy
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // =============== Vertex data ===============

    Vec3f_ptr newW;
    Vec2f_ptr newT;
    Vec3f_ptr newN;
    ULLInt wSize = mesh.wx.size();
    ULLInt tSize = mesh.tu.size();
    ULLInt nSize = mesh.nx.size();
    newW.malloc(wSize);
    newT.malloc(tSize);
    newN.malloc(nSize);

    cudaMemcpyAsync(newW.x, mesh.wx.data(), wSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newW.y, mesh.wy.data(), wSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newW.z, mesh.wz.data(), wSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newT.x, mesh.tu.data(), tSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newT.y, mesh.tv.data(), tSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newN.x, mesh.nx.data(), nSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newN.y, mesh.ny.data(), nSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newN.z, mesh.nz.data(), nSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    v.w += newW;
    v.t += newT;
    v.n += newN;
    v.s.free();
    v.s.malloc(v.w.size);

    // =============== Material data ===============

    ULLInt offsetM = m.size;

    Vec3f_ptr newKd;
    ULLInt kdSize = mesh.kdr.size();
    newKd.malloc(kdSize);

    cudaMemcpyAsync(newKd.x, mesh.kdr.data(), kdSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newKd.y, mesh.kdg.data(), kdSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newKd.z, mesh.kdb.data(), kdSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    m.kd += newKd;

    // =============== Face data ================

    Face_ptr newF;
    ULLInt fSize = mesh.fw.size();
    newF.malloc(fSize);

    cudaMemcpyAsync(newF.v, mesh.fw.data(), fSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newF.t, mesh.ft.data(), fSize * sizeof(LLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newF.n, mesh.fn.data(), fSize * sizeof(LLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newF.m, mesh.fm.data(), fSize * sizeof(LLInt), cudaMemcpyHostToDevice, stream);

    // Increment face indices
    ULLInt gridSize = (fSize + 255) / 256;
    incFaceIdxKernel1<<<gridSize, 256>>>(newF.v, offsetV, fSize);
    incFaceIdxKernel2<<<gridSize, 256>>>(newF.t, offsetT, fSize);
    incFaceIdxKernel2<<<gridSize, 256>>>(newF.n, offsetN, fSize);
    incFaceIdxKernel2<<<gridSize, 256>>>(newF.m, offsetM, fSize);
    f += newF;

    // Destroy the stream
    cudaStreamSynchronize(stream);
}
void Mesh3D::push(std::vector<Mesh> &meshes) {
    for (Mesh &mesh : meshes) push(mesh);
}

// Kernel for incrementing face indices
__global__ void incFaceIdxKernel1(ULLInt *f, ULLInt offset, ULLInt numFs) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFs) f[idx] += offset;
}
__global__ void incFaceIdxKernel2(LLInt *f, ULLInt offset, ULLInt numFs) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFs && f[idx] >= 0) f[idx] += offset;
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