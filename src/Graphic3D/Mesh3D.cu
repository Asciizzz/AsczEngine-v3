#include <Graphic3D.cuh> // For the Graphic.mesh object

#include <fstream> // For file I/O

// ======================= Mesh range =======================

void MeshRange::operator=(MeshRange &range) {
    w1 = range.w1; w2 = range.w2;
    t1 = range.t1; t2 = range.t2;
    n1 = range.n1; n2 = range.n2;
}

void MeshRange::offsetW(ULLInt offset) { w1 += offset; w2 += offset; }
void MeshRange::offsetT(ULLInt offset) { t1 += offset; t2 += offset; }
void MeshRange::offsetN(ULLInt offset) { n1 += offset; n2 += offset; }

// ======================= Mesh object =======================

Mesh::Mesh() {}
Mesh::Mesh(
    // Vertex data
    VectF wx, VectF wy, VectF wz,
    VectF tu, VectF tv,
    VectF nx, VectF ny, VectF nz,
    // Face data
    VectULLI fw, VectLLI ft, VectLLI fn, VectLLI fm,
    // Material data
    VectF kar, VectF kag, VectF kab,
    VectF kdr, VectF kdg, VectF kdb,
    VectF ksr, VectF ksg, VectF ksb,
    VectLLI mkd,
    // Texture data
    VectF txr, VectF txg, VectF txb,
    VectI txw, VectI txh, VectLLI txof,
    // Object data
    MeshRangeMap mrmap, VectStr mrmapKs
) : wx(wx), wy(wy), wz(wz),
    tu(tu), tv(tv),
    nx(nx), ny(ny), nz(nz),

    fw(fw), ft(ft), fn(fn), fm(fm),

    kar(kar), kag(kag), kab(kab),
    kdr(kdr), kdg(kdg), kdb(kdb),
    ksr(ksr), ksg(ksg), ksb(ksb),
    mkd(mkd),

    txr(txr), txg(txg), txb(txb),
    txw(txw), txh(txh), txof(txof),

    mrmapST(mrmap),
    mrmapRT(mrmap),
    mrmapKs(mrmapKs)
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

    // Material data
    for (ULLInt i = 0; i < mesh.kdr.size(); i++) {
        kar.push_back(mesh.kar[i]);
        kag.push_back(mesh.kag[i]);
        kab.push_back(mesh.kab[i]);

        kdr.push_back(mesh.kdr[i]);
        kdg.push_back(mesh.kdg[i]);
        kdb.push_back(mesh.kdb[i]);

        ksr.push_back(mesh.ksr[i]);
        ksg.push_back(mesh.ksg[i]);
        ksb.push_back(mesh.ksb[i]);
    }

    // Texture data + increment texture offset
    for (ULLInt i = 0; i < mesh.txw.size(); i++) {
        txw.push_back(mesh.txw[i]);
        txh.push_back(mesh.txh[i]);
        txof.push_back(mesh.txof[i] + txr.size());
    }
    for (ULLInt i = 0; i < mesh.txr.size(); i++) {
        txr.push_back(mesh.txr[i]);
        txg.push_back(mesh.txg[i]);
        txb.push_back(mesh.txb[i]);
    }

    // Increment face indices
    for (ULLInt i = 0; i < mesh.fw.size(); i++) {
        fw.push_back(mesh.fw[i] + wx.size());

        if (mesh.ft[i] < 0) ft.push_back(-1);
        else ft.push_back(mesh.ft[i] + tu.size());

        if (mesh.fn[i] < 0) fn.push_back(-1);
        else fn.push_back(mesh.fn[i] + nx.size());

        if (mesh.fm[i] < 0) fm.push_back(-1);
        else fm.push_back(mesh.fm[i] + kar.size());
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

void Mesh::translateRuntime(std::string mapkey, Vec3f t) {
    if (mrmapRT.find(mapkey) == mrmapRT.end()) return;
    if (!allocated) return;

    Vec3f_ptr &w = Graphic3D::instance().mesh.v.w;

    ULLInt start = mrmapRT[mapkey].w1;
    ULLInt end = mrmapRT[mapkey].w2;

    ULLInt numWs = end - start;
    ULLInt gridSize = (numWs + 255) / 256;

    translateMeshKernel<<<gridSize, 256>>>(
        w.x + start, w.y + start, w.z + start,
        t.x, t.y, t.z, numWs
    );
    cudaDeviceSynchronize();
}
void Mesh::rotateRuntime(std::string mapkey, Vec3f origin, float r, short axis) {
    if (mrmapRT.find(mapkey) == mrmapRT.end()) return;
    if (!allocated) return;

    Vec3f_ptr &w = Graphic3D::instance().mesh.v.w;
    Vec3f_ptr &n = Graphic3D::instance().mesh.v.n;

    ULLInt startW = mrmapRT[mapkey].w1;
    ULLInt endW = mrmapRT[mapkey].w2;
    ULLInt startN = mrmapRT[mapkey].n1;
    ULLInt endN = mrmapRT[mapkey].n2;

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
void Mesh::scaleRuntime(std::string mapkey, Vec3f origin, float scl) {
    if (mrmapRT.find(mapkey) == mrmapRT.end()) return;
    if (!allocated) return;

    Vec3f_ptr &w = Graphic3D::instance().mesh.v.w;

    ULLInt startW = mrmapRT[mapkey].w1;
    ULLInt endW = mrmapRT[mapkey].w2;

    ULLInt numWs = endW - startW;
    ULLInt gridSize = (numWs + 255) / 256;

    scaleMeshKernel<<<gridSize, 256>>>(
        w.x + startW, w.y + startW, w.z + startW, numWs,
        origin.x, origin.y, origin.z, scl
    );
    cudaDeviceSynchronize();
}

std::string Mesh::printRtMap() {
    std::string rt = "";
    for (std::string key : mrmapKs) {
        rt += "| -" + key + "- | " +
            std::to_string(mrmapRT[key].w1) + " - " + std::to_string(mrmapRT[key].w2) + " | " +
            std::to_string(mrmapRT[key].t1) + " - " + std::to_string(mrmapRT[key].t2) + " | " +
            std::to_string(mrmapRT[key].n1) + " - " + std::to_string(mrmapRT[key].n2) + "\n";
    }
    return rt;
}

// ======================= Vertex_ptr =======================

void Vertex_ptr::malloc(ULLInt ws, ULLInt ts, ULLInt ns) {
    s.malloc(ws);
    w.malloc(ws);
    t.malloc(ts);
    n.malloc(ns);
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
    mkd.malloc(size);
    this->size = size;
}
void Material_ptr::free() {
    ka.free();
    kd.free();
    ks.free();
    mkd.free();
}
void Material_ptr::operator+=(Material_ptr &material) {
    size += material.size;
    ka += material.ka;
    kd += material.kd;
    ks += material.ks;
    mkd += material.mkd;
}

// ======================= Texture_ptr =======================

void Texture_ptr::malloc(ULLInt tsize, ULLInt tcount) {
    tx.malloc(tsize);
    wh.malloc(tcount);
    of.malloc(tcount);
    size = tsize;
    count = tcount;
}
void Texture_ptr::free() {
    tx.free();
    wh.free();
    of.free();
    size = 0;
    count = 0;
}
void Texture_ptr::operator+=(Texture_ptr &texture) {
    size += texture.size;
    count += texture.count;
    tx += texture.tx;
    wh += texture.wh;
    of += texture.of;
}

// ======================= Mesh3D =======================

// Free
void Mesh3D::free() {
    v.free();
    f.free();
    m.free();
}

// Push
void Mesh3D::push(Mesh &mesh, bool print) {
    ULLInt offsetW = v.w.size;
    ULLInt offsetT = v.t.size;
    ULLInt offsetN = v.n.size;

    // Set the runtime values
    for (auto &kv : mesh.mrmapRT) {
        kv.second.offsetW(offsetW);
        kv.second.offsetT(offsetT);
        kv.second.offsetN(offsetN);
    }
    // Set the metadata
    mesh.allocated = true;
        // ... more later
    
    /*
    Push to the meshmap
    If name already exists, increment the repeatname counter
    And set new name: <name>_<counter>

    Example: ball -> ball_1 -> ball_2 -> ...
    */
    if (meshmap.find(mesh.name) != meshmap.end()) {
        repeatname[mesh.name]++;
        mesh.name = mesh.name + "_" + std::to_string(repeatname[mesh.name]);
        meshmap[mesh.name] = mesh;
    } else {
        repeatname[mesh.name] = 0;
        meshmap[mesh.name] = mesh;
    }

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

    // =============== Texture data ===============

    ULLInt offsetTxs = t.size;
    ULLInt offsetTxc = t.count;

    Texture_ptr newTx;
    ULLInt txSize = mesh.txr.size();
    ULLInt txCount = mesh.txw.size();
    newTx.malloc(txSize, txCount);

    cudaMemcpyAsync(newTx.tx.x, mesh.txr.data(), txSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newTx.tx.y, mesh.txg.data(), txSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newTx.tx.z, mesh.txb.data(), txSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newTx.wh.x, mesh.txw.data(), txCount * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newTx.wh.y, mesh.txh.data(), txCount * sizeof(int), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newTx.of.x, mesh.txof.data(), txCount * sizeof(LLInt), cudaMemcpyHostToDevice, stream);

    // Increment texture offset
    ULLInt gridSize = (txCount + 255) / 256;
    incLLIntKernel<<<gridSize, 256>>>(newTx.of.x, offsetTxs, txCount);
    t += newTx;

    // =============== Material data ===============

    ULLInt offsetM = m.size;

    Material_ptr newM;
    ULLInt mSize = mesh.kdr.size();
    newM.malloc(mSize);

    cudaMemcpyAsync(newM.ka.x, mesh.kar.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newM.ka.y, mesh.kag.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newM.ka.z, mesh.kab.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newM.kd.x, mesh.kdr.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newM.kd.y, mesh.kdg.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newM.kd.z, mesh.kdb.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    cudaMemcpyAsync(newM.ks.x, mesh.ksr.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newM.ks.y, mesh.ksg.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newM.ks.z, mesh.ksb.data(), mSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(newM.mkd.x, mesh.mkd.data(), mSize * sizeof(LLInt), cudaMemcpyHostToDevice, stream);

    // Increment texture indices
    gridSize = (mSize + 255) / 256;
    incLLIntKernel<<<gridSize, 256>>>(newM.mkd.x, offsetTxc, mSize);
    m += newM;

    // =============== Face data ================

    Face_ptr newF;
    ULLInt fSize = mesh.fw.size();
    newF.malloc(fSize);

    cudaMemcpyAsync(newF.v, mesh.fw.data(), fSize * sizeof(ULLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newF.t, mesh.ft.data(), fSize * sizeof(LLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newF.n, mesh.fn.data(), fSize * sizeof(LLInt), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(newF.m, mesh.fm.data(), fSize * sizeof(LLInt), cudaMemcpyHostToDevice, stream);

    // Increment face indices
    gridSize = (fSize + 255) / 256;
    incULLIntKernel<<<gridSize, 256>>>(newF.v, offsetW, fSize);
    incLLIntKernel<<<gridSize, 256>>>(newF.t, offsetT, fSize);
    incLLIntKernel<<<gridSize, 256>>>(newF.n, offsetN, fSize);
    incLLIntKernel<<<gridSize, 256>>>(newF.m, offsetM, fSize);
    f += newF;

    // Destroy the stream
    cudaStreamSynchronize(stream);
}
void Mesh3D::push(std::vector<Mesh> &meshes, bool print) {
    for (Mesh &mesh : meshes) push(mesh, print);
}

void Mesh3D::printMeshMap() {
    meshmapstr = "";
    for (auto &kv : meshmap) {
        meshmapstr += kv.first + "\n";
        meshmapstr += kv.second.printRtMap();
    }

    // Write to meshmap.txt
    std::ofstream file("meshmap.txt");
    file << meshmapstr;
}

// Kernel for incrementing face indices
__global__ void incULLIntKernel(ULLInt *f, ULLInt offset, ULLInt numFs) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFs) f[idx] += offset;
}
__global__ void incLLIntKernel(LLInt *f, ULLInt offset, ULLInt numFs) {
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
    float ox, float oy, float oz, float scl
) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numWs) {
        wx[idx] = (wx[idx] - ox) * scl + ox;
        wy[idx] = (wy[idx] - oy) * scl + oy;
        wz[idx] = (wz[idx] - oz) * scl + oz;
    }
}