#include <Mesh3D.cuh>
#include <Render3D.cuh>

// VERTICES

Vertices::Vertices() :
    x(nullptr), y(nullptr), z(nullptr),
    nx(nullptr), ny(nullptr), nz(nullptr),
    u(nullptr), v(nullptr), meshId(nullptr) {}

void Vertices::allocate(uint32_t numVertices) {
    numVtxs = numVertices;
    cudaMalloc(&x, numVertices * sizeof(float));
    cudaMalloc(&y, numVertices * sizeof(float));
    cudaMalloc(&z, numVertices * sizeof(float));
    cudaMalloc(&nx, numVertices * sizeof(float));
    cudaMalloc(&ny, numVertices * sizeof(float));
    cudaMalloc(&nz, numVertices * sizeof(float));
    cudaMalloc(&u, numVertices * sizeof(float));
    cudaMalloc(&v, numVertices * sizeof(float));
    cudaMalloc(&meshId, numVertices * sizeof(uint32_t));
}

void Vertices::resize(uint32_t numVertices) {
    float *newX, *newY, *newZ, *newNX, *newNY, *newNZ, *newU, *newV;
    uint32_t *newMeshId;
    cudaMalloc(&newX, numVertices * sizeof(float));
    cudaMalloc(&newY, numVertices * sizeof(float));
    cudaMalloc(&newZ, numVertices * sizeof(float));
    cudaMalloc(&newNX, numVertices * sizeof(float));
    cudaMalloc(&newNY, numVertices * sizeof(float));
    cudaMalloc(&newNZ, numVertices * sizeof(float));
    cudaMalloc(&newU, numVertices * sizeof(float));
    cudaMalloc(&newV, numVertices * sizeof(float));
    cudaMalloc(&newMeshId, numVertices * sizeof(uint32_t));

    cudaMemcpy(newX, x, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newY, y, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newZ, z, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNX, nx, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNY, ny, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNZ, nz, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newU, u, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newV, v, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newMeshId, meshId, numVtxs * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    free();
    numVtxs = numVertices;
    x = newX; y = newY; z = newZ;
    nx = newNX; ny = newNY; nz = newNZ;
    u = newU; v = newV; meshId = newMeshId;
}

void Vertices::free() {
    if (x) cudaFree(x);
    if (y) cudaFree(y);
    if (z) cudaFree(z);
    if (nx) cudaFree(nx);
    if (ny) cudaFree(ny);
    if (nz) cudaFree(nz);
    if (u) cudaFree(u);
    if (v) cudaFree(v);
    if (meshId) cudaFree(meshId);
}

// INDICES

Indices::Indices() : vertexId(nullptr), meshId(nullptr) {}

void Indices::allocate(uint32_t numIndices) {
    numIdxs = numIndices;
    cudaMalloc(&vertexId, numIndices * sizeof(uint32_t));
    cudaMalloc(&meshId, numIndices * sizeof(uint32_t));
}

void Indices::resize(uint32_t numIndices) {
    uint32_t *newVertexId, *newMeshId;

    cudaMalloc(&newVertexId, numIndices * sizeof(uint32_t));
    cudaMalloc(&newMeshId, numIndices * sizeof(uint32_t));

    cudaMemcpy(newVertexId, vertexId, numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newMeshId, meshId, numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    free();
    numIdxs = numIndices;
    vertexId = newVertexId;
    meshId = newMeshId;
}

void Indices::free() {
    if (vertexId) cudaFree(vertexId);
    if (meshId) cudaFree(meshId);
}

// PROJECTIONS

Projections::Projections() : x(nullptr), y(nullptr), z(nullptr) {}

void Projections::allocate(uint32_t numVertices) {
    numVtxs = numVertices;
    cudaMalloc(&x, numVertices * sizeof(float));
    cudaMalloc(&y, numVertices * sizeof(float));
    cudaMalloc(&z, numVertices * sizeof(float));
}

void Projections::resize(uint32_t numVertices) {
    float *newX, *newY, *newZ;
    cudaMalloc(&newX, numVertices * sizeof(float));
    cudaMalloc(&newY, numVertices * sizeof(float));
    cudaMalloc(&newZ, numVertices * sizeof(float));

    cudaMemcpy(newX, x, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newY, y, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newZ, z, numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);

    free();
    numVtxs = numVertices;
    x = newX; y = newY; z = newZ;
}

void Projections::free() {
    if (x) cudaFree(x);
    if (y) cudaFree(y);
    if (z) cudaFree(z);
}

// MESH3D

Mesh3D::Mesh3D(uint32_t vertexCount, uint32_t indexCount, uint32_t meshId) :
    numVtxs(vertexCount),
    numIdxs(indexCount),
    meshId(meshId)
{
    allocate(vertexCount, indexCount);
}

Mesh3D::~Mesh3D() {
    free();
}

void Mesh3D::allocate(uint32_t vertexCount, uint32_t indexCount) {
    vtxs.allocate(vertexCount);
    idxs.allocate(indexCount);
    prjs.allocate(vertexCount);
}

void Mesh3D::free() {
    vtxs.free();
    idxs.free();
    prjs.free();
}

void Mesh3D::uploadVertices(const std::vector<float>& h_x,
                            const std::vector<float>& h_y,
                            const std::vector<float>& h_z,
                            const std::vector<float>& h_nx,
                            const std::vector<float>& h_ny,
                            const std::vector<float>& h_nz,
                            const std::vector<float>& h_u,
                            const std::vector<float>& h_v) 
{
    cudaMemcpy(vtxs.x, h_x.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vtxs.y, h_y.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vtxs.z, h_z.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vtxs.nx, h_nx.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vtxs.ny, h_ny.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vtxs.nz, h_nz.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vtxs.u, h_u.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vtxs.v, h_v.data(), numVtxs * sizeof(float), cudaMemcpyHostToDevice);
}

void Mesh3D::uploadIndices(const std::vector<uint32_t>& h_indices) {
    cudaMemcpy(idxs.vertexId, h_indices.data(), numIdxs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    setMeshId<<<(numIdxs + 255) / 256, 256>>>(idxs.meshId, numIdxs, meshId);
    setMeshId<<<(numVtxs + 255) / 256, 256>>>(vtxs.meshId, numVtxs, meshId);
}

void Mesh3D::upload(const std::vector<float>& h_x,
                    const std::vector<float>& h_y,
                    const std::vector<float>& h_z,
                    const std::vector<float>& h_nx,
                    const std::vector<float>& h_ny,
                    const std::vector<float>& h_nz,
                    const std::vector<float>& h_u,
                    const std::vector<float>& h_v,
                    const std::vector<uint32_t>& h_indices) {
    uploadVertices(h_x, h_y, h_z, h_nx, h_ny, h_nz, h_u, h_v);
    uploadIndices(h_indices);
}

void Mesh3D::operator+=(const Mesh3D& mesh) {
    uint32_t newVertexCount = numVtxs + mesh.numVtxs;
    uint32_t newIndexCount = numIdxs + mesh.numIdxs;

    vtxs.resize(newVertexCount);
    idxs.resize(newIndexCount);
    prjs.resize(newVertexCount);

    cudaMemcpy(vtxs.x + numVtxs, mesh.vtxs.x, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.y + numVtxs, mesh.vtxs.y, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.z + numVtxs, mesh.vtxs.z, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.nx + numVtxs, mesh.vtxs.nx, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.ny + numVtxs, mesh.vtxs.ny, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.nz + numVtxs, mesh.vtxs.nz, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.u + numVtxs, mesh.vtxs.u, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.v + numVtxs, mesh.vtxs.v, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaMemcpy(idxs.vertexId + numIdxs, mesh.idxs.vertexId, mesh.numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(idxs.meshId + numIdxs, mesh.idxs.meshId, mesh.numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    incrementVertexId<<<(mesh.numIdxs + 255) / 256, 256>>>(
        idxs.vertexId + numIdxs, mesh.numIdxs, numVtxs
    );
    setMeshId<<<(mesh.numIdxs + 255) / 256, 256>>>(
        idxs.meshId + numIdxs, mesh.numIdxs, mesh.meshId
    );
    setMeshId<<<(numIdxs + 255) / 256, 256>>>(
        vtxs.meshId + numVtxs, mesh.numVtxs, mesh.meshId
    );
    cudaDeviceSynchronize();

    numVtxs = newVertexCount;
    numIdxs = newIndexCount;
}

void Mesh3D::operator=(const Mesh3D& mesh) {
    numVtxs = mesh.numVtxs;
    numIdxs = mesh.numIdxs;

    free();
    allocate(mesh.numVtxs, mesh.numIdxs);

    cudaMemcpy(vtxs.x, mesh.vtxs.x, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.y, mesh.vtxs.y, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.z, mesh.vtxs.z, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.nx, mesh.vtxs.nx, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.ny, mesh.vtxs.ny, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.nz, mesh.vtxs.nz, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.u, mesh.vtxs.u, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vtxs.v, mesh.vtxs.v, mesh.numVtxs * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaMemcpy(idxs.vertexId, mesh.idxs.vertexId, mesh.numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
}

// Static transformations
void Mesh3D::translate(Mesh3D &MESH, uint32_t meshId, float dx, float dy, float dz) {
    translateVertices<<<(MESH.numVtxs + 255) / 256, 256>>>(
        MESH.vtxs.x, MESH.vtxs.y, MESH.vtxs.z,
        dx, dy, dz,
        MESH.vtxs.meshId,
        meshId,
        MESH.numVtxs
    );
    cudaDeviceSynchronize();
}

void Mesh3D::rotate(Mesh3D &MESH, uint32_t meshId, float ox, float oy, float oz, float wx, float wy, float wz) {
    rotateVertices<<<(MESH.numVtxs + 255) / 256, 256>>>(
        MESH.vtxs.x, MESH.vtxs.y, MESH.vtxs.z,
        ox, oy, oz, wx, wy, wz,
        MESH.vtxs.meshId,
        meshId,
        MESH.numVtxs
    );
    cudaDeviceSynchronize();
}

void Mesh3D::scale(Mesh3D &MESH, uint32_t meshId, float ox, float oy, float oz, float sx, float sy, float sz) {
    scaleVertices<<<(MESH.numVtxs + 255) / 256, 256>>>(
        MESH.vtxs.x, MESH.vtxs.y, MESH.vtxs.z,
        ox, oy, oz, sx, sy, sz,
        MESH.vtxs.meshId,
        meshId,
        MESH.numVtxs
    );
    cudaDeviceSynchronize();
}

// Transformations
void Mesh3D::translate(float dx, float dy, float dz) {
    Render3D &RENDER = Render3D::instance();
    Mesh3D::translate(RENDER.MESH, meshId, dx, dy, dz);
}
void Mesh3D::rotate(float ox, float oy, float oz, float wx, float wy, float wz) {
    Render3D &RENDER = Render3D::instance();
    Mesh3D::rotate(RENDER.MESH, meshId, ox, oy, oz, wx, wy, wz);
}
void Mesh3D::scale(float ox, float oy, float oz, float sx, float sy, float sz) {
    Render3D &RENDER = Render3D::instance();
    Mesh3D::scale(RENDER.MESH, meshId, ox, oy, oz, sx, sy, sz);
}

void Mesh3D::printVtxs() {
    float *x = new float[numVtxs];
    float *y = new float[numVtxs];
    float *z = new float[numVtxs];
    uint32_t *meshId = new uint32_t[numVtxs];

    cudaMemcpy(x, vtxs.x, numVtxs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, vtxs.y, numVtxs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, vtxs.z, numVtxs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(meshId, vtxs.meshId, numVtxs * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < numVtxs; i++) {
        std::cout << "Vertex " << i << ": " << x[i] << ", " << y[i] << ", " << z[i] << " (" << meshId[i] << ")" << std::endl;
    }

    delete[] x, y, z, meshId;
}

void Mesh3D::printIdxs() {
    uint32_t *vertexIds = new uint32_t[numIdxs];
    uint32_t *meshIds = new uint32_t[numIdxs];

    cudaMemcpy(vertexIds, idxs.vertexId, numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(meshIds, idxs.meshId, numIdxs * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < numIdxs; i++) {
        std::cout << "Index " << i << ": " << vertexIds[i] << " (" << meshIds[i] << ")" << std::endl;
    }

    delete[] vertexIds, meshIds;
}

void Mesh3D::printPrjs() {
    float *x = new float[numVtxs];
    float *y = new float[numVtxs];
    float *z = new float[numVtxs];

    cudaMemcpy(x, prjs.x, numVtxs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, prjs.y, numVtxs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, prjs.z, numVtxs * sizeof(float), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < numVtxs; i++) {
        std::cout << "Projected " << i << ": (" << x[i] << ", " << y[i] << ", " << z[i] << ")" << std::endl;
    }

    delete[] x, y, z;
}

// KERNELS
__global__ void incrementVertexId(uint32_t* indices, uint32_t numIndices, uint32_t offset) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numIndices) indices[i] += offset;
}
__global__ void setMeshId(uint32_t* ids, uint32_t numIds, uint32_t meshId) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numIds) ids[i] = meshId;
}