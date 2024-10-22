#include <Mesh3D.cuh>

// Constructor

Mesh3D::Mesh3D(ULLInt numWs, ULLInt numNs, ULLInt numTs, ULLInt numFs) :
    numWs(numWs), numNs(numNs), numTs(numTs), numFs(numFs)
{
    mallocVertices();
    mallocFaces();
}

Mesh3D::Mesh3D(
    UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color,
    Vecs3ulli &faceWs, Vecs3ulli &faceNs, Vecs3ulli &faceTs
) : numWs(world.size()), numNs(normal.size()), numTs(texture.size()), numFs(faceWs.size())
{
    mallocVertices();
    mallocFaces();
    uploadData(id, world, normal, texture, color, faceWs, faceNs, faceTs);
}
Mesh3D::Mesh3D(
    UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, 
    Vecs3ulli &faceAll
) : numWs(world.size()), numNs(normal.size()), numTs(texture.size()), numFs(faceAll.size())
{
    mallocVertices();
    mallocFaces();
    uploadData(id, world, normal, texture, color, faceAll, faceAll, faceAll);
}

// Memory management

void Mesh3D::mallocVertices() {
    blockNumWs = (numWs + blockSize - 1) / blockSize;
    blockNumNs = (numNs + blockSize - 1) / blockSize;
    blockNumTs = (numTs + blockSize - 1) / blockSize;

    cudaMalloc(&world, numWs * sizeof(Vec3f));
    cudaMalloc(&normal, numNs * sizeof(Vec3f));
    cudaMalloc(&texture, numTs * sizeof(Vec2f));
    cudaMalloc(&screen, numWs * sizeof(Vec4f));

    cudaMalloc(&color, numWs * sizeof(Vec4f));

    cudaMalloc(&wObjId, numWs * sizeof(UInt));
    cudaMalloc(&nObjId, numNs * sizeof(UInt));
    cudaMalloc(&tObjId, numTs * sizeof(UInt));
}

void Mesh3D::resizeVertices(ULLInt numWs, ULLInt numNs, ULLInt numTs) {
    freeVertices();
    this->numWs = numWs;
    this->numNs = numNs;
    this->numTs = numTs;
    mallocVertices();
}

void Mesh3D::freeVertices() {
    cudaFree(world);
    cudaFree(normal);
    cudaFree(texture);
    cudaFree(screen);
    cudaFree(color);
    cudaFree(wObjId);
    cudaFree(nObjId);
    cudaFree(tObjId);
}

void Mesh3D::mallocFaces() {
    blockNumFs = (numFs + blockSize - 1) / blockSize;

    cudaMalloc(&faceWs, numFs * sizeof(Vec3ulli));
    cudaMalloc(&faceNs, numFs * sizeof(Vec3ulli));
    cudaMalloc(&faceTs, numFs * sizeof(Vec3ulli));

    cudaMalloc(&d_numVisibFs, sizeof(ULLInt));
    cudaMemset(d_numVisibFs, 0, sizeof(ULLInt));
    cudaMalloc(&visibFWs, numFs * sizeof(Vec3ulli));
    cudaMalloc(&visibFNs, numFs * sizeof(Vec3ulli));
    cudaMalloc(&visibFTs, numFs * sizeof(Vec3ulli));

}

void Mesh3D::resizeFaces(ULLInt numFs) {
    freeFaces();
    this->numFs = numFs;
    mallocFaces();
}

void Mesh3D::freeFaces() {
    cudaFree(faceWs);
    cudaFree(faceNs);
    cudaFree(faceTs);

    cudaFree(d_numVisibFs);
    cudaFree(visibFWs);
    cudaFree(visibFNs);
    cudaFree(visibFTs);
}

void Mesh3D::free() {
    freeVertices();
    freeFaces();
}

// Upload host data to device

void Mesh3D::uploadData(
    UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color,
    Vecs3ulli &faceWs, Vecs3ulli &faceNs, Vecs3ulli &faceTs
) {
    // Vertices

    // Set obj Id for each vertex attribute
    setObjIdKernel<<<blockNumWs, blockSize>>>(this->wObjId, numWs, id);
    setObjIdKernel<<<blockNumNs, blockSize>>>(this->nObjId, numNs, id);
    setObjIdKernel<<<blockNumTs, blockSize>>>(this->tObjId, numTs, id);

    // Set the actual data
    cudaMemcpy(this->world, world.data(), world.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(this->normal, normal.data(), normal.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(this->texture, texture.data(), texture.size() * sizeof(Vec2f), cudaMemcpyHostToDevice);

    cudaMemcpy(this->color, color.data(), color.size() * sizeof(Vec4f), cudaMemcpyHostToDevice);

    // Faces indices
    cudaMemcpy(this->faceWs, faceWs.data(), faceWs.size() * sizeof(Vec3ulli), cudaMemcpyHostToDevice);
    cudaMemcpy(this->faceNs, faceNs.data(), faceNs.size() * sizeof(Vec3ulli), cudaMemcpyHostToDevice);
    cudaMemcpy(this->faceTs, faceTs.data(), faceTs.size() * sizeof(Vec3ulli), cudaMemcpyHostToDevice);
}

// Mesh operators

void Mesh3D::operator+=(Mesh3D &mesh) {
    // Resize vertices
    ULLInt newNumWs = numWs + mesh.numWs;
    ULLInt newNumNs = numNs + mesh.numNs;
    ULLInt newNumTs = numTs + mesh.numTs;
    Vec3f *newWorld;
    Vec3f *newNormal;
    Vec2f *newTexture;
    Vec4f *newScreen;
    UInt *newWObjId;
    UInt *newNObjId;
    UInt *newTObjId;
    Vec4f *newColor;
    cudaMalloc(&newWorld, newNumWs * sizeof(Vec3f));
    cudaMalloc(&newNormal, newNumNs * sizeof(Vec3f));
    cudaMalloc(&newTexture, newNumTs * sizeof(Vec2f));
    cudaMalloc(&newScreen, newNumWs * sizeof(Vec4f));
    cudaMalloc(&newWObjId, newNumWs * sizeof(UInt));
    cudaMalloc(&newNObjId, newNumNs * sizeof(UInt));
    cudaMalloc(&newTObjId, newNumTs * sizeof(UInt));
    cudaMalloc(&newColor, newNumWs * sizeof(Vec4f));
    // Copy old data
    cudaMemcpy(newWorld, world, numWs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNormal, normal, numNs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTexture, texture, numTs * sizeof(Vec2f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newScreen, screen, numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newWObjId, wObjId, numWs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNObjId, nObjId, numNs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTObjId, tObjId, numTs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newColor, color, numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    // Copy new data
    cudaMemcpy(newWorld + numWs, mesh.world, mesh.numWs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNormal + numNs, mesh.normal, mesh.numNs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTexture + numTs, mesh.texture, mesh.numTs * sizeof(Vec2f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newScreen + numWs, mesh.screen, mesh.numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newWObjId + numWs, mesh.wObjId, mesh.numWs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNObjId + numNs, mesh.nObjId, mesh.numNs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTObjId + numTs, mesh.tObjId, mesh.numTs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newColor + numWs, mesh.color, mesh.numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    // Free old data and update
    freeVertices();
    world = newWorld;
    normal = newNormal;
    texture = newTexture;
    screen = newScreen;
    wObjId = newWObjId;
    nObjId = newNObjId;
    tObjId = newTObjId;
    color = newColor;

    // Resize faces (with offset for the added vertices)
    ULLInt newNumFs = numFs + mesh.numFs;
    ULLInt newBlockNumFs = (newNumFs + blockSize - 1) / blockSize;

    Vec3ulli *newFaceWs;
    Vec3ulli *newFaceNs;
    Vec3ulli *newFaceTs;

    ULLInt *newDNumVisibleFs;
    Vec3ulli *newVisibleFWs;
    Vec3ulli *newVisibleFNs;
    Vec3ulli *newVisibleFTs;

    cudaMalloc(&newFaceWs, newNumFs * sizeof(Vec3ulli));
    cudaMalloc(&newFaceNs, newNumFs * sizeof(Vec3ulli));
    cudaMalloc(&newFaceTs, newNumFs * sizeof(Vec3ulli));

    cudaMemcpy(newFaceWs, faceWs, numFs * sizeof(Vec3ulli), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newFaceNs, faceNs, numFs * sizeof(Vec3ulli), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newFaceTs, faceTs, numFs * sizeof(Vec3ulli), cudaMemcpyDeviceToDevice);

    cudaMemcpy(newFaceWs + numFs, mesh.faceWs, mesh.numFs * sizeof(Vec3ulli), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newFaceNs + numFs, mesh.faceNs, mesh.numFs * sizeof(Vec3ulli), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newFaceTs + numFs, mesh.faceTs, mesh.numFs * sizeof(Vec3ulli), cudaMemcpyDeviceToDevice);
    incrementFaceIdxKernel<<<newBlockNumFs, blockSize>>>(newFaceWs, numWs, numFs, newNumFs);
    incrementFaceIdxKernel<<<newBlockNumFs, blockSize>>>(newFaceNs, numNs, numFs, newNumFs);
    incrementFaceIdxKernel<<<newBlockNumFs, blockSize>>>(newFaceTs, numTs, numFs, newNumFs);

    cudaMalloc(&newDNumVisibleFs, sizeof(ULLInt));
    cudaMalloc(&newVisibleFWs, newNumFs * sizeof(Vec3ulli));
    cudaMalloc(&newVisibleFNs, newNumFs * sizeof(Vec3ulli));
    cudaMalloc(&newVisibleFTs, newNumFs * sizeof(Vec3ulli));

    freeFaces();
    faceWs = newFaceWs;
    faceNs = newFaceNs;
    faceTs = newFaceTs;
    d_numVisibFs = newDNumVisibleFs;
    visibFWs = newVisibleFWs;
    visibFNs = newVisibleFNs;
    visibFTs = newVisibleFTs;

    // Update number of vertices and faces
    numWs = newNumWs;
    numNs = newNumNs;
    numTs = newNumTs;
    numFs = newNumFs;
    blockNumWs = (numWs + blockSize - 1) / blockSize;
    blockNumNs = (numNs + blockSize - 1) / blockSize;
    blockNumTs = (numTs + blockSize - 1) / blockSize;
    blockNumFs = (numFs + blockSize - 1) / blockSize;
}

// Transformations (with obj Id)

void Mesh3D::translate(UInt objID, Vec3f t) {
    translateWorldKernel<<<blockNumWs, blockSize>>>(world, wObjId, false, numWs, objID, t);
}
void Mesh3D::rotate(UInt objID, Vec3f origin, Vec3f rot) {
    rotateWorldKernel<<<blockNumWs, blockSize>>>(world, wObjId, false, numWs, objID, origin, rot);
    rotateNormalKernel<<<blockNumNs, blockSize>>>(normal, nObjId, false, numNs, objID, origin, rot);
}
void Mesh3D::scale(UInt objID, Vec3f origin, Vec3f scl) {
    scaleWorldKernel<<<blockNumWs, blockSize>>>(world, wObjId, false, numWs, objID, origin, scl);
    scaleNormalKernel<<<blockNumNs, blockSize>>>(normal, nObjId, false, numNs, objID, origin, scl);
}

// Transformations (all obj Ids)

void Mesh3D::translate(Vec3f t) {
    translateWorldKernel<<<blockNumWs, blockSize>>>(world, wObjId, true, numWs, 0, t);
}
void Mesh3D::rotate(Vec3f origin, Vec3f rot) {
    rotateWorldKernel<<<blockNumWs, blockSize>>>(world, wObjId, true, numWs, 0, origin, rot);
    rotateNormalKernel<<<blockNumNs, blockSize>>>(normal, nObjId, true, numNs, 0, origin, rot);
}
void Mesh3D::scale(Vec3f origin, Vec3f scl) {
    scaleWorldKernel<<<blockNumWs, blockSize>>>(world, wObjId, true, numWs, 0, origin, scl);
    scaleNormalKernel<<<blockNumNs, blockSize>>>(normal, nObjId, true, numNs, 0, origin, scl);
}

// Kernel for incrementing face indices
__global__ void incrementFaceIdxKernel(Vec3ulli *faces, ULLInt offset, ULLInt numFs, ULLInt newNumFs) { // BETA
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < newNumFs && idx >= numFs) faces[idx] += offset;
}

// Kernel for preparing vertices
__global__ void setObjIdKernel(UInt *objId, ULLInt numWs, UInt id) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWs) objId[idx] = id;
}

// Kernel for transforming vertices
__global__ void translateWorldKernel(Vec3f *world, UInt *wObjId, bool allId, ULLInt numWs, UInt objID, Vec3f t) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs || (!allId && wObjId[idx] != objID)) return;

    world[idx].translate(t);
}
__global__ void rotateWorldKernel(Vec3f *world, UInt *wObjId, bool allId, ULLInt numWs, UInt objID, Vec3f origin, Vec3f rot) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs || (!allId && wObjId[idx] != objID)) return;

    world[idx].rotate(origin, rot);
}
__global__ void scaleWorldKernel(Vec3f *world, UInt *wObjId, bool allId, ULLInt numWs, UInt objID, Vec3f origin, Vec3f scl) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs || (!allId && wObjId[idx] != objID)) return;

    world[idx].scale(origin, scl);
}

// Rotate and scale normals
__global__ void rotateNormalKernel(Vec3f *normal, UInt *nObjId, bool allID, ULLInt numNs, UInt objID, Vec3f origin, Vec3f rot) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNs || (!allID && nObjId[idx] != objID)) return;

    normal[idx].rotate(origin, rot);
}
__global__ void scaleNormalKernel(Vec3f *normal, UInt *nObjId, bool allID, ULLInt numNs, UInt objID, Vec3f origin, Vec3f scl) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNs || (!allID && nObjId[idx] != objID)) return;

    normal[idx].scale(origin, scl);
}