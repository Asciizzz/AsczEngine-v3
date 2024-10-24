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
}

void Mesh3D::resizeVertices(ULLInt numWs, ULLInt numNs, ULLInt numTs) {
    freeVertices();
    this->numWs = numWs;
    this->numNs = numNs;
    this->numTs = numTs;
    mallocVertices();
}

void Mesh3D::freeVertices() {
    if (world) cudaFree(world);
    if (normal) cudaFree(normal);
    if (texture) cudaFree(texture);
    if (screen) cudaFree(screen);
    if (color) cudaFree(color);
}

void Mesh3D::mallocFaces() {
    blockNumFs = (numFs + blockSize - 1) / blockSize;

    cudaMalloc(&faceWs, numFs * sizeof(Vec3ulli));
    cudaMalloc(&faceNs, numFs * sizeof(Vec3ulli));
    cudaMalloc(&faceTs, numFs * sizeof(Vec3ulli));
}

void Mesh3D::resizeFaces(ULLInt numFs) {
    freeFaces();
    this->numFs = numFs;
    mallocFaces();
}

void Mesh3D::freeFaces() {
    if (faceWs) cudaFree(faceWs);
    if (faceNs) cudaFree(faceNs);
    if (faceTs) cudaFree(faceTs);
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
    // Set the vertices data
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
    cudaMemcpy(newColor, color, numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    // Copy new data
    cudaMemcpy(newWorld + numWs, mesh.world, mesh.numWs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNormal + numNs, mesh.normal, mesh.numNs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTexture + numTs, mesh.texture, mesh.numTs * sizeof(Vec2f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newScreen + numWs, mesh.screen, mesh.numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newColor + numWs, mesh.color, mesh.numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    // Free old data and update
    freeVertices();
    world = newWorld;
    normal = newNormal;
    texture = newTexture;
    screen = newScreen;
    color = newColor;

    // Resize faces (with offset for the added vertices)
    ULLInt newNumFs = numFs + mesh.numFs;
    ULLInt newBlockNumFs = (newNumFs + blockSize - 1) / blockSize;

    Vec3ulli *newFaceWs;
    Vec3ulli *newFaceNs;
    Vec3ulli *newFaceTs;

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

    freeFaces();
    faceWs = newFaceWs;
    faceNs = newFaceNs;
    faceTs = newFaceTs;

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

// Kernel for incrementing face indices
__global__ void incrementFaceIdxKernel(Vec3ulli *faces, ULLInt offset, ULLInt numFs, ULLInt newNumFs) { // BETA
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < newNumFs && idx >= numFs) faces[idx] += offset;
}