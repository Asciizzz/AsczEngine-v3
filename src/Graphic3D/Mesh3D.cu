#include <Mesh3D.cuh>

// Mesh struct

Mesh::Mesh(UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, Vecs3x3uli &faces) :
    meshId(world.size(), id), world(world), normal(normal),
    texture(texture), color(color), faces(faces)
{}

Mesh::Mesh(Mesh &mesh, UInt id) :
    meshId(mesh.world.size(), id), world(mesh.world), normal(mesh.normal),
    texture(mesh.texture), color(mesh.color), faces(mesh.faces)
{}

Mesh Mesh::operator+=(Mesh &mesh) {
    ULLInt oldSize = world.size();

    meshId.insert(meshId.end(), mesh.meshId.begin(), mesh.meshId.end());
    world.insert(world.end(), mesh.world.begin(), mesh.world.end());
    normal.insert(normal.end(), mesh.normal.begin(), mesh.normal.end());
    texture.insert(texture.end(), mesh.texture.begin(), mesh.texture.end());
    color.insert(color.end(), mesh.color.begin(), mesh.color.end());
    faces.insert(faces.end(), mesh.faces.begin(), mesh.faces.end());

    // Shift the faces indices
    for (ULLInt i = oldSize; i < faces.size(); i++) {
        faces[i].v += oldSize;
        faces[i].t += oldSize;
        faces[i].n += oldSize;
    }
    return *this;
}

Mesh Mesh::operator+(Mesh mesh) {
    Mesh newMesh = *this;
    newMesh += mesh;
    return newMesh;
}

// Constructor

Mesh3D::Mesh3D(ULLInt numWs, ULLInt numNs, ULLInt numTs, ULLInt numFs) :
    numWs(numWs), numNs(numNs), numTs(numTs), numFs(numFs)
{
    mallocVertices();
    mallocFaces();
}

Mesh3D::Mesh3D(UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, Vecs3x3uli &faces) :
    numWs(world.size()), numNs(normal.size()), numTs(texture.size()), numFs(faces.size())
{
    mallocVertices();
    mallocFaces();
    uploadData(id, world, normal, texture, color, faces);
}

Mesh3D::Mesh3D(Mesh &mesh) :
    numWs(mesh.world.size()),
    numNs(mesh.normal.size()),
    numTs(mesh.texture.size()),
    numFs(mesh.faces.size())
{
    mallocVertices();
    mallocFaces();
    uploadData(mesh.meshId[0], mesh.world, mesh.normal, mesh.texture, mesh.color, mesh.faces);
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

    cudaMalloc(&wMeshId, numWs * sizeof(UInt));
    cudaMalloc(&nMeshId, numNs * sizeof(UInt));
    cudaMalloc(&tMeshId, numTs * sizeof(UInt));

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
    cudaFree(world);
    cudaFree(normal);
    cudaFree(texture);
    cudaFree(screen);

    cudaFree(wMeshId);
    cudaFree(nMeshId);
    cudaFree(tMeshId);

    cudaFree(color);
}

void Mesh3D::mallocFaces() {
    blockNumFs = (numFs + blockSize - 1) / blockSize;
    cudaMalloc(&faces, numFs * sizeof(Vec3x3uli));
}

void Mesh3D::resizeFaces(ULLInt numFs) {
    freeFaces();
    this->numFs = numFs;
    mallocFaces();
}

void Mesh3D::freeFaces() {
    cudaFree(faces);
}

void Mesh3D::free() {
    freeVertices();
    freeFaces();
}

// Upload host data to device

void Mesh3D::uploadData(UInt id, Vecs3f &world, Vecs3f &normal, Vecs2f &texture, Vecs4f &color, Vecs3x3uli &faces) {
    // Vertices

    // Set mesh ID for each vertex attribute
    setMeshIdKernel<<<blockNumWs, blockSize>>>(this->wMeshId, numWs, id);
    setMeshIdKernel<<<blockNumNs, blockSize>>>(this->nMeshId, numNs, id);
    setMeshIdKernel<<<blockNumTs, blockSize>>>(this->tMeshId, numTs, id);

    // Set the actual data
    cudaMemcpy(this->world, world.data(), world.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(this->normal, normal.data(), normal.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(this->texture, texture.data(), texture.size() * sizeof(Vec2f), cudaMemcpyHostToDevice);
    
    cudaMemcpy(this->color, color.data(), color.size() * sizeof(Vec4f), cudaMemcpyHostToDevice);
    
    // Faces indices
    cudaMemcpy(this->faces, faces.data(), faces.size() * sizeof(Vec3x3uli), cudaMemcpyHostToDevice);
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
    UInt *newWMeshId;
    UInt *newNMeshId;
    UInt *newTMeshId;
    Vec4f *newColor;
    cudaMalloc(&newWorld, newNumWs * sizeof(Vec3f));
    cudaMalloc(&newNormal, newNumNs * sizeof(Vec3f));
    cudaMalloc(&newTexture, newNumTs * sizeof(Vec2f));
    cudaMalloc(&newScreen, newNumWs * sizeof(Vec4f));
    cudaMalloc(&newWMeshId, newNumWs * sizeof(UInt));
    cudaMalloc(&newNMeshId, newNumNs * sizeof(UInt));
    cudaMalloc(&newTMeshId, newNumTs * sizeof(UInt));
    cudaMalloc(&newColor, newNumWs * sizeof(Vec4f));
    // Copy old data
    cudaMemcpy(newWorld, world, numWs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNormal, normal, numNs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTexture, texture, numTs * sizeof(Vec2f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newScreen, screen, numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newWMeshId, wMeshId, numWs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNMeshId, nMeshId, numNs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTMeshId, tMeshId, numTs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newColor, color, numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    // Copy new data
    cudaMemcpy(newWorld + numWs, mesh.world, mesh.numWs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNormal + numNs, mesh.normal, mesh.numNs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTexture + numTs, mesh.texture, mesh.numTs * sizeof(Vec2f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newScreen + numWs, mesh.screen, mesh.numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newWMeshId + numWs, mesh.wMeshId, mesh.numWs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNMeshId + numNs, mesh.nMeshId, mesh.numNs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTMeshId + numTs, mesh.tMeshId, mesh.numTs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newColor + numWs, mesh.color, mesh.numWs * sizeof(Vec4f), cudaMemcpyDeviceToDevice);
    // Free old data
    freeVertices();
    // Update vertices
    world = newWorld;
    normal = newNormal;
    texture = newTexture;
    screen = newScreen;
    wMeshId = newWMeshId;
    nMeshId = newNMeshId;
    tMeshId = newTMeshId;
    color = newColor;

    // Resize faces (with offset for the added vertices)
    ULLInt newNumFs = numFs + mesh.numFs;
    ULLInt newBlockNumFs = (newNumFs + blockSize - 1) / blockSize;
    Vec3x3uli *newFaces;
    cudaMalloc(&newFaces, newNumFs * sizeof(Vec3x3uli));
    cudaMemcpy(newFaces, faces, numFs * sizeof(Vec3x3uli), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newFaces + numFs, mesh.faces, mesh.numFs * sizeof(Vec3x3uli), cudaMemcpyDeviceToDevice);
    incrementFaceIdxKernel<<<newBlockNumFs, blockSize>>>(newFaces, numWs, numNs, numTs, numFs, newNumFs);

    cudaFree(faces);
    faces = newFaces;

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

// Transformations (with mesh ID)

void Mesh3D::translate(UInt mID, Vec3f t) {
    translateWorldKernel<<<blockNumWs, blockSize>>>(world, wMeshId, false, numWs, mID, t);
}
void Mesh3D::rotate(UInt mID, Vec3f origin, Vec3f rot) {
    rotateWorldKernel<<<blockNumWs, blockSize>>>(world, wMeshId, false, numWs, mID, origin, rot);
    rotateNormalKernel<<<blockNumNs, blockSize>>>(normal, nMeshId, false, numNs, mID, origin, rot);
}
void Mesh3D::scale(UInt mID, Vec3f origin, Vec3f scl) {
    scaleWorldKernel<<<blockNumWs, blockSize>>>(world, wMeshId, false, numWs, mID, origin, scl);
    scaleNormalKernel<<<blockNumNs, blockSize>>>(normal, nMeshId, false, numNs, mID, origin, scl);
}

// Transformations (all mesh IDs)

void Mesh3D::translate(Vec3f t) {
    translateWorldKernel<<<blockNumWs, blockSize>>>(world, wMeshId, true, numWs, 0, t);
}
void Mesh3D::rotate(Vec3f origin, Vec3f rot) {
    rotateWorldKernel<<<blockNumWs, blockSize>>>(world, wMeshId, true, numWs, 0, origin, rot);
    rotateNormalKernel<<<blockNumNs, blockSize>>>(normal, nMeshId, true, numNs, 0, origin, rot);
}
void Mesh3D::scale(Vec3f origin, Vec3f scl) {
    scaleWorldKernel<<<blockNumWs, blockSize>>>(world, wMeshId, true, numWs, 0, origin, scl);
    scaleNormalKernel<<<blockNumNs, blockSize>>>(normal, nMeshId, true, numNs, 0, origin, scl);
}

// Kernel for preparing vertices
__global__ void incrementFaceIdxKernel(Vec3x3uli *faces, ULLInt offsetW, ULLInt offsetN, ULLInt offsetT, ULLInt numFs, ULLInt newNumFs) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < newNumFs && idx >= numFs) {
        faces[idx].v += offsetW;
        faces[idx].n += offsetN;
        faces[idx].t += offsetT;
    }
}

__global__ void setMeshIdKernel(UInt *meshId, ULLInt numWs, UInt id) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWs) meshId[idx] = id;
}

// Kernel for transforming vertices
__global__ void translateWorldKernel(Vec3f *world, UInt *wMeshId, bool allId, ULLInt numWs, UInt mID, Vec3f t) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs || (!allId && wMeshId[idx] != mID)) return;

    world[idx].translate(t);
}
__global__ void rotateWorldKernel(Vec3f *world, UInt *wMeshId, bool allId, ULLInt numWs, UInt mID, Vec3f origin, Vec3f rot) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs || (!allId && wMeshId[idx] != mID)) return;

    world[idx].rotate(origin, rot);
}
__global__ void scaleWorldKernel(Vec3f *world, UInt *wMeshId, bool allId, ULLInt numWs, UInt mID, Vec3f origin, Vec3f scl) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWs || (!allId && wMeshId[idx] != mID)) return;

    world[idx].scale(origin, scl);
}

// Rotate and scale normals
__global__ void rotateNormalKernel(Vec3f *normal, UInt *nMeshId, bool allID, ULLInt numNs, UInt mID, Vec3f origin, Vec3f rot) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNs || (!allID && nMeshId[idx] != mID)) return;

    normal[idx].rotate(origin, rot);
}
__global__ void scaleNormalKernel(Vec3f *normal, UInt *nMeshId, bool allID, ULLInt numNs, UInt mID, Vec3f origin, Vec3f scl) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNs || (!allID && nMeshId[idx] != mID)) return;

    normal[idx].scale(origin, scl);
}