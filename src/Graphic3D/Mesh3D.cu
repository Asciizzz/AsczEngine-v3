#include <Mesh3D.cuh>

// Mesh object

Mesh::Mesh(
    Vecs3f world, Vecs3f normal, Vecs2f texture, Vecs4f color,
    ULLInts faceWs, ULLInts faceTs, ULLInts faceNs
) : world(world), normal(normal), texture(texture), color(color),
    faceWs(faceWs), faceTs(faceTs), faceNs(faceNs),
    numWs(world.size()),
    numNs(normal.size()),
    numTs(texture.size()),
    numFs(faceWs.size() / 3) {}

Mesh::Mesh(
    Vecs3f world, Vecs3f normal, Vecs2f texture, Vecs4f color, ULLInts faceSame
) : world(world), normal(normal), texture(texture), color(color),
    faceWs(faceSame), faceTs(faceSame), faceNs(faceSame),
    numWs(world.size()),
    numNs(normal.size()),
    numTs(texture.size()),
    numFs(faceSame.size() / 3) {}

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
void Mesh3D::resizeVertices(ULLInt numWs, ULLInt numNs, ULLInt numTs) {
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
    // Append faces
    Vecptr4ulli newFaces;
    newFaces.malloc(mesh.numFs * 3);
    for (ULLInt i = 0; i < mesh.numFs * 3; i++) {
        cudaMemcpy(&newFaces.v[i], &mesh.faceWs[i], sizeof(ULLInt), cudaMemcpyHostToDevice);
        cudaMemcpy(&newFaces.t[i], &mesh.faceTs[i], sizeof(ULLInt), cudaMemcpyHostToDevice);
        cudaMemcpy(&newFaces.n[i], &mesh.faceNs[i], sizeof(ULLInt), cudaMemcpyHostToDevice);
        cudaMemcpy(&newFaces.o[i], 0, sizeof(ULLInt), cudaMemcpyHostToDevice);
    }

    // Increment face indices
    ULLInt offsetV = world.size;
    ULLInt offsetT = texture.size;
    ULLInt offsetN = normal.size;

    ULLInt gridSize = (mesh.numFs * 3 + 255) / 256;
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFaces.v, offsetV, mesh.numFs * 3);
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFaces.t, offsetT, mesh.numFs * 3);
    incrementFaceIdxKernel<<<gridSize, 256>>>(newFaces.n, offsetN, mesh.numFs * 3);

    faces += newFaces;

    // Append vertices
    Vecptr3f newWorld;
    Vecptr3f newNormal;
    Vecptr2f newTexture;
    Vecptr4f newColor;
    Vecptr4f newScreen;
    newWorld.malloc(mesh.numWs);
    newNormal.malloc(mesh.numNs);
    newTexture.malloc(mesh.numTs);
    newColor.malloc(mesh.numWs);
    newScreen.malloc(mesh.numWs);

    // NOTE: THE ENTIRE PROCESS BELOW IS INCREDIBLY SLOW

    std::vector<float> worldX(mesh.numWs);
    std::vector<float> worldY(mesh.numWs);
    std::vector<float> worldZ(mesh.numWs);
    for (ULLInt i = 0; i < mesh.numWs; i++) {
        worldX[i] = mesh.world[i].x;
        worldY[i] = mesh.world[i].y;
        worldZ[i] = mesh.world[i].z;
    }
    cudaMemcpy(newWorld.x, worldX.data(), mesh.numWs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newWorld.y, worldY.data(), mesh.numWs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newWorld.z, worldZ.data(), mesh.numWs * sizeof(float), cudaMemcpyHostToDevice);
    world += newWorld;

    std::vector<float> normalX(mesh.numNs);
    std::vector<float> normalY(mesh.numNs);
    std::vector<float> normalZ(mesh.numNs);
    for (ULLInt i = 0; i < mesh.numNs; i++) {
        normalX[i] = mesh.normal[i].x;
        normalY[i] = mesh.normal[i].y;
        normalZ[i] = mesh.normal[i].z;
    }
    cudaMemcpy(newNormal.x, normalX.data(), mesh.numNs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newNormal.y, normalY.data(), mesh.numNs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newNormal.z, normalZ.data(), mesh.numNs * sizeof(float), cudaMemcpyHostToDevice);
    normal += newNormal;

    std::vector<float> textureX(mesh.numTs);
    std::vector<float> textureY(mesh.numTs);
    for (ULLInt i = 0; i < mesh.numTs; i++) {
        textureX[i] = mesh.texture[i].x;
        textureY[i] = mesh.texture[i].y;
    }
    cudaMemcpy(newTexture.x, textureX.data(), mesh.numTs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newTexture.y, textureY.data(), mesh.numTs * sizeof(float), cudaMemcpyHostToDevice);
    texture += newTexture;

    std::vector<float> colorX(mesh.numWs);
    std::vector<float> colorY(mesh.numWs);
    std::vector<float> colorZ(mesh.numWs);
    std::vector<float> colorW(mesh.numWs);
    for (ULLInt i = 0; i < mesh.numWs; i++) {
        colorX[i] = mesh.color[i].x;
        colorY[i] = mesh.color[i].y;
        colorZ[i] = mesh.color[i].z;
        colorW[i] = mesh.color[i].w;
    }
    cudaMemcpy(newColor.x, colorX.data(), mesh.numWs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newColor.y, colorY.data(), mesh.numWs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newColor.z, colorZ.data(), mesh.numWs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(newColor.w, colorW.data(), mesh.numWs * sizeof(float), cudaMemcpyHostToDevice);
    color += newColor;

    screen.free();
    screen.malloc(world.size);
}

// Kernel for incrementing face indices
__global__ void incrementFaceIdxKernel(ULLInt *f, ULLInt offset, ULLInt numFs) { // BETA
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFs) f[idx] += offset;
}