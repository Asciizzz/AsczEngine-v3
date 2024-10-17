#include <Mesh3D.cuh>

// Constructor

Mesh3D::Mesh3D(ULLInt numVs, ULLInt numFs) :
    numVs(numVs), numFs(numFs)
{
    mallocVertices();
    mallocFaces();
}

Mesh3D::Mesh3D(
    UInt id,
    Vecs3f &pos,
    Vecs3f &normal,
    Vecs2f &tex,
    Vecs3uli &faces
) :
    numVs(pos.size()), numFs(faces.size())
{
    mallocVertices();
    mallocFaces();
    uploadData(id, pos, normal, tex, faces);
}

// Memory management

void Mesh3D::mallocVertices() {
    cudaMalloc(&pos, numVs * sizeof(Vec3f));
    cudaMalloc(&normal, numVs * sizeof(Vec3f));
    cudaMalloc(&tex, numVs * sizeof(Vec2f));
    cudaMalloc(&mID, numVs * sizeof(UInt));
}

void Mesh3D::resizeVertices(ULLInt numVs) {
    freeVertices();
    this->numVs = numVs;
    mallocVertices();
}

void Mesh3D::freeVertices() {
    cudaFree(pos);
    cudaFree(normal);
    cudaFree(tex);
    cudaFree(mID);
}

void Mesh3D::mallocFaces() {
    cudaMalloc(&faces, numFs * sizeof(Vec3uli));
}

void Mesh3D::resizeFaces(ULLInt numFs) {
    freeFaces();
    this->numFs = numFs;
    mallocFaces();
}

void Mesh3D::freeFaces() {
    cudaFree(faces);
}

// Upload host data to device

void Mesh3D::uploadData(
    UInt id,
    Vecs3f &pos,
    Vecs3f &normal,
    Vecs2f &tex,
    Vecs3uli &faces
) {
    if (pos.size() != numVs ||
        normal.size() != numVs ||
        tex.size() != numVs ||
        faces.size() != numFs
    ) {
        std::cerr << "Error: Data size mismatch" << std::endl;
        return;
    }

    setMeshIDKernel<<<(numVs + 255) / 256, 256>>>(this->mID, numVs, id);
    cudaMemcpy(this->pos, pos.data(), pos.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(this->normal, normal.data(), normal.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(this->tex, tex.data(), tex.size() * sizeof(Vec2f), cudaMemcpyHostToDevice);
    cudaMemcpy(this->faces, faces.data(), faces.size() * sizeof(Vec3uli), cudaMemcpyHostToDevice);
}

// Mesh operators

void Mesh3D::operator+=(Mesh3D &mesh) {
    // Resize vertices
    ULLInt newNumVs = numVs + mesh.numVs;
    Vec3f *newPos;
    Vec3f *newNormal;
    Vec2f *newTex;
    UInt *newMID;
    cudaMalloc(&newPos, newNumVs * sizeof(Vec3f));
    cudaMalloc(&newNormal, newNumVs * sizeof(Vec3f));
    cudaMalloc(&newTex, newNumVs * sizeof(Vec2f));
    cudaMalloc(&newMID, newNumVs * sizeof(UInt));
    cudaMemcpy(newPos, pos, numVs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNormal, normal, numVs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTex, tex, numVs * sizeof(Vec2f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newMID, mID, numVs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newPos + numVs, mesh.pos, mesh.numVs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newNormal + numVs, mesh.normal, mesh.numVs * sizeof(Vec3f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newTex + numVs, mesh.tex, mesh.numVs * sizeof(Vec2f), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newMID + numVs, mesh.mID, mesh.numVs * sizeof(UInt), cudaMemcpyDeviceToDevice);
    freeVertices();
    pos = newPos;
    normal = newNormal;
    tex = newTex;
    mID = newMID;

    // Resize faces (with offset for the added vertices)
    ULLInt newNumFs = numFs + mesh.numFs;
    Vec3uli *newFaces;
    cudaMalloc(&newFaces, newNumFs * sizeof(Vec3uli));
    cudaMemcpy(newFaces, faces, numFs * sizeof(Vec3uli), cudaMemcpyDeviceToDevice);
    cudaMemcpy(newFaces + numFs, mesh.faces, mesh.numFs * sizeof(Vec3uli), cudaMemcpyDeviceToDevice);
    incrementFaceIdxKernel<<<(mesh.numFs + 255) / 256, 256>>>(newFaces, mesh.numFs, numVs);
    cudaFree(faces);
    faces = newFaces;

    // Update number of vertices and faces
    numVs = newNumVs;
    numFs = newNumFs;
}

// DEBUG

void Mesh3D::printVertices(bool p_pos, bool p_normal, bool p_tex, bool p_mID) {
    Vec3f *hPos = new Vec3f[numVs];
    Vec3f *hNormal = new Vec3f[numVs];
    Vec2f *hTex = new Vec2f[numVs];
    UInt *hMID = new UInt[numVs];

    cudaMemcpy(hPos, pos, numVs * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(hNormal, normal, numVs * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(hTex, tex, numVs * sizeof(Vec2f), cudaMemcpyDeviceToHost);
    cudaMemcpy(hMID, mID, numVs * sizeof(UInt), cudaMemcpyDeviceToHost);

    for (ULLInt i = 0; i < numVs; i++) {
        printf("Vertex %llu\n", i);
        if (p_pos) printf("| Pos: (%f, %f, %f)\n", hPos[i].x, hPos[i].y, hPos[i].z);
        if (p_normal) printf("| Normal: (%f, %f, %f)\n", hNormal[i].x, hNormal[i].y, hNormal[i].z);
        if (p_tex) printf("| Tex: (%f, %f)\n", hTex[i].x, hTex[i].y);
        if (p_mID) printf("| mID: %u\n", hMID[i]);
    }

    delete[] hPos;
    delete[] hNormal;
    delete[] hTex;
    delete[] hMID;
}

void Mesh3D::printFaces() {
    Vec3uli *hFaces = new Vec3uli[numFs];
    cudaMemcpy(hFaces, faces, numFs * sizeof(Vec3uli), cudaMemcpyDeviceToHost);

    for (ULLInt i = 0; i < numFs; i++) {
        printf("Face %lu: (%lu, %lu, %lu)\n", i, hFaces[i].x, hFaces[i].y, hFaces[i].z);
    }

    delete[] hFaces;
}

// Kernel for preparing vertices
__global__ void incrementFaceIdxKernel(Vec3uli *faces, ULLInt numFs, ULLInt offset) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numFs) faces[idx] += offset;
}

__global__ void setMeshIDKernel(UInt *mID, ULLInt numVs, UInt id) {
    ULLInt idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVs) return;

    mID[idx] = id;
}