#ifndef MESH3D_H
#define MESH3D_H

#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint> // For fixed-width integer types

#include <MathUtil.cuh>
#include <Camera3D.cuh>

struct Vertices {
    uint32_t numVtxs;
    float *x, *y, *z; // Position
    float *nx, *ny, *nz; // Normal
    float *u, *v; // Texture coordinates
    uint32_t *meshId; // Mesh ID

    Vertices();
    void allocate(uint32_t numVertices);
    void resize(uint32_t numVertices);
    void free();
};

struct Indices {
    uint32_t numIdxs;
    uint32_t *vertexId;
    uint32_t *meshId;

    Indices();
    void allocate(uint32_t numIndices);
    void resize(uint32_t numIndices);
    void free();
};

struct Projections {
    uint32_t numVtxs;
    float *x, *y, *z;

    Projections();
    void allocate(uint32_t numVertices);
    void resize(uint32_t numVertices);
    void free();
};

class Mesh3D {
public:
    uint32_t numVtxs;
    uint32_t numIdxs;
    uint32_t meshId;

    Vertices vtxs;
    Indices idxs;
    Projections prjs;

    Mesh3D(uint32_t vertexCount=0, uint32_t indexCount=0, uint32_t meshId=0);
    ~Mesh3D();

    void allocate(uint32_t vertexCount, uint32_t indexCount);
    void free();

    void uploadVertices(const std::vector<float>& h_x,
                        const std::vector<float>& h_y,
                        const std::vector<float>& h_z,
                        const std::vector<float>& h_nx,
                        const std::vector<float>& h_ny,
                        const std::vector<float>& h_nz,
                        const std::vector<float>& h_u,
                        const std::vector<float>& h_v);
    void uploadIndices(const std::vector<uint32_t>& h_indices);
    void upload(const std::vector<float>& h_x,
                const std::vector<float>& h_y,
                const std::vector<float>& h_z,
                const std::vector<float>& h_nx,
                const std::vector<float>& h_ny,
                const std::vector<float>& h_nz,
                const std::vector<float>& h_u,
                const std::vector<float>& h_v,
                const std::vector<uint32_t>& h_indices);

    // Append more data AND increment the indices
    void operator+=(const Mesh3D& mesh);
    void operator=(const Mesh3D& mesh);

    // Static methods for mesh transformations
    static void translate(Mesh3D &MESH, uint32_t meshId, float dx, float dy, float dz);
    static void rotate(Mesh3D &MESH, uint32_t meshId, float ox, float oy, float oz, float wx, float wy, float wz);
    static void scale(Mesh3D &MESH, uint32_t meshId, float ox, float oy, float oz, float sx, float sy, float sz);

    // Transformations
    void translate(float dx, float dy, float dz);
    void rotate(float ox, float oy, float oz, float wx, float wy, float wz);
    void scale(float ox, float oy, float oz, float sx, float sy, float sz);

    // Debugging
    void printVtxs();
    void printIdxs();
    void printPrjs();
};

// Kernel for mapping the indices
__global__ void incrementVertexId(uint32_t* indices, uint32_t numIndices, uint32_t offset);
__global__ void setMeshId(uint32_t* ids, uint32_t numIds, uint32_t meshId);
#endif