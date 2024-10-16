#include <MathUtil.cuh>

__host__ __device__
void MathUtil::translate3D(
    float &x, float &y, float &z,
    float dx, float dy, float dz
) {
    x += dx;
    y += dy;
    z += dz;
}

__host__ __device__
void MathUtil::rotate3D(
    float &x, float &y, float &z,
    float ox, float oy, float oz,
    float wx, float wy, float wz
) {
    // Translate the vertex to the origin
    x -= ox; y -= oy; z -= oz;

    // Rotate the vertex
    float temp_x = x;
    float temp_y = y;
    float temp_z = z;
    x = temp_x * (cos(wy) * cos(wz)) + temp_y * (cos(wz) * sin(wx) * sin(wy) - cos(wx) * sin(wz)) + temp_z * (cos(wx) * cos(wz) * sin(wy) + sin(wx) * sin(wz));
    y = temp_x * (cos(wy) * sin(wz)) + temp_y * (cos(wx) * cos(wz) + sin(wx) * sin(wy) * sin(wz)) + temp_z * (-cos(wz) * sin(wx) + cos(wx) * sin(wy) * sin(wz));
    z = temp_x * (-sin(wy)) + temp_y * cos(wy) * sin(wx) + temp_z * cos(wx) * cos(wy);

    // Translate the vertex back to the original position
    x += ox; y += oy; z += oz;
}

__host__ __device__
void MathUtil::scale3D(
    float &x, float &y, float &z,
    float ox, float oy, float oz,
    float sx, float sy, float sz
) {
    // Translate the vertex to the origin
    x -= ox; y -= oy; z -= oz;

    // Scale the vertex
    x *= sx; y *= sy; z *= sz;

    // Translate the vertex back to the original position
    x += ox; y += oy; z += oz;
}

__host__ __device__
void MathUtil::scale3D(
    float &x, float &y, float &z,
    float ox, float oy, float oz,
    float scl
) {
    scale3D(x, y, z, ox, oy, oz, scl, scl, scl);
}

// Kernel for mesh transformations

// TRANSFORMATIONS

__global__ void translateVertices(
    float *px, float *py, float *pz,
    float dx, float dy, float dz,
    uint32_t *vMeshIds,
    uint32_t meshId,
    uint32_t numVertices
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices || vMeshIds[i] != meshId) return;

    MathUtil::translate3D(px[i], py[i], pz[i], dx, dy, dz);
}

__global__ void rotateVertices(
    float *px, float *py, float *pz,
    float ox, float oy, float oz,
    float wx, float wy, float wz,
    uint32_t *vMeshIds,
    uint32_t meshId,
    uint32_t numVertices
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices || vMeshIds[i] != meshId) return;

    MathUtil::rotate3D(px[i], py[i], pz[i], ox, oy, oz, wx, wy, wz);
}

__global__ void scaleVertices(
    float *px, float *py, float *pz,
    float ox, float oy, float oz,
    float sx, float sy, float sz,
    uint32_t *vMeshIds,
    uint32_t meshId,
    uint32_t numVertices
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices || vMeshIds[i] != meshId) return;

    MathUtil::scale3D(px[i], py[i], pz[i], ox, oy, oz, sx, sy, sz);
}