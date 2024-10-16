#include <Render3D.cuh>

void Render3D::toCameraProjection() {
    uint32_t numVertices = MESH.vtxs.numVtxs;
    uint32_t blockSize = 256;
    uint32_t numBlocks = (numVertices + blockSize - 1) / blockSize;

    projectVertices<<<numBlocks, blockSize>>>(
        MESH.prjs.x, MESH.prjs.y, MESH.prjs.z,
        MESH.prjs.nx, MESH.prjs.ny, MESH.prjs.nz,
        MESH.prjs.u, MESH.prjs.v,
        MESH.vtxs.x, MESH.vtxs.y, MESH.vtxs.z,
        MESH.vtxs.nx, MESH.vtxs.ny, MESH.vtxs.nz,
        MESH.vtxs.u, MESH.vtxs.v,
        CAMERA, numVertices
    );
    cudaDeviceSynchronize();
}

__global__ void projectVertices(
    float *px, float *py, float *pz,
    float *pnx, float *pny, float *pnz,
    float *pu, float *pv,
    const float *vx, const float *vy, const float *vz,
    const float *vnx, const float *vny, const float *vnz,
    const float *vu, const float *vv,
    Camera3D camera, uint32_t numVertices
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    float diff_x = vx[i] - camera.x;
    float diff_y = vy[i] - camera.y;
    float diff_z = vz[i] - camera.z;

    // Apply yaw
    float cos_yaw = cos(-camera.yaw);
    float sin_yaw = sin(-camera.yaw);
    float temp_x = diff_x * cos_yaw + diff_z * sin_yaw;
    float temp_z = -diff_x * sin_yaw + diff_z * cos_yaw;

    // Apply pitch
    float cos_pitch = cos(-camera.pitch);
    float sin_pitch = sin(-camera.pitch);
    float final_y = temp_z * sin_pitch + diff_y * cos_pitch;
    float final_z = temp_z * cos_pitch - diff_y * sin_pitch;

    float proj_x = (temp_x * camera.screenDist) / final_z;
    float proj_y = -(final_y * camera.screenDist) / final_z;

    if (final_z < 0) {
        proj_x *= -10;
        proj_y *= -10;
    }

    proj_x += camera.centerX;
    proj_y += camera.centerY;

    px[i] = proj_x; py[i] = proj_y; pz[i] = final_z;
    pnx[i] = vnx[i]; pny[i] = vny[i]; pnz[i] = vnz[i];
    pu[i] = vu[i]; pv[i] = vv[i];
}