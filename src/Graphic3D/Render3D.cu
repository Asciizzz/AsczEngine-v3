#include <Render3D.cuh>

Render3D::Render3D() {
    CAMERA = Camera3D();
    MESH = Mesh3D();
    BUFFER = Buffer3D(CAMERA.width, CAMERA.height);
}

void Render3D::toCameraProjection() {
    uint32_t numVertices = MESH.vtxs.numVtxs;
    uint32_t blockSize = 256;
    uint32_t numBlocks = (numVertices + blockSize - 1) / blockSize;

    projectVertices<<<numBlocks, blockSize>>>(
        MESH.prjs.x, MESH.prjs.y, MESH.prjs.z,
        MESH.vtxs.x, MESH.vtxs.y, MESH.vtxs.z,
        CAMERA, numVertices
    );
    cudaDeviceSynchronize();
}

__global__ void projectVertices(
    float *px, float *py, float *pz,
    const float *vx, const float *vy, const float *vz,
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

    px[i] = proj_x;
    py[i] = proj_y;
    pz[i] = final_z;
}