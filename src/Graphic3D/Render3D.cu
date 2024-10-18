#include <Render3D.cuh>

void Render3D::setResolution(float w, float h) {
    res = {w, h};
    res_half = {w / 2, h / 2};
    camera.setResolution(w, h);
    buffer.resize(w, h, pixel_size);
}

void Render3D::allocateProjection() {
    cudaMalloc(&projection, mesh.numVs * sizeof(Vec4f));
}
void Render3D::freeProjection() {
    if (projection) cudaFree(projection);
}
void Render3D::resizeProjection() {
    freeProjection();
    allocateProjection();
}

// Pipeline

void Render3D::vertexProjection() {
    vertexProjectionKernel<<<mesh.blockNumVs, mesh.blockSize>>>(
        projection, mesh.world, camera, mesh.numVs
    );
    cudaDeviceSynchronize();
}

// Kernels

__global__ void vertexProjectionKernel(Vec4f *projection, Vec3f *world, Camera3D camera, ULLInt numVs) {
    ULLInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVs) return;

    Vec4f v4 = world[i].toVec4f();
    Vec4f t4 = camera.mvp * v4;
    Vec3f t3 = t4.toVec3f();

    // Screen space 
    t3.x = (t3.x + 1) * camera.res.x / 2;
    t3.y = (1 - t3.y) * camera.res.y / 2;
    t3.z = camera.near + (camera.far - camera.near) * t3.z;

    Vec4f p = t3.toVec4f();
    p.w = camera.isInsideFrustum(world[i]) ? 1 : 0;

    projection[i] = p;
}