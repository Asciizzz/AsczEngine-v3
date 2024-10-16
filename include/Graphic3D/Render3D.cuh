#ifndef RENDER3D_CUH
#define RENDER3D_CUH

#include <Buffer3D.cuh>

class Render3D {
public:
    // Singleton
    static Render3D &instance() {
        static Render3D instance;
        return instance;
    }
    Render3D(const Render3D&) = delete;
    Render3D &operator=(const Render3D&) = delete;

    Camera3D CAMERA;
    Mesh3D MESH;
    Buffer3D BUFFER;

    void toCameraProjection();

private:
    Render3D() {};
};

// BETA: Kernel for projecting the vertices
__global__ void projectVertices(
    // Projected vertices
    float *px, float *py, float *pz,
    float *pnx, float *pny, float *pnz,
    float *pu, float *pv,
    // Original vertices
    const float *vx, const float *vy, const float *vz,
    const float *vnx, const float *vny, const float *vnz,
    const float *vu, const float *vv,
    // Other parameters
    Camera3D camera, uint32_t numVertices
);

#endif