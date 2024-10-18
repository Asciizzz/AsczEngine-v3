#ifndef RENDER3D_CUH
#define RENDER3D_CUH

#include <Buffer3D.cuh>

class Render3D {
public:
    // Singleton
    static Render3D& instance() {
        static Render3D instance;
        return instance;
    }
    Render3D(const Render3D&) = delete;
    Render3D &operator=(const Render3D&) = delete;

    Camera3D CAMERA;
    Mesh3D MESH;

    Vec2f RES = {800, 600};
    Vec2f RES_HALF = {400, 300};
    void setResolution(float w, float h);

private:
    Render3D() {}
};

#endif