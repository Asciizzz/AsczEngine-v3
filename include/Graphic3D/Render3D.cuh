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

    Vec2f RES = {800, 600};
    Vec2f RES_HALF = {400, 300};
    int PIXEL_SIZE = 4;
    void setResolution(float w, float h);

    Mesh3D MESH;
    Camera3D CAMERA;
    Buffer3D BUFFER;

private:
    Render3D() {}
};

#endif