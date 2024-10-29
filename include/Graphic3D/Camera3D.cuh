#ifndef CAMERA3D_CUH
#define CAMERA3D_CUH

// #include <Matrix.cuh>
#include <string>

#include <Plane3D.cuh>

class Camera3D {
public:
    Camera3D() {};

    // Some camera settings
    float mSens = 0.1f;
    bool focus = true;
    float slowFactor = 0.2f;
    float fastFactor = 5.0f;

    Vec3f pos, rot; // Pitch, Yaw, Roll (roll rarely used)
    void restrictRot();

    Vec3f forward, right, up;
    Mat4f view;
    void updateView();

    float fov = M_PI_2;
    float aspect = 1;
    float near = 0.1;
    float far = 43210;
    Mat4f projection;
    void updateProjection();

    Mat4f mvp;
    void updateMVP();

    // BETA: movement
    Vec3f vel;

    // BETA: camera plane
    Plane3D plane;
    void updatePlane();

    // Debug
    std::string data();
};

#endif