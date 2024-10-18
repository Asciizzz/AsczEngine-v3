#ifndef CAMERA3D_CUH
#define CAMERA3D_CUH

#include <Matrix.cuh>

class Camera3D {
public:
    Camera3D();

    Vec3f pos;
    Vec3f rot; // Pitch, Yaw, Roll (roll rarely used)
    void restrictRot();

    float fov, aspect;
    float near, far;

    Vec3f forward;
    Vec3f right;
    Vec3f up;
    Mat4f view;
    void updateView();
};

#endif