#ifndef CAMERA3D_CUH
#define CAMERA3D_CUH

#include <Matrix.cuh>

class Camera3D {
public:
    Camera3D();

    float mSens = 0.1f;
    bool focus = true;

    Vec3f pos, rot; // Pitch, Yaw, Roll (roll rarely used)
    void restrictRot();

    Vec3f forward, right, up;
    Mat4f view;
    void updateView();

    float fov = M_PI_2, aspect = 1;
    float near = 0.5, far = 500;
    Mat4f projection;
    void updateProjection();

    Mat4f mvp;
    void updateMVP();

    // Frustum culling check
    bool isInsideFrustum(Vec3f &v);
};

#endif