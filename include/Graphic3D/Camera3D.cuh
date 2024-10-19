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

    float aspect = 1;
    Vec2f res = {800, 600};
    void setResolution(float w, float h);
    float fov = M_PI_2;
    float near = 0.5;
    float far = 1000;
    Mat4f projection;
    void updateProjection();

    Mat4f mvp;
    void updateMVP();

    // Frustum culling check
    __host__ __device__ bool isInsideFrustum(Vec3f &v);
};

#endif