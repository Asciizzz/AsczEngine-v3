#include <Camera3D.cuh>

Camera3D::Camera3D() {}

void Camera3D::restrictRot() {
    if (rot.x < -M_PI_2) rot.x = -M_PI_2;
    else if (rot.x > M_PI_2) rot.x = M_PI_2;

    if (rot.y > M_2_PI) rot.y -= M_2_PI;
    else if (rot.y < 0) rot.y += M_2_PI;
}

void Camera3D::updateView() {
    forward.x = cos(rot.y) * cos(rot.x);
    forward.y = sin(rot.x);
    forward.z = sin(rot.y) * cos(rot.x);
    forward.norm();

    right = forward & Vec3f(0, 1, 0);
    right.norm();

    up = right & forward;
    up.norm();

    // Translation matrix
    float tMat[4][4] = {
        {1, 0, 0, -pos.x},
        {0, 1, 0, -pos.y},
        {0, 0, 1, -pos.z},
        {0, 0, 0, 1}
    };

    // Rotation matrix
    float rMat[4][4] = {
        {right.x, right.y, right.z, 0},
        {up.x, up.y, up.z, 0},
        {-forward.x, -forward.y, -forward.z, 0},
        {0, 0, 0, 1}
    };

    view = Mat4f(rMat) * Mat4f(tMat);
}