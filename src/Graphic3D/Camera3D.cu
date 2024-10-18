#include <Camera3D.cuh>

Camera3D::Camera3D() {}

void Camera3D::restrictRot() {
    if (rot.x < -M_PI_2) rot.x = -M_PI_2;
    else if (rot.x > M_PI_2) rot.x = M_PI_2;

    if (rot.y > M_2_PI) rot.y -= M_2_PI;
    else if (rot.y < 0) rot.y += M_2_PI;
}

void Camera3D::updateView() {
    forward.x = sin(rot.y) * cos(rot.x);
    forward.y = sin(rot.x);
    forward.z = cos(rot.y) * cos(rot.x);
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

void Camera3D::updateProjection() {
    float f = 1 / tan(fov / 2);
    float ar = aspect;

    float p22 = (far + near) / (near - far);
    float p23 = (2 * far * near) / (near - far);

    float pMat[4][4] = {
        {f / ar, 0, 0, 0},
        {0, f, 0, 0},
        {0, 0, p22, p23},
        {0, 0, -1, 0}
    };
    projection = Mat4f(pMat);
}

void Camera3D::updateMVP() {
    updateView();
    updateProjection();

    mvp = projection * view;
}

bool Camera3D::isInsideFrustum(Vec3f &v) {
    Vec4f v4 = mvp * v.toVec4f();

    return  v4.x >= -v4.w && v4.x <= v4.w &&
            v4.y >= -v4.w && v4.y <= v4.w &&
            v4.z >= 0 && v4.z <= v4.w;
}