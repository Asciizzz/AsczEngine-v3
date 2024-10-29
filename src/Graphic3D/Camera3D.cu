#include <Camera3D.cuh>

void Camera3D::restrictRot() {
    if (rot.x <= -M_PI_2) rot.x = -M_PI_2 + 0.001;
    else if (rot.x >= M_PI_2) rot.x = M_PI_2 - 0.001;

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
    mvp = projection * view;
}

void Camera3D::update() {
    restrictRot();

    // Model-View-Projection matrix
    updateView();
    updateProjection();
    updateMVP();

    // Camera plane
    updatePlane();
}

void Camera3D::updatePlane() {
    nearPlane = Plane3D(forward, pos + forward * near);
    farPlane = Plane3D(forward * -1, pos + forward * far);
}

// Debug
std::string Camera3D::data() {
    std::string str = "Camera3D\n";
    str += "| Pos: " + std::to_string(pos.x) + ", " + std::to_string(pos.y) + ", " + std::to_string(pos.z) + "\n";
    str += "| Rot: " + std::to_string(rot.x) + ", " + std::to_string(rot.y) + ", " + std::to_string(rot.z) + "\n";
    str += "| Fd: " + std::to_string(forward.x) + ", " + std::to_string(forward.y) + ", " + std::to_string(forward.z) + "\n";
    str += "| Rg: " + std::to_string(right.x) + ", " + std::to_string(right.y) + ", " + std::to_string(right.z) + "\n";
    str += "| Up: " + std::to_string(up.x) + ", " + std::to_string(up.y) + ", " + std::to_string(up.z) + "\n";
    str += "| Fov: " + std::to_string(fov * 180 / M_PI) + "\n";
    str += "| Asp: " + std::to_string(aspect) + "\n";
    str += "| Nrr: " + std::to_string(near) + "\n";
    str += "| Far: " + std::to_string(far) + "\n";
    return str;
}