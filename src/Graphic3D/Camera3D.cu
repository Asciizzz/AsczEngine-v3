#include <Camera3D.cuh>

Camera3D::Camera3D() : x(0), y(0), z(0), pitch(0), yaw(0), roll(0) {
    resizeWindow(1600, 900);
    setFov(90.0f);
}

void Camera3D::resizeWindow(int w, int h) {
    width = w;
    height = h;
    centerX = w / 2;
    centerY = h / 2;
}

void Camera3D::setFov(float f) {
    fov = f;
    screenDist = (centerX / 2) / tan(fov * M_PI / 360.0f);
}

__host__ __device__
void Camera3D::project(float x, float y, float z, float& px, float& py, float& pz) {
    float diff_x = x - this->x;
    float diff_y = y - this->y;
    float diff_z = z - this->z;

    // Apply yaw
    float cos_yaw = cos(-yaw);
    float sin_yaw = sin(-yaw);
    float temp_x = diff_x * cos_yaw + diff_z * sin_yaw;
    float temp_z = -diff_x * sin_yaw + diff_z * cos_yaw;

    // Apply pitch
    float cos_pitch = cos(-pitch);
    float sin_pitch = sin(-pitch);
    float final_y = temp_z * sin_pitch + diff_y * cos_pitch;
    float final_z = temp_z * cos_pitch - diff_y * sin_pitch;

    px = (temp_x * screenDist) / final_z;
    py = -(final_y * screenDist) / final_z;

    if (final_z < 0) {
        px *= -10;
        py *= -10;
    }

    px += centerX;
    py += centerY;
    pz = final_z;
}

void Camera3D::movement(float dTimeSec) {
    // Get the normalized direction vector
    float nx = cos(pitch) * sin(yaw);
    float ny = sin(pitch);
    float nz = cos(pitch) * cos(yaw);

    if (vel != 0) {
        x += nx * vel * dTimeSec;
        y += ny * vel * dTimeSec;
        z += nz * vel * dTimeSec;
    }
}